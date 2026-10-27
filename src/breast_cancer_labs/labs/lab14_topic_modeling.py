"""Practical 14, variant 1: topic modeling of real scientific abstracts."""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import RAW_DIR, markdown_table, save_environment

SCHOLARLY_STOPWORDS = {"study", "studies", "results", "result", "research", "paper", "using", "used",
                       "use", "based", "method", "methods", "analysis", "data", "proposed",
                       "different", "new", "also", "however", "can", "may", "et", "al", "abstract"}

# Human interpretations, keyed by model and topic count; populated from saved top words.
TOPIC_NAMES: dict[tuple[str, int], list[str]] = {
    ("LDA", 3): ["Табиғи орта және инженерлік үдерістер", "Медицина және молекулалық биология", "Қоғамдық денсаулық пен білім"],
    ("NMF", 3): ["Биологиялық түрлер және өсім", "Клиникалық медицина және денсаулық", "Энергетика және жылу техникасы"],
    ("LDA", 5): ["Жылу, су және қоршаған орта (аралас)", "Пациенттер және клиникалық қауіп факторлары",
                 "Білім, дене салмағы және материалдар (аралас)", "Энергия және төзімділік (аралас)",
                 "Қоғамдық денсаулық және биологиялық түрлер (аралас)"],
    ("NMF", 5): ["Жылу техникасы және кеуекті материалдар", "Клиникалық болжам және емдеу",
                 "Биологиялық түрлер және таралу", "Денсаулық сақтау және қоғамдық қажеттіліктер",
                 "Халықаралық және тарихи қатынастар"],
}


def restore_abstract(inverted: dict) -> str:
    positions = {int(position): word for word, offsets in inverted.items() for position in offsets}
    return " ".join(positions[position] for position in sorted(positions))


def clean_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z]{3,}", text.lower()))


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab14_topic_modeling")
    payload = json.loads((RAW_DIR / "scientific_abstracts_openalex.json").read_text())
    records = []
    for item in payload["results"]:
        inverted = item.get("abstract_inverted_index")
        if inverted:
            records.append({"openalex_id": item["id"], "doi": item.get("doi"), "title": item["title"],
                            "source_field": ((item.get("primary_topic") or {}).get("field") or {}).get("display_name", "unknown"),
                            "abstract": restore_abstract(inverted)})
    corpus = pd.DataFrame(records)
    corpus["clean_text"] = corpus.abstract.map(clean_text)
    # OpenAlex's language metadata mislabels two non-English abstracts in this sample.
    # Check common English function words; the excluded records were inspected manually.
    english = corpus.abstract.map(lambda s: len(re.findall(r"\b(?:the|and|of|to|in|is|with|for)\b", s.lower())) >= 5)
    empty = corpus.clean_text.eq("")
    duplicate = corpus.clean_text.duplicated()
    cleaning = pd.DataFrame([{"received": len(payload["results"]), "with_abstract": len(corpus),
                              "empty_after_cleaning": int(empty.sum()), "duplicates": int(duplicate.sum()),
                              "non_english_excluded": int((~english).sum())}])
    corpus.loc[~english].to_csv(output / "excluded_language.csv", index=False)
    corpus = corpus.loc[~empty & ~duplicate & english].reset_index(drop=True).rename_axis("document_id")
    if len(corpus) < 40:
        raise ValueError("The assignment requires at least 40 nonempty unique abstracts")
    stopwords = sorted(set(ENGLISH_STOP_WORDS) | SCHOLARLY_STOPWORDS)
    vectorizer = CountVectorizer(stop_words=stopwords, min_df=2, max_df=.9)
    counts = vectorizer.fit_transform(corpus.clean_text)
    tfidf = TfidfTransformer().fit_transform(counts)
    words = vectorizer.get_feature_names_out()
    if np.asarray(counts.sum(axis=1)).min() == 0:
        raise ValueError("A document has no retained terms")
    corpus.to_csv(output / "corpus.csv")
    cleaning.to_csv(output / "cleaning_summary.csv", index=False)
    pd.DataFrame(counts.toarray(), index=corpus.index, columns=words).to_csv(output / "bag_of_words.csv")
    pd.DataFrame(tfidf.toarray(), index=corpus.index, columns=words).to_csv(output / "tfidf.csv")
    models, mixtures, topic_rows, comparison, examples = {}, {}, [], [], []
    for k in [3, 5]:
        for method, matrix, model in [
            ("LDA", counts, LatentDirichletAllocation(n_components=k, random_state=42, learning_method="batch", max_iter=60)),
            ("NMF", tfidf, NMF(n_components=k, init="nndsvda", random_state=42, max_iter=1000)),
        ]:
            weights = model.fit_transform(matrix)
            # NMF weights are not probabilities; normalize only for visualization.
            totals = weights.sum(axis=1, keepdims=True)
            normalized = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
            names = TOPIC_NAMES.get((method, k), [f"Тақырып {j+1}" for j in range(k)])
            models[method, k], mixtures[method, k] = model, normalized
            comparison.append({"method": method, "topics": k, "documents": matrix.shape[0], "terms": matrix.shape[1],
                               "train_perplexity": model.perplexity(counts) if method == "LDA" else np.nan,
                               "tfidf_reconstruction_error": model.reconstruction_err_ if method == "NMF" else np.nan,
                               "iterations": model.n_iter_, "zero_topic_rows": int((totals == 0).sum())})
            pd.DataFrame(weights, index=corpus.index, columns=[f"topic_{j+1}" for j in range(k)]).to_csv(output / f"{method.lower()}_{k}_raw_weights.csv")
            pd.DataFrame(normalized, index=corpus.index, columns=[f"topic_{j+1}" for j in range(k)]).to_csv(output / f"{method.lower()}_{k}_document_topics.csv")
            pd.DataFrame(model.components_, columns=words).rename_axis("topic_zero_based").to_csv(output / f"{method.lower()}_{k}_topic_terms.csv")
            for j, component in enumerate(model.components_):
                indices = np.argsort(component)[-7:][::-1]
                top_words = ", ".join(words[indices])
                topic_rows.append({"method": method, "topics": k, "topic": j + 1,
                                   "kazakh_name": names[j], "top_words": top_words})
                for doc in np.argsort(normalized[:, j])[-2:][::-1]:
                    examples.append({"method": method, "topics": k, "topic": j+1, "kazakh_name": names[j],
                                     "document_id": int(doc), "weight": normalized[doc, j],
                                     "title": corpus.loc[doc, "title"], "openalex_id": corpus.loc[doc, "openalex_id"],
                                     "abstract_excerpt": corpus.loc[doc, "abstract"][:600]})
    topics, comparison, examples = pd.DataFrame(topic_rows), pd.DataFrame(comparison), pd.DataFrame(examples)
    topics.to_csv(output / "topics.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    examples.to_csv(output / "topic_document_examples.csv", index=False)
    similarities = cosine_similarity(models["LDA", 5].components_, models["NMF", 5].components_)
    left, right = linear_sum_assignment(-similarities)
    alignment = pd.DataFrame({"lda_topic": left+1, "nmf_topic": right+1, "term_cosine_similarity": similarities[left, right]})
    alignment.to_csv(output / "topic_alignment.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for row, method in enumerate(["LDA", "NMF"]):
        for j, component in enumerate(models[method, 5].components_):
            indices = np.argsort(component)[-7:]
            axes[row, j].barh(words[indices], component[indices])
            axes[row, j].set_title(f"{method}: topic {j+1}")
    save_figure(fig, output / "topic_words.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    for ax, method in zip(axes, ["LDA", "NMF"]):
        plot = ax.imshow(mixtures[method, 5], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set(xticks=range(5), xticklabels=range(1, 6), xlabel="Topic", ylabel="Document ID", title=f"{method}: normalized topic weights")
        fig.colorbar(plot, ax=ax)
    save_figure(fig, output / "document_topics.png")
    example_text = []
    for method in ["LDA", "NMF"]:
        for topic in [1, 2]:
            subset = examples[(examples.method == method) & (examples.topics == 5) & (examples.topic == topic)]
            topic_record = topics[(topics.method == method) & (topics.topics == 5) & (topics.topic == topic)].iloc[0]
            example_text.append(f"### {method}, тақырып {topic}: {topic_record.kazakh_name}\n\nКлючевые слова: {topic_record.top_words}.\n")
            for row in subset.itertuples():
                example_text.append(f"Документ {row.document_id}, вес {row.weight:.4f}: [{row.title}]({row.openalex_id}). "
                                    f"Фрагмент аннотации: «{row.abstract_excerpt}…».\n")
    report = f"""# Практическая работа №14 — вариант 1

Тема: скрытые темы научных статей по аннотациям.
[Задание](../../Data_Mining_Practica_14.docx).

## Корпус и обработка

Использованы **{len(corpus)} реальных англоязычных аннотаций научных статей 2024 года** из
[OpenAlex](https://docs.openalex.org/api-entities/works/work-object#abstract_inverted_index).
Запрос: type=article, language=en, has_abstract=true, publication_year=2024,
sample=60, seed=42, per-page=60. Ответ сохранён в [исходный JSON](../../data/raw/scientific_abstracts_openalex.json);
идентификаторы, DOI, названия и восстановленные аннотации — в [corpus.csv](corpus.csv).
Темы первичного каталога сохранены только как справка и не передаются в модель.

{markdown_table(cleaning)}

Текст приведён к нижнему регистру, оставлены английские буквенные токены длиной от 3,
Метаданные языка OpenAlex дополнительно проверены по английским служебным словам
и просмотром исключённых записей: две неанглийские аннотации исключены и сохранены отдельно.
Текст нормализован, удалены английские и общенаучные стоп-слова; min_df=2, max_df=.9.
Флексии не лемматизированы: формы слов могут остаться разными признаками — это ограничение.
Словарь общий: Bag-of-Words и TF-IDF имеют размер **{counts.shape}**.
Матрицы сохранены отдельно. Искусственные тексты и размножение документов не использовались.

## LDA и NMF: три и пять тем

LDA обучается на счётчиках слов (batch, 60 итераций), NMF — на TF-IDF
(nndsvda, до 1000 итераций); random_state=42.
Корпус описывается целиком, показатели ниже рассчитаны **на обучающем корпусе**
и не являются оценкой обобщения на новые статьи.

{markdown_table(comparison)}

Perplexity LDA и ошибка реконструкции NMF измеряют разные задачи и не сравниваются между собой.
При большем числе компонент ошибка реконструкции обычно падает; это само по себе
не доказывает лучшую интерпретируемость. Для одинакового подробного сравнения показаны пять тем,
но это не утверждение оптимальности k=5. У LDA perplexity для k=3 ниже, чем для k=5,
а пятикомпонентные темы сильнее смешивают направления: heat/water и energy/resistance.
NMF при k=5 дополнительно отделяет международные отношения от медицины и биологии;
поэтому для обзорной навигации здесь предпочтительнее NMF, с обязательным чтением примеров.
Человеческие названия ниже — интерпретации слов и документов, а не известные классы.

{markdown_table(topics)}

![Ключевые слова пяти тем](topic_words.png)
![Документ–тема](document_topics.png)

LDA даёт вероятностные доли тем. NMF даёт неотрицательные веса; их нормировка до суммы 1
используется только для изображения и не превращает их в вероятности.
Номера тем между методами не совпадают; сопоставление по косинусному сходству
словарных весов с оптимальным взаимно однозначным соответствием:

{markdown_table(alignment)}

## Примеры двух тем каждого метода

{chr(10).join(example_text)}

В LDA тема 1 смешивает рекультивацию почв и теплоперенос, то есть её научное содержание
неоднородно; тема 2 объединяет клинические факторы принятия катетеризации и риска ожирения.
В NMF тема 1 связывает пористые сорбенты и моделирование теплового потока,
а тема 2 — исходы химиотерапии и прогноз смертности при циррозе.
Эти пары подтверждают, что «тема» может объединять разные прикладные задачи по общей терминологии.
Высокий вес означает близость к найденной теме, а не истинный класс статьи.
Сопоставлять следует слова и содержание аннотаций: один номер темы у разных методов
не означает одинаковое научное направление.

## Вывод

Из реальных аннотаций построены частотная и TF-IDF матрицы, проверены LDA/NMF с 3 и 5 темами.
LDA даёт вероятностные смеси, но k=5 увеличил обучающую perplexity и дал смешанные темы.
NMF с пятью компонентами полезнее для обзорной навигации по характерным терминам,
хотя широкие и смешанные темы требуют чтения статей и не заменяют экспертную классификацию.
Ограничения — малая выборка, один год публикаций, отсутствие лемматизации и проверка одного seed.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab14_topic_modeling`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [RAW_DIR / "scientific_abstracts_openalex.json"], documents=len(corpus),
                     topic_counts=[3, 5], selected_topics=5, min_df=2, max_df=.9, language="en")
    print(topics.to_string(index=False))


if __name__ == "__main__":
    main()
