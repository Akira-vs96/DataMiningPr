"""Lab 07: Apriori and FP-Growth on discretized clinical records."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.data import load_clean_dataset
from breast_cancer_labs.plotting import apply_style, save_figure


FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "worst_radius",
    "worst_texture",
]
PRIMARY_SUPPORT = 0.20
MIN_CONFIDENCE = 0.60
SUPPORT_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def format_itemset(value: frozenset[str]) -> str:
    return ", ".join(sorted(value))


def build_basket(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = data[FEATURES].median()
    basket = pd.DataFrame(index=data.index)
    for feature in FEATURES:
        basket[f"{feature}=low"] = data[feature] < thresholds[feature]
        basket[f"{feature}=high"] = data[feature] >= thresholds[feature]
    basket["diagnosis=benign"] = data["diagnosis"].eq("B")
    basket["diagnosis=malignant"] = data["diagnosis"].eq("M")
    threshold_table = thresholds.rename("median_threshold").rename_axis("feature").reset_index()
    return basket.astype(bool), threshold_table


def itemsets_table(itemsets: pd.DataFrame) -> pd.DataFrame:
    table = itemsets.copy()
    table["length"] = table["itemsets"].map(len)
    table["itemsets"] = table["itemsets"].map(format_itemset)
    return table.sort_values(["support", "length"], ascending=[False, True]).reset_index(drop=True)


def rules_table(itemsets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rules = association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    diagnosis_mask = raw_rules["consequents"].map(
        lambda items: len(items) == 1 and next(iter(items)).startswith("diagnosis=")
    )
    diagnosis_rules = raw_rules.loc[diagnosis_mask].copy()
    columns = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]

    def formatted(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[columns].copy()
        result["antecedents"] = result["antecedents"].map(format_itemset)
        result["consequents"] = result["consequents"].map(format_itemset)
        return result.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

    return formatted(raw_rules), formatted(diagnosis_rules)


def main() -> None:
    apply_style()
    output_dir = prepare_output_dir("lab07_association_rules")
    data = load_clean_dataset()
    basket, thresholds = build_basket(data)
    basket.to_csv(output_dir / "binary_basket.csv", index=False)
    thresholds.to_csv(output_dir / "discretization_thresholds.csv", index=False)

    apriori_sets = apriori(
        basket, min_support=PRIMARY_SUPPORT, use_colnames=True, max_len=3
    )
    fp_sets = fpgrowth(
        basket, min_support=PRIMARY_SUPPORT, use_colnames=True, max_len=3
    )
    apriori_items = itemsets_table(apriori_sets)
    fp_items = itemsets_table(fp_sets)
    all_rules, diagnosis_rules = rules_table(apriori_sets)
    fp_rules, fp_diagnosis_rules = rules_table(fp_sets)

    apriori_items.to_csv(output_dir / "apriori_itemsets.csv", index=False)
    fp_items.to_csv(output_dir / "fpgrowth_itemsets.csv", index=False)
    all_rules.to_csv(output_dir / "apriori_rules.csv", index=False)
    fp_rules.to_csv(output_dir / "fpgrowth_rules.csv", index=False)
    diagnosis_rules.to_csv(output_dir / "diagnosis_rules.csv", index=False)
    diagnosis_rules.head(10).to_csv(output_dir / "top10_diagnosis_rules.csv", index=False)

    threshold_rows = []
    for support in SUPPORT_VALUES:
        sets = apriori(basket, min_support=support, use_colnames=True, max_len=3)
        rules = association_rules(sets, metric="confidence", min_threshold=MIN_CONFIDENCE)
        threshold_rows.append(
            {"min_support": support, "frequent_itemsets": len(sets), "rules": len(rules)}
        )
    support_comparison = pd.DataFrame(threshold_rows)
    support_comparison.to_csv(output_dir / "support_comparison.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(support_comparison["min_support"], support_comparison["frequent_itemsets"], marker="o", label="Itemsets")
    axis.plot(support_comparison["min_support"], support_comparison["rules"], marker="s", label="Rules")
    axis.set(xlabel="Minimum support", ylabel="Count", title="Effect of support threshold")
    axis.legend()
    save_figure(fig, output_dir / "support_threshold_effect.png")

    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(
        all_rules["confidence"],
        all_rules["lift"],
        s=all_rules["support"] * 350,
        alpha=0.55,
        color="#457b9d",
    )
    axis.axhline(1, color="gray", linestyle="--")
    axis.set(xlabel="Confidence", ylabel="Lift", title="Association rules")
    save_figure(fig, output_dir / "rules_scatter.png")

    top = diagnosis_rules.head(8)
    graph = nx.DiGraph()
    for _, row in top.iterrows():
        graph.add_edge(row["antecedents"], row["consequents"], weight=row["lift"])
    fig, axis = plt.subplots(figsize=(12, 8))
    positions = nx.spring_layout(graph, seed=42, k=1.2)
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        node_color="#dbe9f4",
        node_size=2600,
        font_size=7,
        arrows=True,
        edge_color="#457b9d",
    )
    axis.set_title("Top diagnosis-related association rules")
    axis.axis("off")
    save_figure(fig, output_dir / "top_rules_network.png")

    same_itemsets = len(apriori_sets) == len(fp_sets)
    same_diagnosis_rules = len(diagnosis_rules) == len(fp_diagnosis_rules)
    strongest = diagnosis_rules.iloc[0]
    report = f"""# Лабораторная работа №7 — ассоциативные правила

Каждая из **{len(data)}** строк пациента преобразована в транзакцию: для восьми
заранее выбранных признаков записано состояние `low/high` относительно медианы,
а диагноз добавлен отдельным item. Это учебная бинаризация, а не исходная
market-basket структура.

При min_support={PRIMARY_SUPPORT:.2f}, confidence≥{MIN_CONFIDENCE:.2f} и
max_len=3 Apriori нашёл **{len(apriori_sets)}** itemsets и **{len(all_rules)}**
правил. FP-Growth нашёл **{len(fp_sets)}** itemsets и **{len(fp_rules)}** правил.
Совпадение количества itemsets: **{same_itemsets}**; diagnosis-правил:
**{same_diagnosis_rules}**.

Самое сильное правило для диагноза: `{strongest['antecedents']}` →
`{strongest['consequents']}` (support={strongest['support']:.3f},
confidence={strongest['confidence']:.3f}, lift={strongest['lift']:.3f}).

Практическая интерпретация ограничена: правила описывают совместную встречаемость
дискретизированных признаков, не причинность и не валидированный диагностический
протокол. Их можно использовать для формулирования гипотез и выбора сочетаний
признаков для последующей проверяемой модели.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(support_comparison.to_string(index=False))
    print(f"Apriori: {len(apriori_sets)} itemsets / {len(all_rules)} rules")
    print(f"FP-Growth: {len(fp_sets)} itemsets / {len(fp_rules)} rules")


if __name__ == "__main__":
    main()
