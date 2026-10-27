"""Practical 13, variant 1: influential nodes in the Karate Club network."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from breast_cancer_labs.config import prepare_output_dir
from breast_cancer_labs.plotting import apply_style, save_figure
from breast_cancer_labs.practical_common import markdown_table, save_environment


def main() -> None:
    apply_style()
    output = prepare_output_dir("lab13_graph_analysis")
    graph = nx.karate_club_graph()
    # weight is interaction strength, not distance. Shortest paths use hop count.
    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    groups = {node: i + 1 for i, group in enumerate(communities) for node in group}
    centrality = pd.DataFrame({"degree": pd.Series(dict(graph.degree())),
                               "degree_centrality": pd.Series(nx.degree_centrality(graph)),
                               "closeness": pd.Series(nx.closeness_centrality(graph, distance=None)),
                               "betweenness": pd.Series(nx.betweenness_centrality(graph, weight=None)),
                               "pagerank": pd.Series(nx.pagerank(graph, alpha=.85, weight="weight")),
                               "community": pd.Series(groups)})
    centrality.index.name = "node_id"
    centrality.to_csv(output / "graph_results.csv")
    nx.to_pandas_edgelist(graph).to_csv(output / "edges.csv", index=False)
    metrics = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
               "directed": graph.is_directed(), "connected": nx.is_connected(graph),
               "components": nx.number_connected_components(graph), "self_loops": nx.number_of_selfloops(graph),
               "average_degree": sum(dict(graph.degree()).values()) / len(graph),
               "density": nx.density(graph), "communities": len(communities),
               "community_sizes": [len(c) for c in communities],
               "modularity": nx.community.modularity(graph, communities, weight="weight")}
    (output / "graph_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    ranks = pd.DataFrame({name: centrality.sort_values(name, ascending=False, kind="stable").head(5).index
                          for name in ["degree_centrality", "closeness", "betweenness", "pagerank"]})
    ranks.index = range(1, 6)
    ranks.rename_axis("rank").to_csv(output / "top5_rankings.csv")
    pos = nx.spring_layout(graph, seed=42, weight="weight")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, title, colors in [(axes[0], "Karate Club: node size = PageRank", "skyblue"),
                              (axes[1], "Louvain communities", [groups[n] for n in graph])]:
        nx.draw_networkx(graph, pos, ax=ax, node_color=colors, cmap=plt.cm.tab10 if isinstance(colors, list) else None,
                         node_size=[250 + 6500 * centrality.loc[n, "pagerank"] for n in graph],
                         edge_color="lightgray", font_size=9)
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, output / "network_communities.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    centrality.nlargest(10, "pagerank").pagerank.plot.bar(ax=axes[0], title="Top 10 weighted PageRank")
    axes[1].scatter(centrality.degree_centrality, centrality.betweenness)
    for node in set(ranks.pagerank) | set(ranks.betweenness):
        axes[1].annotate(str(node), (centrality.loc[node, "degree_centrality"], centrality.loc[node, "betweenness"]))
    axes[1].set(xlabel="Degree centrality", ylabel="Betweenness", title="Connections and brokerage")
    save_figure(fig, output / "centrality_comparison.png")
    pr, bw = int(ranks.pagerank.iloc[0]), int(ranks.betweenness.iloc[0])
    report = f"""# Практическая работа №13 — вариант 1

Тема: влиятельные участники социальной сети.
[Задание](../../Data_Mining_Practica_13.docx).
Источник: встроенный [NetworkX Karate Club](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html),
социальные связи 34 участников клуба Захари. Номера узлов — идентификаторы, не имена людей.

## Построение и структура графа

Использован `nx.karate_club_graph()`: {metrics['nodes']} узла, {metrics['edges']} рёбер,
граф неориентированный, компонент связности {metrics['components']}, петель {metrics['self_loops']}.
Средняя степень — {metrics['average_degree']:.4f}, плотность — {metrics['density']:.4f}.
Список рёбер сохранён в [edges.csv](edges.csv), поэтому структуру можно проверить.
Вес ребра означает силу взаимодействия: PageRank и Louvain учитывают его как силу;
closeness и betweenness считают кратчайшие пути по числу рёбер, без весов.
Визуальное расстояние spring-layout не является географическим или социальным измерением.

## Сравнение первых пяти узлов по центральности

{markdown_table(ranks.reset_index(names='rank'))}

{markdown_table(centrality.loc[sorted(set(ranks.to_numpy().ravel()))].reset_index())}

Лидер PageRank — узел **{pr}** ({centrality.loc[pr, 'pagerank']:.4f}),
лидер betweenness — **{bw}** ({centrality.loc[bw, 'betweenness']:.4f}).
PageRank оценивает вес связей с другими важными узлами, betweenness — долю кратчайших путей,
проходящих через посредника. Поэтому участник плотной группы может иметь высокий PageRank,
а узел, связывающий группы, — высокую betweenness при другом числе связей.
Degree измеряет число соседей, closeness — доступность остальных узлов по коротким путям.

![Сеть и сообщества](network_communities.png)
![Сравнение центральностей](centrality_comparison.png)

## Louvain

При seed=42 и стандартном resolution=1 найдено {len(communities)} сообществ
размеров {metrics['community_sizes']}, взвешенная модульность — {metrics['modularity']:.4f}.
Модульность оценивает относительную концентрацию связей внутри найденных групп;
это не точность классификации и не доказательство социальной причины разделения клуба.
Номера сообществ произвольны; настройки и веса влияют на разбиение.

## Вывод

Все узлы связаны в одну компоненту, но роли участников различаются по выбранной метрике.
PageRank выделил узел {pr}, а betweenness — узел {bw}.
Louvain выявил {len(communities)} сообществ с модульностью {metrics['modularity']:.4f}.
Выводы относятся к этой небольшой исторической сети и используемой интерпретации весов.

Запуск: `PYTHONPATH=src .venv/bin/python -m breast_cancer_labs.labs.lab13_graph_analysis`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    save_environment(output, [], dataset="networkx.karate_club_graph", pagerank_alpha=.85, louvain_seed=42,
                     weighted_metrics=["pagerank", "louvain"], unweighted_metrics=["degree", "closeness", "betweenness"])
    print(ranks.to_string())


if __name__ == "__main__":
    main()
