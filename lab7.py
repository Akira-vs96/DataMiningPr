from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


INPUT = Path("processed_data/bread_basket_cleaned.csv")
OUTPUT_DIR = Path("association_results")
PRIMARY_SUPPORT = 0.02
SUPPORTS = [0.01, 0.02, 0.03, 0.04, 0.05]


def format_itemset(value: frozenset[str]) -> str:
    return ", ".join(sorted(value))


def prepare_basket() -> tuple[pd.DataFrame, list[list[str]]]:
    data = pd.read_csv(INPUT)
    transactions = (
        data.groupby("transaction", sort=True)["item"]
        .apply(lambda values: sorted(set(values)))
        .tolist()
    )
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(encoded, columns=encoder.columns_)
    return basket, transactions


def itemsets_table(itemsets: pd.DataFrame) -> pd.DataFrame:
    result = itemsets.copy()
    result["length"] = result["itemsets"].map(len)
    result["itemsets"] = result["itemsets"].map(format_itemset)
    return result.sort_values(
        ["support", "length", "itemsets"], ascending=[False, True, True]
    ).reset_index(drop=True)


def rules_table(itemsets: pd.DataFrame) -> pd.DataFrame:
    rules = association_rules(
        itemsets, metric="confidence", min_threshold=0.4
    )
    columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
    ]
    result = rules[columns].copy()
    result["antecedents"] = result["antecedents"].map(format_itemset)
    result["consequents"] = result["consequents"].map(format_itemset)
    return result.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)


def save_threshold_plot(thresholds: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(
        thresholds["min_support"],
        thresholds["frequent_itemsets"],
        marker="o",
        label="Frequent itemsets",
    )
    plt.plot(
        thresholds["min_support"],
        thresholds["rules"],
        marker="s",
        label="Rules (confidence >= 0.4)",
    )
    plt.xlabel("Minimum support")
    plt.ylabel("Count")
    plt.title("Effect of minimum support")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_effect.png", dpi=160)
    plt.close()


def save_rule_scatter(rules: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(
        rules["confidence"],
        rules["lift"],
        s=rules["support"] * 4000,
        alpha=0.65,
    )
    plt.axhline(1.0, linestyle="--", color="gray")
    plt.xlabel("Confidence")
    plt.ylabel("Lift")
    plt.title("Association rules: confidence vs lift")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rules_confidence_lift.png", dpi=160)
    plt.close()


def save_rule_network(rules: pd.DataFrame) -> None:
    top = rules.head(8)
    graph = nx.DiGraph()
    for _, row in top.iterrows():
        graph.add_edge(
            row["antecedents"],
            row["consequents"],
            weight=float(row["lift"]),
        )
    plt.figure(figsize=(12, 8))
    positions = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        positions,
        node_size=2600,
        node_color="#b9d7ea",
        font_size=8,
        arrows=True,
        arrowsize=18,
    )
    plt.title("Top association rules by lift")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_rules_network.png", dpi=160)
    plt.close()


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    basket, transactions = prepare_basket()
    n_transactions = len(transactions)
    n_items = basket.shape[1]
    item_support = basket.mean().sort_values(ascending=False)
    item_support.head(10).rename_axis("item").rename("support").to_csv(
        OUTPUT_DIR / "item_support.csv", header=True
    )

    threshold_rows = []
    for support in SUPPORTS:
        itemsets = apriori(
            basket, min_support=support, use_colnames=True
        )
        threshold_rows.append(
            {
                "min_support": support,
                "frequent_itemsets": len(itemsets),
                "rules": len(rules_table(itemsets))
                if len(itemsets) > 1
                else 0,
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUTPUT_DIR / "support_comparison.csv", index=False)
    save_threshold_plot(thresholds)

    apriori_itemsets = apriori(
        basket, min_support=PRIMARY_SUPPORT, use_colnames=True
    )
    fp_itemsets = fpgrowth(
        basket, min_support=PRIMARY_SUPPORT, use_colnames=True
    )
    apriori_items = itemsets_table(apriori_itemsets)
    fp_items = itemsets_table(fp_itemsets)
    apriori_rules = rules_table(apriori_itemsets)
    fp_rules = rules_table(fp_itemsets)

    apriori_items.to_csv(OUTPUT_DIR / "apriori_itemsets.csv", index=False)
    apriori_items.head(10).to_csv(
        OUTPUT_DIR / "top_10_frequent_itemsets.csv", index=False
    )
    fp_items.to_csv(OUTPUT_DIR / "fpgrowth_itemsets.csv", index=False)
    apriori_rules.to_csv(OUTPUT_DIR / "apriori_rules.csv", index=False)
    fp_rules.to_csv(OUTPUT_DIR / "fpgrowth_rules.csv", index=False)

    positive_rules = apriori_rules[apriori_rules["lift"] > 1].copy()
    high_confidence = apriori_rules[apriori_rules["confidence"] >= 0.70].copy()
    top_rules = apriori_rules.head(5).copy()
    positive_rules.to_csv(OUTPUT_DIR / "rules_lift_above_1.csv", index=False)
    high_confidence.to_csv(
        OUTPUT_DIR / "rules_confidence_at_least_070.csv", index=False
    )
    top_rules.to_csv(OUTPUT_DIR / "top_5_rules.csv", index=False)

    save_rule_scatter(apriori_rules)
    save_rule_network(apriori_rules)

    counts_by_length = (
        apriori_items.groupby("length").size().reindex([1, 2, 3], fill_value=0)
    )
    with (OUTPUT_DIR / "analysis_report.txt").open("w", encoding="utf-8") as report:
        report.write("Association rules analysis on cleaned Bread Basket data\n")
        report.write("=" * 58 + "\n\n")
        report.write(f"Input: {INPUT}\n")
        report.write(f"Transactions: {n_transactions}\n")
        report.write(f"Unique items: {n_items}\n")
        report.write(f"Primary minimum support: {PRIMARY_SUPPORT:.2f}\n")
        report.write("Rules minimum confidence: 0.40\n\n")
        report.write("Top 10 individual item supports:\n")
        report.write(item_support.head(10).to_string(float_format=lambda x: f"{x:.4f}"))
        report.write("\n\nApriori frequent itemsets by length:\n")
        for length, count in counts_by_length.items():
            report.write(f"  {length}-itemsets: {count}\n")
        report.write(f"  Total: {len(apriori_items)}\n")
        report.write(f"  Rules: {len(apriori_rules)}\n")
        report.write(f"  Lift > 1 rules: {len(positive_rules)}\n")
        report.write(f"  Confidence >= 0.70 rules: {len(high_confidence)}\n\n")
        report.write("Apriori vs FP-Growth:\n")
        report.write(f"  Apriori itemsets: {len(apriori_items)}; rules: {len(apriori_rules)}\n")
        report.write(f"  FP-Growth itemsets: {len(fp_items)}; rules: {len(fp_rules)}\n")
        report.write(
            f"  Maximum Apriori lift: {apriori_rules['lift'].max():.4f}\n"
        )
        report.write(f"  Maximum FP-Growth lift: {fp_rules['lift'].max():.4f}\n")
        report.write(
            "  Both algorithms should produce the same itemsets and rules; "
            "FP-Growth avoids explicit candidate generation and is preferable "
            "for larger datasets.\n\n"
        )
        report.write("Support threshold comparison:\n")
        report.write(thresholds.to_string(index=False))
        report.write("\n\nTop 5 rules by lift:\n")
        report.write(
            top_rules.to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )
        report.write("\n\nBusiness interpretation and recommendations:\n")
        for index, row in top_rules.iterrows():
            report.write(
                f"  {index + 1}. Customers buying [{row['antecedents']}] "
                f"also tend to buy [{row['consequents']}] "
                f"(confidence {row['confidence']:.2%}, lift {row['lift']:.2f}). "
                "Use as a cross-sell or bundle candidate.\n"
            )
        report.write(
            "  - Place high-lift antecedent and consequent products together "
            "or use them in recommendation prompts.\n"
            "  - Test bundle discounts for the strongest multi-item rules.\n"
            "  - Use the 0.02 support threshold as a practical compromise: "
            "it retains recurring combinations without the very large rule "
            "volume produced by lower thresholds.\n"
        )


if __name__ == "__main__":
    main()
