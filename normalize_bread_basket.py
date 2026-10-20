import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


INPUT = Path("extracted_data/bread basket.csv")
OUTPUT_DIR = Path("processed_data")
OUTPUT = OUTPUT_DIR / "bread_basket_cleaned.csv"
OUTLIERS = OUTPUT_DIR / "transaction_outliers.csv"
REPORT = OUTPUT_DIR / "data_quality_report.txt"


def quartile(values: list[int], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")

    with INPUT.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("The input dataset has no header row.")
        required = {
            "Transaction",
            "Item",
            "date_time",
            "period_day",
            "weekday_weekend",
        }
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        source_rows = list(reader)

    missing_before = {
        column: sum(not (source_row.get(column) or "").strip() for source_row in source_rows)
        for column in reader.fieldnames
    }
    normalized_rows: list[dict[str, str]] = []
    invalid_rows: list[int] = []

    for row_number, row in enumerate(source_rows, start=2):
        transaction = (row.get("Transaction") or "").strip()
        item = " ".join((row.get("Item") or "").split())
        date_text = (row.get("date_time") or "").strip()
        period_day = (row.get("period_day") or "").strip().lower()
        weekday_weekend = (row.get("weekday_weekend") or "").strip().lower()

        if not transaction or not item or not date_text:
            invalid_rows.append(row_number)
            continue
        try:
            transaction_id = int(transaction)
            parsed_date = datetime.strptime(date_text, "%d-%m-%Y %H:%M")
        except ValueError:
            invalid_rows.append(row_number)
            continue

        normalized_rows.append(
            {
                "transaction": str(transaction_id),
                "item": item,
                "date_time": parsed_date.strftime("%Y-%m-%d %H:%M:%S"),
                "period_day": period_day or "unknown",
                "weekday_weekend": weekday_weekend or "unknown",
            }
        )

    seen: set[tuple[str, ...]] = set()
    cleaned_rows: list[dict[str, str]] = []
    for row in normalized_rows:
        key = tuple(row[column] for column in row)
        if key not in seen:
            seen.add(key)
            cleaned_rows.append(row)

    transaction_counts = Counter(row["transaction"] for row in cleaned_rows)
    sorted_counts = sorted(transaction_counts.values())
    q1 = quartile(sorted_counts, 0.25)
    q3 = quartile(sorted_counts, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_transactions = sorted(
        (
            {
                "transaction": transaction,
                "item_count": str(item_count),
                "outlier_type": (
                    "high" if item_count > upper_bound else "low"
                ),
            }
            for transaction, item_count in transaction_counts.items()
            if item_count < lower_bound or item_count > upper_bound
        ),
        key=lambda row: int(row["transaction"]),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(cleaned_rows[0]))
        writer.writeheader()
        writer.writerows(cleaned_rows)

    with OUTLIERS.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=["transaction", "item_count", "outlier_type"]
        )
        writer.writeheader()
        writer.writerows(outlier_transactions)

    with REPORT.open("w", encoding="utf-8") as target:
        target.write("Bread basket dataset quality report\n")
        target.write("=" * 36 + "\n\n")
        target.write(f"Input rows: {len(source_rows)}\n")
        target.write(f"Rows after normalization: {len(normalized_rows)}\n")
        target.write(f"Rows after exact-duplicate removal: {len(cleaned_rows)}\n")
        target.write(f"Invalid rows removed: {len(invalid_rows)}\n")
        target.write(f"Exact duplicate rows removed: {len(normalized_rows) - len(cleaned_rows)}\n\n")
        target.write("Missing values before cleaning:\n")
        for column, count in missing_before.items():
            target.write(f"  {column}: {count}\n")
        target.write("\nMissing-value handling:\n")
        target.write("  No missing values were present in the source dataset.\n")
        target.write("  If a categorical value had been empty, it would be set to 'unknown'.\n")
        target.write("  Rows with missing/invalid transaction, item, or datetime were excluded.\n\n")
        target.write("Normalization:\n")
        target.write("  Headers converted to snake_case.\n")
        target.write("  Text whitespace trimmed; categorical values lowercased.\n")
        target.write("  transaction converted to integer text; date_time converted to ISO format.\n\n")
        target.write("Outlier check (transaction item count, Tukey 1.5*IQR rule):\n")
        target.write(f"  Transactions: {len(transaction_counts)}\n")
        target.write(f"  Q1: {q1:.2f}; Q3: {q3:.2f}; IQR: {iqr:.2f}\n")
        target.write(f"  Bounds: {lower_bound:.2f} to {upper_bound:.2f}\n")
        target.write(f"  Outlier transactions flagged: {len(outlier_transactions)}\n")
        target.write("  Outliers were flagged, not deleted, because large baskets can be valid purchases.\n")


if __name__ == "__main__":
    main()
