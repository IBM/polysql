import sys
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp
from tqdm import tqdm

# Add src to python path to import modules
sys.path.append(str(Path.cwd() / "src"))

from polysql.evaluation.metrics.dialect_comparison import (
    QueryInput,
    generic_dialect_metric,
)


def get_db_path(dataset_path: str, db_id: str) -> Path:
    """
    Resolve the SQLite database path based on dataset path conventions.
    """
    dataset_path_str = str(dataset_path).lower()
    if "bird" in dataset_path_str or "minidev" in dataset_path_str:
        candidates = [
            Path(f"data/BIRD/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite"),
            Path(f"data/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"),
        ]
        for c in candidates:
            if c.exists():
                return c
    # Fallback to standard
    return Path(f"data/databases/{db_id}/{db_id}.sqlite")


def run_simulation(results_path: str, max_rows: int = 1000):
    """
    Compare Simulated Execution (on SQLite) vs Native Execution (on MySQL/Postgres).
    Ground Truth: bird_mini_dev_{native} (gen_type={native})
    Prediction: bird_mini_dev_sqlite (gen_type={native}) -> Transpiled to SQLite
    """
    print(f"Loading results from {results_path}...", flush=True)
    df = pd.read_csv(results_path)
    print(f"Results loaded. Shape: {df.shape}", flush=True)

    # 1. Extract Ground Truth
    native_df = df[
        (df["dataset_name"].isin(["bird_mini_dev_mysql", "bird_mini_dev_postgres"]))
        & (df["gen_type"].isin(["mysql", "postgres", "postgresql"]))
    ][["question_id", "model_name", "gen_type", "results_equal"]].copy()

    native_df = native_df.rename(columns={"results_equal": "native_truth"})

    print(f"Found {len(native_df)} native ground truth records.", flush=True)

    # 2. Extract Simulation Candidates
    sim_df = df[
        (df["dataset_name"] == "bird_mini_dev_sqlite")
        & (df["gen_type"].isin(["mysql", "postgres", "postgresql"]))
    ].copy()

    print(
        f"Found {len(sim_df)} simulation candidates (SQLite -> MySQL/Postgres).",
        flush=True,
    )

    # 3. Join
    merged = pd.merge(
        sim_df, native_df, on=["question_id", "model_name", "gen_type"], how="inner"
    )

    print(f"Matched {len(merged)} records for comparison.", flush=True)

    if max_rows and len(merged) > max_rows:
        merged = merged.sample(n=max_rows, random_state=42)

    print(f"Processing {len(merged)} rows...")

    results = []

    def strip_qualifiers(expression):
        for table in expression.find_all(exp.Table):
            table.set("db", None)
            table.set("catalog", None)
            table.set("this", exp.Identifier(this=table.name, quoted=True))
        return expression

    for i, (idx, row) in tqdm(enumerate(merged.iterrows(), total=len(merged))):
        if i == 49:
            continue
        question_id = row.get("question_id")
        db_id = row.get("db_id")
        dataset_path = row.get("dataset_path")
        gen_type = row.get("gen_type")
        predicted_sql = row.get("predicted_sql")
        gold_sql = row.get("gold_sql")
        native_truth = row.get("native_truth")

        # 1. Resolve DB Path (Should be SQLite path for bird_mini_dev_sqlite)
        db_path = get_db_path(dataset_path, db_id)

        if not db_path.exists():
            results.append({"status": "db_not_found"})
            continue

        # 2. Transpile
        transpiled_sql = None
        transpilation_error = None

        read_dialect = gen_type
        if read_dialect == "postgresql":
            read_dialect = "postgres"

        try:
            expression = sqlglot.parse_one(predicted_sql, read=read_dialect)
            expression = strip_qualifiers(expression)
            transpiled_sql = expression.sql(dialect="sqlite")
        except Exception as e:
            transpilation_error = str(e)

        if transpilation_error:
            results.append(
                {
                    "status": "transpilation_failed",
                    "error": transpilation_error,
                    "native_truth": native_truth,
                }
            )
            continue

        # 3. Execute on Source SQLite
        try:
            q1 = QueryInput(query=gold_sql, backend="sqlite")
            q2 = QueryInput(query=transpiled_sql, backend="sqlite")

            metric_result = generic_dialect_metric(
                q1, q2, db_path=db_path, load_data=False, source_db_type="sqlite"
            )

            results.append(
                {
                    "question_id": question_id,
                    "gen_type": gen_type,
                    "status": "success",
                    "simulated_match": metric_result.results_equal,
                    "native_truth": native_truth,
                    "exec_error": metric_result.query2_error,
                }
            )

        except Exception as e:
            results.append({"status": "execution_failed", "error": str(e)})

    # Summary
    result_df = pd.DataFrame(results)

    valid_df = result_df[
        (result_df["status"] == "success")
        & (result_df["simulated_match"].notna())
        & (result_df["native_truth"].notna())
    ].copy()

    if not valid_df.empty:
        y_pred = valid_df["simulated_match"].astype(bool)
        y_true = valid_df["native_truth"].astype(bool)

        tp = ((y_pred == True) & (y_true == True)).sum()
        tn = ((y_pred == False) & (y_true == False)).sum()
        fp = ((y_pred == True) & (y_true == False)).sum()
        fn = ((y_pred == False) & (y_true == True)).sum()

        total = len(valid_df)
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print("\n=== Native vs Simulated Validation (MySQL & Postgres) ===")
        print(f"Total Compared: {total}")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    else:
        print("No valid comparisons found.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        default="results/2025-12-05_23-33-21_happy_borg_copy/all_results.csv",
    )
    parser.add_argument("--max_rows", type=int, default=100)
    args = parser.parse_args()
    run_simulation(args.results_path, max_rows=args.max_rows)
