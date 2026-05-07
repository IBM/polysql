"""Enrich results with sqlglot-based transpilation metrics.

This module adds two evaluation strategies using sqlglot transpilation:
1. sqlglot_pred_to_source: Transpile prediction back to source (single DB)
2. sqlglot_gold_to_target: Transpile gold to target, compare with prediction
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from polysql.evaluation.analysis.transpilation import (
    extract_identifiers,
    extract_schema_from_prompt,
    extract_sql_from_response,
    evaluate_llm_gold_to_target,
    evaluate_sqlglot_gold_to_target,
    evaluate_sqlglot_pred_to_source,
    get_db_path,
    get_target_schema,
    transpile_with_llm,
)
from polysql.evaluation.metrics.dialect_comparison import (
    QueryInput,
    generic_dialect_metric,
)
from polysql.evaluation.core.model import CrossProviderInferenceEngineWithMoreRISTModels

# Column constants
SQLGLOT_PRED_TO_SOURCE_COLUMN = "sqlglot_pred_to_source"
SQLGLOT_GOLD_TO_TARGET_COLUMN = "sqlglot_gold_to_target"
LLM_GOLD_TO_TARGET_COLUMN = "llm_gold_to_target"
LLM_TRANSPILATION_CORRECT_COLUMN = "llm_transpilation_correct"
MERGE_KEY_COLUMN = "__merge_key"
KEY_COLUMNS = ["experiment_id", "model_name", "gen_type", "question_id", "db_id"]


def _build_merge_key(df: pd.DataFrame) -> pd.Series:
    """Build merge key from KEY_COLUMNS."""
    return df[KEY_COLUMNS].astype(str).agg("||".join, axis=1)


def enrich_sqlglot_metrics(
    results_dir: Path,
    max_rows: Optional[int] = None,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
) -> Path:
    """Add sqlglot-based evaluation metrics to enriched results.

    Args:
        results_dir: Directory containing all_results_enriched.csv
        max_rows: Optional limit on number of rows to process
        model_name: Model name for LLM transpilation
        provider: Provider for LLM transpilation

    Returns:
        Path to updated all_results_enriched.csv
    """
    results_dir = Path(results_dir)
    enriched_path = results_dir / "all_results_enriched.csv"

    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched results not found: {enriched_path}")

    print(f"Loading enriched results from {enriched_path}...")
    df = pd.read_csv(enriched_path)
    print(f"Loaded {len(df)} rows")

    # Add merge key
    df[MERGE_KEY_COLUMN] = _build_merge_key(df)

    # Filter for bird_mini_dev_sqlite
    # Method 1: Can work with any gen_type (transpile back to sqlite)
    # Method 2: Only works with gen_type='mysql' (only MySQL target databases exist)
    target_mask = (
        (df["dataset_name"] == "bird_mini_dev_sqlite")
        & (df["gen_type"].isin(["mysql", "postgres", "postgresql"]))
    )

    # Initialize columns if they don't exist
    if SQLGLOT_PRED_TO_SOURCE_COLUMN not in df.columns:
        df[SQLGLOT_PRED_TO_SOURCE_COLUMN] = pd.NA
    if SQLGLOT_GOLD_TO_TARGET_COLUMN not in df.columns:
        df[SQLGLOT_GOLD_TO_TARGET_COLUMN] = pd.NA
    if LLM_GOLD_TO_TARGET_COLUMN not in df.columns:
        df[LLM_GOLD_TO_TARGET_COLUMN] = pd.NA
    if LLM_TRANSPILATION_CORRECT_COLUMN not in df.columns:
        df[LLM_TRANSPILATION_CORRECT_COLUMN] = pd.NA

    # Filter for rows that don't have LLM metrics yet (process if any metric is missing)
    pending_mask = target_mask & (
        df[SQLGLOT_PRED_TO_SOURCE_COLUMN].isna() |
        df[SQLGLOT_GOLD_TO_TARGET_COLUMN].isna() |
        df[LLM_GOLD_TO_TARGET_COLUMN].isna() |
        df[LLM_TRANSPILATION_CORRECT_COLUMN].isna()
    )

    pending_df = df[pending_mask].copy()

    print(f"Found {len(pending_df)} rows to process")

    if len(pending_df) == 0:
        print("No pending rows to process")
        return enriched_path

    if max_rows and len(pending_df) > max_rows:
        print(f"Limiting to {max_rows} rows")
        pending_df = pending_df.sample(n=max_rows, random_state=42)

    # Create inference engine for LLM transpilation
    inference_engine = CrossProviderInferenceEngineWithMoreRISTModels(
        model=model_name,
        provider=provider,
        data_classification_policy=["public"],
        max_tokens=4096,
        temperature=0.0,
        seed=42,
    )

    # Process each row
    results = []

    for idx, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Enriching"):
        result = {MERGE_KEY_COLUMN: row[MERGE_KEY_COLUMN]}

        db_id = row["db_id"]
        dataset_path = row.get("dataset_path", "")
        gen_type = row["gen_type"]
        gold_sql = row["gold_sql"]
        predicted_sql = row["predicted_sql"]

        # Normalize gen_type
        read_dialect = gen_type
        write_dialect = gen_type
        if gen_type == "postgresql":
            read_dialect = "postgres"
            write_dialect = "postgres"

        # === METHOD 1: Transpile Prediction to Source ===
        # Skip if already computed
        if pd.notna(row.get(SQLGLOT_PRED_TO_SOURCE_COLUMN)):
            result[SQLGLOT_PRED_TO_SOURCE_COLUMN] = row[SQLGLOT_PRED_TO_SOURCE_COLUMN]
        else:
            sqlite_db_path = get_db_path(dataset_path, db_id)
            pred_to_source = evaluate_sqlglot_pred_to_source(
                predicted_sql=predicted_sql,
                gold_sql=gold_sql,
                read_dialect=read_dialect,
                sqlite_db_path=sqlite_db_path
            )
            result[SQLGLOT_PRED_TO_SOURCE_COLUMN] = pred_to_source

        # === METHOD 2: Transpile Gold to Target (with schema mapping) ===
        # Skip if already computed
        if pd.notna(row.get(SQLGLOT_GOLD_TO_TARGET_COLUMN)):
            result[SQLGLOT_GOLD_TO_TARGET_COLUMN] = row[SQLGLOT_GOLD_TO_TARGET_COLUMN]
        else:
            sqlite_db_path = get_db_path(dataset_path, db_id)
            gold_to_target = evaluate_sqlglot_gold_to_target(
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                write_dialect=write_dialect,
                sqlite_db_path=sqlite_db_path
            )
            result[SQLGLOT_GOLD_TO_TARGET_COLUMN] = gold_to_target

        # === METHOD 3: LLM Transpile Gold to Target ===
        full_prompt = row.get("full_prompt", "")
        schema = extract_schema_from_prompt(full_prompt)

        if not schema:
            result[LLM_GOLD_TO_TARGET_COLUMN] = False
            result[LLM_TRANSPILATION_CORRECT_COLUMN] = False
        else:
            sqlite_db_path = get_db_path(dataset_path, db_id)
            matches_prediction, transpilation_correct, transpiled_sql = evaluate_llm_gold_to_target(
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                schema=schema,
                write_dialect=write_dialect,
                sqlite_db_path=sqlite_db_path,
                inference_engine=inference_engine,
                verbose=True
            )
            result[LLM_GOLD_TO_TARGET_COLUMN] = matches_prediction
            result[LLM_TRANSPILATION_CORRECT_COLUMN] = transpilation_correct

        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge back into original DataFrame
    df = df.set_index(MERGE_KEY_COLUMN)
    results_df = results_df.set_index(MERGE_KEY_COLUMN)

    # Update only the processed rows
    df.update(results_df)
    df = df.reset_index()

    # Drop merge key column
    df = df.drop(columns=[MERGE_KEY_COLUMN])

    # Save
    print(f"Saving enriched results to {enriched_path}...")
    df.to_csv(enriched_path, index=False)
    print("Done!")

    # Print summary
    processed_mask = df[SQLGLOT_PRED_TO_SOURCE_COLUMN].notna()
    processed_count = processed_mask.sum()

    if processed_count > 0:
        pred_to_source_success = df[processed_mask][SQLGLOT_PRED_TO_SOURCE_COLUMN].sum()
        gold_to_target_success = df[processed_mask][SQLGLOT_GOLD_TO_TARGET_COLUMN].sum()
        llm_gold_to_target_success = df[processed_mask][LLM_GOLD_TO_TARGET_COLUMN].sum()

        # Check LLM transpilation accuracy
        llm_transpilation_mask = df[LLM_TRANSPILATION_CORRECT_COLUMN].notna()
        llm_transpilation_count = llm_transpilation_mask.sum()
        if llm_transpilation_count > 0:
            llm_transpilation_success = df[llm_transpilation_mask][LLM_TRANSPILATION_CORRECT_COLUMN].sum()
        else:
            llm_transpilation_success = 0

        print(f"\n=== Summary ===")
        print(f"Processed rows: {processed_count}")
        print(f"Method 1 (pred→source) success: {pred_to_source_success}/{processed_count} ({pred_to_source_success/processed_count:.1%})")
        print(f"Method 2 (gold→target) success: {gold_to_target_success}/{processed_count} ({gold_to_target_success/processed_count:.1%})")
        print(f"Method 3 (LLM gold→target) success: {llm_gold_to_target_success}/{processed_count} ({llm_gold_to_target_success/processed_count:.1%})")
        if llm_transpilation_count > 0:
            print(f"\n🎯 LLM Transpilation Accuracy: {llm_transpilation_success}/{llm_transpilation_count} ({llm_transpilation_success/llm_transpilation_count:.1%})")
            print(f"   (This measures if LLM-transpiled queries produce the same results as original gold queries)")

    return enriched_path


def main():
    parser = ArgumentParser(description="Enrich results with sqlglot transpilation metrics")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--model-name", default="gpt-oss-120b", help="Model name for LLM transpilation")
    parser.add_argument("--provider", default="rits", help="Provider for LLM transpilation")
    args = parser.parse_args()

    enrich_sqlglot_metrics(
        Path(args.results_dir),
        max_rows=args.max_rows,
        model_name=args.model_name,
        provider=args.provider
    )


if __name__ == "__main__":
    main()
