"""Enrich per-example results with judge classifications."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from polysql.evaluation.analysis.insights import (
    CLASSIFICATION_KEY_FIELD,
    EvaluationInsightsGenerator,
)
from polysql.evaluation.analysis.transpilation import (
    batch_transpile_with_llm,
    evaluate_sqlglot_gold_to_target,
    evaluate_sqlglot_pred_to_source,
    extract_schema_from_prompt,
    get_db_path,
)
from polysql.evaluation.core.model import CrossProviderInferenceEngineWithMoreRISTModels
from polysql.evaluation.metrics.dialect_comparison import (
    QueryInput,
    generic_dialect_metric,
)

load_dotenv()

KEY_COLUMNS = ["experiment_id", "model_name", "gen_type", "question_id", "db_id"]
REQUIRED_COLUMNS = KEY_COLUMNS + [
    "question",
    "gold_sql",
    "predicted_sql",
    "results_equal",
    "gold_error",
    "pred_error",
    "full_prompt",
]
ERROR_CLASS_COLUMN = "error_classification"
EXPLANATION_COLUMN = "explanation"
SQLITE_CORRECT_COLUMN = "sqlite_correct"
JUDGE_SCORE_WITHOUT_EXE_COLUMN = "judge_score_without_exe"
SQLGLOT_PRED_TO_SOURCE_COLUMN = "sqlglot_pred_to_source"
SQLGLOT_GOLD_TO_TARGET_COLUMN = "sqlglot_gold_to_target"
LLM_GOLD_TO_TARGET_COLUMN = "llm_gold_to_target"
MERGE_KEY_COLUMN = "__merge_key"
CLASSIFICATION_FAILED_SENTINEL = "CLASSIFICATION_FAILED"


def _validate_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataframe '{df_name}' is missing required columns: {missing}"
        )


def _ensure_no_nulls(df: pd.DataFrame, columns: Iterable[str], df_name: str) -> None:
    for column in columns:
        if df[column].isnull().any():
            raise ValueError(
                f"Column '{column}' in dataframe '{df_name}' contains null values."
            )


def _build_merge_key(df: pd.DataFrame, df_name: str) -> pd.Series:
    _validate_columns(df, KEY_COLUMNS, df_name)
    _ensure_no_nulls(df, KEY_COLUMNS, df_name)
    key_series = df[KEY_COLUMNS].astype(str).agg("||".join, axis=1)
    if key_series.duplicated().any():
        raise ValueError(
            f"{df_name} contains duplicate experiment/question combinations"
        )
    return key_series


def _prepare_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    _validate_columns(df, REQUIRED_COLUMNS, df_name)
    prepared = df.copy()
    prepared[MERGE_KEY_COLUMN] = _build_merge_key(prepared, df_name)
    return prepared


def _prepare_enriched_dataframe(
    base_df: pd.DataFrame, enriched: pd.DataFrame
) -> pd.DataFrame:
    prepared = enriched.copy()
    for column in KEY_COLUMNS:
        if column not in prepared.columns:
            raise ValueError(
                "all_results_enriched.csv is missing required key columns."
            )
    for column in (
        ERROR_CLASS_COLUMN,
        EXPLANATION_COLUMN,
        SQLITE_CORRECT_COLUMN,
        JUDGE_SCORE_WITHOUT_EXE_COLUMN,
        SQLGLOT_PRED_TO_SOURCE_COLUMN,
        SQLGLOT_GOLD_TO_TARGET_COLUMN,
        LLM_GOLD_TO_TARGET_COLUMN,
    ):
        if column not in prepared.columns:
            prepared[column] = pd.NA
    if "results_equal" not in prepared.columns:
        prepared["results_equal"] = pd.NA
    if prepared.empty:
        prepared[MERGE_KEY_COLUMN] = pd.Series(dtype="object")
        return prepared

    if prepared[KEY_COLUMNS].isnull().any().any():
        if len(prepared) != len(base_df):
            raise ValueError(
                "all_results_enriched.csv contains null key columns and cannot be aligned to all_results.csv"
            )
        prepared.loc[:, KEY_COLUMNS] = prepared[KEY_COLUMNS].fillna(
            base_df.loc[:, KEY_COLUMNS]
        )

    prepared[MERGE_KEY_COLUMN] = _build_merge_key(prepared, "all_results_enriched.csv")

    unknown_keys = set(prepared[MERGE_KEY_COLUMN]) - set(base_df[MERGE_KEY_COLUMN])
    if unknown_keys:
        raise ValueError(
            "all_results_enriched.csv contains rows that do not exist in all_results.csv"
        )

    return prepared


def _compute_sqlite_correct(base_df: pd.DataFrame) -> pd.Series:
    """Compute whether each question was correctly answered in the SQLite experiment."""
    sqlite_correct = pd.Series(index=base_df.index, dtype="boolean")

    sqlite_rows = base_df[base_df["gen_type"] == "sqlite"]
    sqlite_lookup = {}
    for _, row in sqlite_rows.iterrows():
        key = (row["model_name"], row["question_id"], row["db_id"])
        sqlite_lookup[key] = row["results_equal"]

    for idx, row in base_df.iterrows():
        if row["gen_type"] == "sqlite":
            sqlite_correct.loc[idx] = pd.NA
        else:
            key = (row["model_name"], row["question_id"], row["db_id"])
            if key not in sqlite_lookup:
                # We skip raising error here to allow partial runs, or handle consistently
                # But strictly per original logic:
                raise ValueError(
                    f"No SQLite baseline found for model={row['model_name']}, "
                    f"question_id={row['question_id']}, db_id={row['db_id']}"
                )
            sqlite_correct.loc[idx] = sqlite_lookup[key]

    return sqlite_correct


def _sample_uniformly_across_experiments(
    df: pd.DataFrame, max_rows: int, random_seed: int = 42
) -> pd.DataFrame:
    """Sample rows uniformly across experiments with random sampling within each experiment."""
    if df.empty or max_rows <= 0:
        return df.head(0)

    experiment_groups = df.groupby("experiment_id", group_keys=False)
    num_experiments = len(experiment_groups)

    if max_rows >= len(df):
        return df

    samples_per_exp = max_rows // num_experiments
    remainder = max_rows % num_experiments

    if samples_per_exp == 0:
        experiment_ids = df["experiment_id"].unique()
        np.random.seed(random_seed)
        selected_experiments = np.random.choice(
            experiment_ids, size=max_rows, replace=False
        )
        sampled_dfs = []
        for exp_id in selected_experiments:
            exp_group = df[df["experiment_id"] == exp_id]
            sampled_dfs.append(exp_group.sample(n=1, random_state=random_seed))
        return pd.concat(sampled_dfs, ignore_index=True)

    sampled_dfs = []
    experiment_ids = list(experiment_groups.groups.keys())
    np.random.seed(random_seed)
    extra_sample_experiments = set(
        np.random.choice(experiment_ids, size=remainder, replace=False)
    )

    for exp_id, group in experiment_groups:
        n_samples = samples_per_exp
        if exp_id in extra_sample_experiments:
            n_samples += 1

        n_samples = min(n_samples, len(group))

        sampled = group.sample(n=n_samples, random_state=random_seed)
        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs, ignore_index=True)


def _rows_to_predictions(rows: pd.DataFrame) -> list[dict]:
    predictions = []
    for _, row in rows.iterrows():
        predictions.append(
            {
                "question_id": int(row["question_id"]),
                "db_id": row["db_id"],
                "gen_type": row["gen_type"],
                "question": row["question"],
                "gold_sql": row["gold_sql"],
                "predicted_sql": row["predicted_sql"],
                "results_equal": row["results_equal"],
                "gold_error": row["gold_error"],
                "pred_error": row["pred_error"],
                "full_prompt": row.get("full_prompt", "") or "",
                CLASSIFICATION_KEY_FIELD: row[MERGE_KEY_COLUMN],
            }
        )
    return predictions


def _rows_to_predictions_for_scoring_without_exe(rows: pd.DataFrame) -> list[dict]:
    """Convert rows to format for score_predictions_without_exe."""
    from polysql.evaluation.analysis.insights import EvaluationInsightsGenerator

    generator = EvaluationInsightsGenerator()
    predictions = []
    for _, row in rows.iterrows():
        # Extract full schema first
        full_schema = generator._extract_schema(row.get("full_prompt", ""))
        # Filter to only relevant tables
        filtered_schema = generator._filter_schema_to_relevant_tables(
            full_schema, row["gold_sql"], row["predicted_sql"]
        )
        predictions.append(
            {
                "question": row["question"],
                "schema": filtered_schema,
                "gold_sql": row["gold_sql"],
                "gold_result": row["gold_result"],
                "gold_error": row.get("gold_error", ""),
                "predicted_sql": row["predicted_sql"],
                "gen_type": row["gen_type"],
                CLASSIFICATION_KEY_FIELD: row[MERGE_KEY_COLUMN],
            }
        )
    return predictions


def enrich_results_directory(
    results_dir: Path,
    *,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
    judge_max_tokens: int = 2048,
    aggregator_max_tokens: int = 4096,
    temperature: float = 0.0,
    max_rows: Optional[int] = None,
    classify_only_gap_instances: bool = False,
    enable_transpilation: bool = False,
    transpilation_model_name: str = "gpt-oss-120b",
    transpilation_provider: str = "rits",
    sqlglot_use_schema_mapping: bool = False,
    llm_use_mapping_instructions: bool = False,
) -> Path:
    """Generate per-example classifications and persist all_results_enriched.csv."""
    results_dir = Path(results_dir)
    results_csv = results_dir / "all_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"all_results.csv not found under {results_dir}")

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    base_df = pd.read_csv(results_csv)
    prepared_base = _prepare_dataframe(base_df, "all_results.csv")

    # --- CHANGE START: Compute SQLite correct early ---
    # We compute this now so we can use it for filtering pending_rows if the flag is set.
    prepared_base[SQLITE_CORRECT_COLUMN] = _compute_sqlite_correct(prepared_base)
    # --- CHANGE END ---

    enriched_path = results_dir / "all_results_enriched.csv"
    full_enriched_path = results_dir / "all_results_enriched_full.csv"

    # Always start fresh - don't load previous results
    prepared_enriched = pd.DataFrame(columns=prepared_base.columns)
    for column in (
        ERROR_CLASS_COLUMN,
        EXPLANATION_COLUMN,
        SQLITE_CORRECT_COLUMN,
        JUDGE_SCORE_WITHOUT_EXE_COLUMN,
        SQLGLOT_PRED_TO_SOURCE_COLUMN,
        SQLGLOT_GOLD_TO_TARGET_COLUMN,
        LLM_GOLD_TO_TARGET_COLUMN,
    ):
        prepared_enriched[column] = pd.Series(dtype="object")
    prepared_enriched[MERGE_KEY_COLUMN] = pd.Series(dtype="object")

    # Start with all rows as pending (no caching)
    pending_rows = prepared_base.copy()

    # --- CHANGE START: Apply gap instance filter ---
    if classify_only_gap_instances:
        # Filter for: SQLite Correct AND Model Incorrect
        gap_mask = (prepared_base[SQLITE_CORRECT_COLUMN] == True) & (
            prepared_base["results_equal"] == False
        )
        pending_rows = pending_rows[gap_mask]
    # --- CHANGE END ---

    if max_rows is not None:
        print(f"\n[Sampling] Sampling {max_rows} rows for enrichment...", flush=True)
        pending_rows = _sample_uniformly_across_experiments(pending_rows, max_rows)

    # Initialize generator early so it's available for all scoring operations
    generator = EvaluationInsightsGenerator(
        model_name=model_name,
        provider=provider,
        judge_max_tokens=judge_max_tokens,
        aggregator_max_tokens=aggregator_max_tokens,
        temperature=temperature,
    )

    if not pending_rows.empty:
        # Note: We still filter != True here to be safe, though the gap_mask
        # above specifically checks for == False if enabled.
        needs_classification = pending_rows[pending_rows["results_equal"] != True]
        predictions = _rows_to_predictions(needs_classification)

        if predictions:
            judgments = generator.judge_predictions(predictions)
        else:
            judgments = []

        judgment_map = {
            judgment[CLASSIFICATION_KEY_FIELD]: judgment for judgment in judgments
        }

        keys_that_need_classification = set(needs_classification[MERGE_KEY_COLUMN])
        keys_that_got_classification = set(judgment_map.keys())
        failed_classification_keys = (
            keys_that_need_classification - keys_that_got_classification
        )

        pending_rows = pending_rows.copy()
        pending_rows[ERROR_CLASS_COLUMN] = pending_rows[MERGE_KEY_COLUMN].map(
            lambda key: judgment_map[key]["category"]
            if key in judgment_map
            else (
                CLASSIFICATION_FAILED_SENTINEL
                if key in failed_classification_keys
                else pd.NA
            )
        )
        pending_rows[EXPLANATION_COLUMN] = pending_rows[MERGE_KEY_COLUMN].map(
            lambda key: judgment_map[key]["explanation"]
            if key in judgment_map
            else (
                "Failed to parse LLM judge output"
                if key in failed_classification_keys
                else pd.NA
            )
        )

        prepared_enriched = pd.concat(
            [prepared_enriched, pending_rows], ignore_index=True
        )

    # === SCORE WITHOUT EXE ===
    # Pre-filter for bird_mini_dev_sqlite with mysql/postgres (for proxy evaluation)
    score_without_exe_target_mask = (
        prepared_base["dataset_name"] == "bird_mini_dev_sqlite"
    ) & (prepared_base["gen_type"].isin(["mysql", "postgres", "postgresql"]))

    score_without_exe_pending = prepared_base[score_without_exe_target_mask]

    if max_rows is not None:
        print(f"[Score Without Exe] Sampling {max_rows} rows for scoring...", flush=True)
        score_without_exe_pending = _sample_uniformly_across_experiments(
            score_without_exe_pending, max_rows
        )

    if not score_without_exe_pending.empty:
        predictions = _rows_to_predictions_for_scoring_without_exe(
            score_without_exe_pending
        )
        scores = generator.score_predictions_without_exe(predictions)

        score_map = {
            score[CLASSIFICATION_KEY_FIELD]: score["score"] for score in scores
        }

        score_without_exe_pending = score_without_exe_pending.copy()
        score_without_exe_pending[JUDGE_SCORE_WITHOUT_EXE_COLUMN] = (
            score_without_exe_pending[MERGE_KEY_COLUMN].map(
                lambda key: score_map.get(key, pd.NA)
            )
        )

        prepared_enriched = pd.concat(
            [prepared_enriched, score_without_exe_pending], ignore_index=True
        )

    # === TRANSPILATION EVALUATION ===
    if enable_transpilation:
        # Filter for cross-dialect experiments (bird_mini_dev_sqlite with mysql/postgres targets)
        transpilation_target_mask = (
            prepared_base["dataset_name"] == "bird_mini_dev_sqlite"
        ) & (prepared_base["gen_type"].isin(["mysql", "postgres", "postgresql"]))

        transpilation_pending = prepared_base[transpilation_target_mask]

        if max_rows is not None:
            print(f"[Transpilation] Sampling {max_rows} rows for transpilation...", flush=True)
            transpilation_pending = _sample_uniformly_across_experiments(
                transpilation_pending, max_rows
            )

        if not transpilation_pending.empty:
            # Create inference engine for LLM transpilation
            inference_engine = CrossProviderInferenceEngineWithMoreRISTModels(
                model=transpilation_model_name,
                provider=transpilation_provider,
                data_classification_policy=["public"],
                max_tokens=4096,
                temperature=0.0,
                seed=42,
            )

            # ===== PHASE 1: Process sqlglot methods (no LLM calls) =====
            transpilation_results = []

            for idx, row in transpilation_pending.iterrows():
                result = {
                    MERGE_KEY_COLUMN: row[MERGE_KEY_COLUMN],
                    "transpilation_enriched": True,  # Mark this row as enriched
                }

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

                sqlite_db_path = get_db_path(dataset_path, db_id)

                # Method 1: Transpile prediction back to source
                pred_to_source = evaluate_sqlglot_pred_to_source(
                    predicted_sql=predicted_sql,
                    gold_sql=gold_sql,
                    read_dialect=read_dialect,
                    sqlite_db_path=sqlite_db_path,
                )
                result[SQLGLOT_PRED_TO_SOURCE_COLUMN] = pred_to_source

                # Method 2: Transpile gold to target (optionally with schema mapping)
                gold_to_target = evaluate_sqlglot_gold_to_target(
                    gold_sql=gold_sql,
                    predicted_sql=predicted_sql,
                    write_dialect=write_dialect,
                    sqlite_db_path=sqlite_db_path,
                    apply_schema_mapping=sqlglot_use_schema_mapping,
                )
                result[SQLGLOT_GOLD_TO_TARGET_COLUMN] = gold_to_target

                transpilation_results.append(result)

            # ===== PHASE 2: Batch LLM transpilation =====
            # Collect all LLM requests
            llm_requests = []
            llm_metadata = []  # Store metadata for post-processing
            gold_sql_map = {}  # Map merge_key → gold_sql for O(1) lookup

            for idx, row in transpilation_pending.iterrows():
                full_prompt = row.get("full_prompt", "")
                schema = extract_schema_from_prompt(full_prompt)

                if schema:
                    gen_type = row["gen_type"]
                    write_dialect = "postgres" if gen_type == "postgresql" else gen_type
                    merge_key = row[MERGE_KEY_COLUMN]

                    llm_requests.append(
                        {
                            "gold_sql": row["gold_sql"],
                            "schema": schema,
                            "source_dialect": "sqlite",
                            "target_dialect": write_dialect,
                            "_key": merge_key,
                        }
                    )

                    llm_metadata.append(
                        {
                            "merge_key": merge_key,
                            "predicted_sql": row["predicted_sql"],
                            "write_dialect": write_dialect,
                            "sqlite_db_path": get_db_path(
                                row.get("dataset_path", ""), row["db_id"]
                            ),
                        }
                    )

                    # Store for fast lookup
                    gold_sql_map[merge_key] = row["gold_sql"]
                else:
                    # Schema extraction failed - log warning
                    merge_key = row[MERGE_KEY_COLUMN]
                    print(f"[Warning] Schema extraction failed for {merge_key}", flush=True)

            # Batch transpile all queries
            if llm_requests:
                print(
                    f"[Transpilation] Batch transpiling {len(llm_requests)} queries...",
                    flush=True,
                )
                llm_results = batch_transpile_with_llm(
                    llm_requests,
                    inference_engine,
                    include_mapping_instructions=llm_use_mapping_instructions,
                )
                print(
                    "[Transpilation] Batch transpilation complete, evaluating results...",
                    flush=True,
                )

                # Process LLM results and evaluate
                llm_evaluations = {}
                for idx, (meta, (transpiled_sql, error)) in enumerate(
                    zip(llm_metadata, llm_results)
                ):
                    print(
                        f"[Evaluation] Processing result {idx + 1}/{len(llm_metadata)}...",
                        flush=True,
                    )
                    merge_key = meta["merge_key"]

                    if error or not transpiled_sql:
                        llm_evaluations[merge_key] = {
                            LLM_GOLD_TO_TARGET_COLUMN: pd.NA,
                        }
                        continue

                    # Test 1: Does LLM-transpiled match prediction?
                    matches_prediction = pd.NA

                    try:
                        q1 = QueryInput(
                            query=transpiled_sql, backend=meta["write_dialect"]
                        )
                        q2 = QueryInput(
                            query=meta["predicted_sql"], backend=meta["write_dialect"]
                        )

                        metric_result = generic_dialect_metric(
                            q1,
                            q2,
                            db_path=meta["sqlite_db_path"],
                            load_data=False,
                            source_db_type="sqlite",
                        )

                        # Check execution status
                        if not metric_result.both_executed:
                            if not metric_result.query1_executed:
                                # Transpiled gold failed to execute
                                matches_prediction = pd.NA  # Bad transpilation
                            elif not metric_result.query2_executed:
                                # Original prediction failed to execute
                                matches_prediction = False  # Bad prediction (not transpilation's fault)
                            else:
                                # Both failed
                                matches_prediction = pd.NA
                        else:
                            # Both executed successfully
                            if metric_result.results_equal is None:
                                matches_prediction = pd.NA  # Comparison error
                            else:
                                matches_prediction = metric_result.results_equal  # True or False

                    except Exception:
                        # DB connection errors, timeouts, etc.
                        matches_prediction = pd.NA
                    finally:
                        llm_evaluations[merge_key] = {
                            LLM_GOLD_TO_TARGET_COLUMN: matches_prediction,
                        }

            # ===== PHASE 3: Merge all results =====
            # Add LLM results to transpilation_results
            for result in transpilation_results:
                merge_key = result[MERGE_KEY_COLUMN]
                if merge_key in llm_evaluations:
                    result.update(llm_evaluations[merge_key])
                else:
                    result[LLM_GOLD_TO_TARGET_COLUMN] = pd.NA

            transpilation_df = pd.DataFrame(transpilation_results)
            transpilation_df = transpilation_df.set_index(MERGE_KEY_COLUMN)

            # Merge back into prepared_enriched
            prepared_enriched = prepared_enriched.set_index(MERGE_KEY_COLUMN)
            prepared_enriched.update(transpilation_df)
            prepared_enriched = prepared_enriched.reset_index()

    # Build column list - only include transpilation_enriched if it exists
    enrichment_columns = [
        MERGE_KEY_COLUMN,
        ERROR_CLASS_COLUMN,
        EXPLANATION_COLUMN,
        JUDGE_SCORE_WITHOUT_EXE_COLUMN,
        SQLGLOT_PRED_TO_SOURCE_COLUMN,
        SQLGLOT_GOLD_TO_TARGET_COLUMN,
        LLM_GOLD_TO_TARGET_COLUMN,
    ]

    if "transpilation_enriched" in prepared_enriched.columns:
        enrichment_columns.insert(1, "transpilation_enriched")  # Add after MERGE_KEY_COLUMN

    enrichment_df = (
        prepared_enriched[enrichment_columns]
        .drop_duplicates(MERGE_KEY_COLUMN, keep="last")
        .set_index(MERGE_KEY_COLUMN)
    )

    ordered = prepared_base.copy()
    reindexed = enrichment_df.reindex(ordered[MERGE_KEY_COLUMN])

    # Assign transpilation_enriched if it exists
    if "transpilation_enriched" in reindexed.columns:
        ordered["transpilation_enriched"] = reindexed["transpilation_enriched"].values

    ordered[ERROR_CLASS_COLUMN] = reindexed[ERROR_CLASS_COLUMN].values
    ordered[EXPLANATION_COLUMN] = reindexed[EXPLANATION_COLUMN].values
    ordered[JUDGE_SCORE_WITHOUT_EXE_COLUMN] = reindexed[
        JUDGE_SCORE_WITHOUT_EXE_COLUMN
    ].values
    ordered[SQLGLOT_PRED_TO_SOURCE_COLUMN] = reindexed[
        SQLGLOT_PRED_TO_SOURCE_COLUMN
    ].values
    ordered[SQLGLOT_GOLD_TO_TARGET_COLUMN] = reindexed[
        SQLGLOT_GOLD_TO_TARGET_COLUMN
    ].values
    ordered[LLM_GOLD_TO_TARGET_COLUMN] = reindexed[LLM_GOLD_TO_TARGET_COLUMN].values

    # We already computed this early on, so we can just assign it from prepared_base
    # or re-assign to ensure consistency (since we copied prepared_base to ordered)
    ordered[SQLITE_CORRECT_COLUMN] = prepared_base[SQLITE_CORRECT_COLUMN]

    ordered = ordered.drop(columns=[MERGE_KEY_COLUMN])

    # Save full file with all columns
    full_path = enriched_path.parent / (enriched_path.stem + "_full.csv")
    print(f"\n[Complete] Writing full enriched results to {full_path}...", flush=True)
    ordered.to_csv(full_path, index=False)
    print(f"[Complete] Full results saved to {full_path}", flush=True)

    # Create analysis file: only enriched rows + only necessary columns
    analysis_columns = [
        "model_name",
        "question_id",
        "dataset_name",
        "gen_type",
        "results_equal",
        "judge_score_without_exe",
        "sqlglot_pred_to_source",
        "sqlglot_gold_to_target",
        "llm_gold_to_target",
        "transpilation_enriched",
    ]

    # Filter to enriched rows + native evaluation rows (for comparison)
    # Native rows: dataset_name ends with _{gen_type} (e.g., bird_mini_dev_mysql with gen_type=mysql)
    native_mask = ordered.apply(
        lambda row: str(row["dataset_name"]).endswith(f"_{row['gen_type']}"), axis=1
    )

    if "transpilation_enriched" in ordered.columns:
        # Include: enriched cross-dialect rows OR native evaluation rows
        enriched_mask = ordered["transpilation_enriched"] == True
        analysis_mask = enriched_mask | native_mask
        analysis_df = ordered[analysis_mask].copy()

        n_enriched = enriched_mask.sum()
        n_native = native_mask.sum()
        n_total = analysis_mask.sum()
        print(
            f"\n[Analysis] Filtered to {n_total} rows: "
            f"{n_enriched} enriched (cross-dialect) + {n_native} native (for comparison)",
            flush=True,
        )
    else:
        # Fallback: filter to rows with any transpilation result OR native rows
        transpilation_cols = [SQLGLOT_PRED_TO_SOURCE_COLUMN, SQLGLOT_GOLD_TO_TARGET_COLUMN, LLM_GOLD_TO_TARGET_COLUMN]
        available_cols = [col for col in transpilation_cols if col in ordered.columns]

        if available_cols:
            enriched_mask = ordered[available_cols].notna().any(axis=1)
            analysis_mask = enriched_mask | native_mask
            analysis_df = ordered[analysis_mask].copy()
            print(
                f"\n[Analysis] Filtered to {len(analysis_df)} rows "
                f"(enriched + native, heuristic - no explicit flag)",
                flush=True,
            )
        else:
            analysis_df = ordered.copy()
            print(f"\n[Analysis] No transpilation - using all {len(analysis_df)} rows", flush=True)

    # Select only necessary columns (skip missing ones)
    available_columns = [col for col in analysis_columns if col in analysis_df.columns]
    analysis_df = analysis_df[available_columns]

    print(f"[Analysis] Writing analysis file to {enriched_path}...", flush=True)
    analysis_df.to_csv(enriched_path, index=False)
    print(
        f"[Complete] Enrichment complete!\n"
        f"  - Full results: {full_path}\n"
        f"  - Analysis file: {enriched_path}",
        flush=True
    )
    return enriched_path


def main() -> None:
    parser = ArgumentParser(description="Enrich results with judge classifications.")
    parser.add_argument(
        "--results-dir", required=True, help="Path to a results directory"
    )
    parser.add_argument("--model-name", default="gpt-oss-120b")
    parser.add_argument("--provider", default="rits")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--aggregator-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of pending rows processed (for quick tests)",
    )
    parser.add_argument(
        "--classify-only-gap-instances",
        action="store_true",
        help="Only classify instances where SQLite was correct but the model result was incorrect.",
    )
    parser.add_argument(
        "--enable-transpilation",
        action="store_true",
        help="Enable SQL transpilation evaluation (sqlglot and LLM-based methods).",
    )
    parser.add_argument(
        "--transpilation-model-name",
        default="gpt-oss-120b",
        help="Model name for LLM transpilation (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--transpilation-provider",
        default="rits",
        help="Provider for LLM transpilation (default: rits)",
    )
    parser.add_argument(
        "--sqlglot-use-schema-mapping",
        action="store_true",
        help="Enable fuzzy schema mapping heuristics on top of sqlglot (default: False, pure sqlglot evaluation)",
    )
    parser.add_argument(
        "--llm-use-mapping-instructions",
        action="store_true",
        help="Include detailed identifier mapping instructions in LLM prompt (default: False, pure LLM evaluation)",
    )
    args = parser.parse_args()

    enrich_results_directory(
        Path(args.results_dir),
        model_name=args.model_name,
        provider=args.provider,
        judge_max_tokens=args.judge_max_tokens,
        aggregator_max_tokens=args.aggregator_max_tokens,
        temperature=args.temperature,
        max_rows=args.max_rows,
        classify_only_gap_instances=args.classify_only_gap_instances,
        enable_transpilation=args.enable_transpilation,
        transpilation_model_name=args.transpilation_model_name,
        transpilation_provider=args.transpilation_provider,
        sqlglot_use_schema_mapping=args.sqlglot_use_schema_mapping,
        llm_use_mapping_instructions=args.llm_use_mapping_instructions,
    )


if __name__ == "__main__":
    main()
