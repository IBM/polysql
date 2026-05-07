"""Simple evaluation loop for NL2DSL datasets using NL2DSLModel.

This script demonstrates how to:
1. Load N examples from a dataset (BIRD format or NL2DSL format)
2. Generate SQL/Ibis code using NL2DSLModel
3. Track and display results

Usage:
    # Use NL2DSL dataset format (with schemas)
    python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --n 10

    # Use specific model and engine
    python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --model gpt-4 --engine openai --n 5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from polysql.evaluation.backends.connections import ibis_schema_getter, schema_getter
from polysql.evaluation.prompts.sql import SQL_GEN_TYPES, parse_sql_from_response


def _strip_to_import_ibis(code: str) -> str:
    """
    Strip model output to remove schema definitions and explanations, keeping only query logic.

    The schema will be prepended separately, so we need to remove any schema
    definitions that the model included to avoid duplication.

    Strategy:
    1. Remove markdown code blocks (```python, ```)
    2. If no 'import ibis', return as-is (fail fast)
    3. If 'import ibis' exists, look for where actual query code starts
    4. Query code typically starts after table definitions (ibis.table calls)
    5. Skip explanatory text - look for actual Python variable assignments
    """
    if not isinstance(code, str):
        return code

    # Step 1: Remove markdown code blocks
    import re

    # Remove opening markdown blocks (```python, ```py, ```)
    code = re.sub(r"^```(?:python|py)?\s*$", "", code, flags=re.MULTILINE)
    # Remove closing markdown blocks (```)
    code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE)
    # Remove inline markdown if at start/end
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
    if code.endswith("```"):
        code = "\n".join(code.rsplit("\n", 1)[:-1]) if "\n" in code else code[:-3]

    lines = code.split("\n")
    query_start_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue
        # Skip import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        # Skip table definitions (schema)
        if "=ibis.table(" in stripped.replace(" ", "") or "= ibis.table(" in stripped:
            continue
        # Skip markdown artifacts that weren't caught earlier
        if stripped.startswith("```"):
            continue
        # Skip explanatory text patterns (ENHANCED)
        # - Numbered lists: "1. Do something", "2. Then do"
        # - Bullet points: "- First step"
        # - Plain English explanations without code markers
        # - Common model explanation patterns: "I need to...", "To do this...", etc.
        if (
            (len(stripped) > 0 and stripped[0].isdigit() and ". " in stripped[:5])
            or stripped.startswith(("- ", "* ", "• "))
            or (
                stripped.startswith(
                    (
                        "Let ",
                        "The ",
                        "This ",
                        "We ",
                        "I ",
                        "First",
                        "Then",
                        "Next",
                        "Now",
                        "Note:",
                        "Here",
                        "To ",
                        "In ",
                        "For ",
                        "Based ",
                        "Given ",
                        "Since ",
                    )
                )
                and "=" not in stripped
                and not stripped.endswith(":")
            )
        ):
            continue
        # Skip lines that look like explanations (no assignment or known patterns)
        # Code lines typically have '=' for assignment or start with known keywords
        if "=" not in stripped and not stripped.startswith(
            ("result", "f ", "s ", "joinchain", "agg", "g ", "j", "j1", "j2", "j3")
        ):
            # Additional check: if line contains common English words without code markers
            english_indicators = [
                "need to",
                "want to",
                "should",
                "will",
                "can",
                "must",
                "going to",
                "have to",
                "break this",
            ]
            if any(indicator in stripped.lower() for indicator in english_indicators):
                continue
            # If line doesn't have parentheses or dots (common in code), likely explanation
            if "(" not in stripped and "." not in stripped:
                continue
        # Found the start of actual query code
        query_start_idx = i
        break

    # If we found query code, return from that point
    if query_start_idx is not None:
        return "\n".join(lines[query_start_idx:])

    # Fallback: return everything from first 'import ibis' (old behavior)
    lowered = code.lower()
    anchor = "import ibis"
    idx = lowered.find(anchor)
    if idx == -1:
        return code
    return code[idx:]


from polysql.evaluation.config.datasets import DatasetConfig
from polysql.evaluation.config.types import (
    EvaluationResult,
    PredictionResult,
)
from polysql.evaluation.core.model import NL2DSLModel
from polysql.evaluation.metrics.dialect_comparison import (
    QueryInput,
    generic_dialect_metric,
)
from polysql.evaluation.prompts.base import get_prompts
from polysql.evaluation.utils.data_loading import load_standard_dataset

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_evaluation_loop(
    dataset_path: str,
    model_name: str,
    n_examples: int,
    gen_type: str,
    model_engine: str,
    instructions_level: int = 1,
    verbose: bool = False,
    load_data: bool = True,
    dataset_config: Optional[DatasetConfig] = None,
) -> EvaluationResult:
    """
    Run evaluation loop on N examples from dataset.

    Args:
        dataset_path: Path to dataset JSON file
        model_name: Model identifier (e.g., "gpt-4", "granite-20b-code-instruct")
        n_examples: Number of examples to evaluate
        gen_type: Generation type ("sql" or "ibis")
        model_engine: Model provider ("openai", "watsonx", "rits", etc.)
        instructions_level: Level of instructions (1, 2, or 3)
        verbose: Whether to print detailed output
        load_data: Whether to load data into backends (default: True)
        dataset_config: Dataset configuration (required)

    Returns:
        EvaluationResult Pydantic model with predictions, errors, and statistics
    """
    # Dataset config is required
    assert dataset_config is not None, (
        "dataset_config is required - no default config available"
    )

    print(f"Loading dataset from: {dataset_config.data_path}")
    dataset = load_standard_dataset(
        dataset_path=dataset_config.data_path,
        sample_size=n_examples,
        random_state=0,
        dataset_config=dataset_config,
    )
    print(f"Loaded {len(dataset)} examples")

    print("\nInitializing NL2DSLModel:")
    print(f"  Model: {model_name}")
    print(f"  Engine: {model_engine}")
    print(f"  Gen Type: {gen_type}")

    model = NL2DSLModel(
        model_name_or_path=model_name,
        gen_type=gen_type,
        model_engine=model_engine,
    )

    predictions_list: List[PredictionResult] = []
    errors_list: List[str] = []
    total = len(dataset)
    generated = 0
    generation_failed = 0
    executed = 0
    execution_failed = 0
    correct = 0

    print(f"\n{'=' * 80}")
    print(f"Starting evaluation on {len(dataset)} examples")
    print(f"{'=' * 80}\n")

    if verbose:
        print("Generating prompts...")

    prompts = get_prompts(
        dataset=dataset,
        instruction_level=instructions_level,
        gen_type=gen_type,
        schema_getter=schema_getter,
        ibis_schema_getter=ibis_schema_getter,
    )
    prompt_contents = [prompt[0]["content"] for prompt in prompts]
    predictions = model.api_completion(prompt_contents)

    # if its predicted sql, not ibis, parse sql from response

    if gen_type in SQL_GEN_TYPES:
        predictions = [
            parse_sql_from_response(pred) if pred else "Prediction not returned"
            for pred in predictions
        ]

    # Debug: Check for None predictions
    none_count = sum(1 for p in predictions if p is None)
    if none_count > 0:
        print(
            f"\n⚠️  WARNING: Model returned {none_count}/{len(predictions)} None predictions"
        )
        print(f"   Model: {model_name}, Engine: {model_engine}")
        none_indices = [i for i, p in enumerate(predictions) if p is None]
        print(f"   Indices with None: {none_indices[:10]}")  # Show first 10

        # Show sample of non-None predictions for comparison
        non_none_samples = [p for p in predictions if p is not None][:3]
        print("   Sample non-None predictions:")
        for i, sample in enumerate(non_none_samples):
            preview = str(sample)[:100] if sample else "Empty"
            print(f"     [{i}]: {preview}...")

    # Store raw predictions before any processing
    raw_predictions = predictions.copy()

    if gen_type.startswith("sqlite-"):
        # Handle sqlite-{dialect} pattern (e.g., sqlite-postgres, sqlite-duckdb)

        # Validate that source database is actually SQLite
        if dataset_config.source_db_type != "sqlite":
            raise ValueError(
                f"Gen type '{gen_type}' requires SQLite source database, "
                f"but dataset has source_db_type='{dataset_config.source_db_type}'. "
                f"Use a different gen_type or a SQLite-based dataset."
            )

        # Extract target dialect from gen_type
        target_dialect = gen_type.split("-", 1)[
            1
        ]  # e.g., "postgres" from "sqlite-postgres"

        # Store raw SQLite predictions before transpilation
        raw_predictions = predictions.copy()

        # Transpile SQLite queries to target dialect using sqlglot
        import sqlglot
        import re

        transpiled = []
        for query in predictions:
            try:
                # Strip markdown code blocks if present
                cleaned = query.strip()
                cleaned = re.sub(r"```sql\s*", "", cleaned)
                cleaned = re.sub(r"```\s*", "", cleaned)
                cleaned = cleaned.strip()

                # Transpile using sqlglot
                result = sqlglot.transpile(
                    cleaned,
                    read="sqlite",
                    write=target_dialect,
                    pretty=True
                )[0]
                transpiled.append(result)
            except Exception as e:
                print(f"Error transpiling SQL: {e}\nQuery:\n{query}")
                transpiled.append(f"TRANSPILATION_ERROR: {e}")

        predictions = transpiled
        print(f"Transpiled {len(predictions)} SQLite queries to {target_dialect}")

    if verbose:
        print(
            f"Got {len(predictions)} predictions from model!\nmoving on to the metric"
        )
    for i, (instance, prompt) in enumerate(zip(dataset, prompts), 1):
        # print(f"\n[Example {i}/{total}] Starting evaluation...")

        try:
            prediction = predictions[i - 1]
            raw_prediction = raw_predictions[i - 1]
            question = instance[dataset_config.nl_query_field]
            db_id = instance.get(dataset_config.db_id_field, "unknown")
            db_path = instance.get(dataset_config.db_path_field, "")
            if verbose:
                print(f"  DB: {db_id}, Path: {db_path}")

            gold_sql = instance.get(dataset_config.gold_query_field, "")

            generated += 1

            # Use sample-specific dialect if available, otherwise use config default
            gold_dialect = instance.get(
                "gold_query_dialect", dataset_config.gold_query_dialect
            )
            gold_query = QueryInput(
                query=gold_sql,
                backend=gold_dialect,
                query_type=dataset_config.gold_query_type,
            )

            # Create predicted query - fail immediately if prediction is None
            # Extract backend from gen_type
            # Handle sqlite-{dialect} pattern (e.g., sqlite-postgres → postgres)
            if gen_type.startswith("sqlite-"):
                backend = gen_type.split("-", 1)[1]
            else:
                backend = gen_type.replace("-ss", "")

            pred_query = QueryInput(
                query=prediction,
                backend=backend,
                query_type="sql",
            )

            if verbose:
                print(
                    f"  Comparing queries (gold: {gold_query.backend}, pred: {pred_query.backend})..."
                )
            metric_result = generic_dialect_metric(
                gold_query,
                pred_query,
                db_path,
                load_data=load_data,
                dataset_name=dataset_config.name,
                source_db_type=dataset_config.source_db_type,
            )
            if verbose:
                print("  Comparison complete!")

            # Check for data loading timeout errors - these should fail the entire experiment
            data_loading_timeout_indicators = [
                "Query comparison timed out",
                "Pipeline execution failed at `step=normalize`",
                "Pipeline execution failed at `step=load`",
            ]

            if metric_result.query2_error:
                for indicator in data_loading_timeout_indicators:
                    if indicator in metric_result.query2_error:
                        error_msg = (
                            f"DATA LOADING TIMEOUT detected for {db_id}. "
                            f"This indicates the {dataset_config.source_db_type} → {gen_type} "
                            f"conversion exceeded the timeout threshold. "
                            f"This is a systemic issue that invalidates accuracy metrics. "
                            f"Error: {metric_result.query2_error[:200]}"
                        )
                        raise RuntimeError(error_msg)

            if metric_result.both_executed:
                executed += 1
                if metric_result.results_equal:
                    correct += 1
                    if verbose:
                        print("✓ Results match!")
                else:
                    if verbose:
                        print("✗ Results don't match")
                        print(f"  Gold shape: {metric_result.query1_shape}")
                        print(f"  Pred shape: {metric_result.query2_shape}")
            else:
                execution_failed += 1
                if verbose:
                    if not metric_result.query1_executed:
                        print(f"✗ Gold query failed: {metric_result.query1_error}")
                    if not metric_result.query2_executed:
                        print(f"✗ Predicted query failed: {metric_result.query2_error}")

            # Convert DataFrames to JSON-serializable format (list of dicts)
            # Only include results when both queries executed successfully
            gold_result_json = None
            predicted_result_json = None

            if metric_result.both_executed:
                if metric_result.query1_result is not None:
                    try:
                        gold_result_json = metric_result.query1_result.to_dict(
                            "records"
                        )
                    except Exception:
                        pass

                if metric_result.query2_result is not None:
                    try:
                        predicted_result_json = metric_result.query2_result.to_dict(
                            "records"
                        )
                    except Exception:
                        pass

            if (
                "SQLite objects created in a thread can only be used in that same thread."
                in str(metric_result.query2_error)
            ):
                print(
                    "THREADING ERROR detected: SQLite objects created in a thread can only be used in that same thread. "
                    "This indicates that the evaluation loop is using multiple threads with SQLite, "
                    "which is not supported. Please run the evaluation with a single worker (no parallelism) when using SQLite datasets."
                )
                exit(1)

            predictions_list.append(
                PredictionResult(
                    question_id=i,
                    db_id=db_id,
                    question=question,
                    gold_sql=gold_sql,
                    predicted_sql=prediction,
                    predicted_code=None,  # No Ibis/Substrait support
                    both_executed=metric_result.both_executed,
                    results_equal=metric_result.results_equal,
                    gold_error=metric_result.query1_error,
                    pred_error=metric_result.query2_error,
                    gold_result=gold_result_json,
                    predicted_result=predicted_result_json,
                    full_prompt=prompt[0]["content"],
                )
            )

        except RuntimeError as e:
            # Re-raise RuntimeErrors (like data loading timeouts) to fail the entire experiment
            if "DATA LOADING TIMEOUT" in str(e):
                raise
            # For other RuntimeErrors, treat as evaluation error
            error_msg = f"Example {i}: Runtime error during evaluation: {e}"
            print(f"✗ {error_msg}")
            errors_list.append(error_msg)
            execution_failed += 1

            if (
                "SQLite objects created in a thread can only be used in that same thread."
                in str(e)
            ):
                print(
                    "THREADING ERROR detected: SQLite objects created in a thread can only be used in that same thread. "
                    "This indicates that the evaluation loop is using multiple threads with SQLite, "
                    "which is not supported. Please run the evaluation with a single worker (no parallelism) when using SQLite datasets."
                )
                exit(1)

            # Still append a result with error information
            predictions_list.append(
                PredictionResult(
                    question_id=i,
                    db_id=instance.get(dataset_config.db_id_field, "unknown"),
                    question=instance.get(dataset_config.nl_query_field, ""),
                    gold_sql=instance.get(dataset_config.gold_query_field, ""),
                    predicted_sql=predictions[i - 1]
                    if i - 1 < len(predictions)
                    else "",
                    predicted_code=raw_predictions[i - 1]
                    if i - 1 < len(raw_predictions)
                    else None,
                    both_executed=False,
                    results_equal=False,
                    gold_error=None,
                    pred_error=f"EVALUATION_ERROR: {e}",
                    full_prompt=prompt[0]["content"] if prompt else "",
                )
            )

        except Exception as e:
            # Catch any other unexpected errors to ensure experiment continues
            error_msg = f"Example {i}: Unexpected error during evaluation: {e}"
            print(f"✗ {error_msg}")
            errors_list.append(error_msg)
            execution_failed += 1

            if (
                "SQLite objects created in a thread can only be used in that same thread."
                in str(e)
            ):
                print(
                    "THREADING ERROR detected: SQLite objects created in a thread can only be used in that same thread. "
                    "This indicates that the evaluation loop is using multiple threads with SQLite, "
                    "which is not supported. Please run the evaluation with a single worker (no parallelism) when using SQLite datasets."
                )
                exit(1)

            # Still append a result with error information
            predictions_list.append(
                PredictionResult(
                    question_id=i,
                    db_id=instance.get(dataset_config.db_id_field, "unknown"),
                    question=instance.get(dataset_config.nl_query_field, ""),
                    gold_sql=instance.get(dataset_config.gold_query_field, ""),
                    predicted_sql=predictions[i - 1]
                    if i - 1 < len(predictions)
                    else "",
                    predicted_code=raw_predictions[i - 1]
                    if i - 1 < len(raw_predictions)
                    else None,
                    both_executed=False,
                    results_equal=False,
                    gold_error=None,
                    pred_error=f"EVALUATION_ERROR: {e}",
                    full_prompt=prompt[0]["content"] if prompt else "",
                )
            )

    accuracy = correct / total if total > 0 else 0.0

    # print(f"\n{'=' * 80}")
    # print("Evaluation Complete")
    # print(f"{'=' * 80}")
    # print(f"Total examples: {total}")
    # print("\nGeneration:")
    # print(f"  Generated: {generated}")
    # print(f"  Failed: {generation_failed}")
    # print("\nExecution:")
    # print(f"  Both executed: {executed}")
    # print(f"  Execution failed: {execution_failed}")
    # print("\nAccuracy:")
    # print(f"  Correct results: {correct}/{total}")
    # print(f"  Accuracy: {accuracy * 100:.1f}%")

    return EvaluationResult(
        exp_config=None,
        predictions=predictions_list,
        errors=errors_list,
        total=total,
        generated=generated,
        generation_failed=generation_failed,
        executed=executed,
        execution_failed=execution_failed,
        correct=correct,
        accuracy=accuracy,
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate NL2DSL models on datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on NL2DSL dataset
  python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --n 10

  # Evaluate Granite model
  python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --model granite-20b-code-instruct --engine watsonx --n 5

  # Generate Ibis code instead of SQL
  python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --gen-type ibis --n 10

  # Quiet mode (minimal output)
  python examples/evaluation_loop.py --dataset tests/assets/dataset_dev_100_0_none.json --n 20 --quiet
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset JSON file (e.g., tests/assets/dataset_dev_100_0_none.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="Model name (e.g., 'gpt-4', 'granite-20b-code-instruct')",
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of examples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--gen-type",
        type=str,
        default="sqlite",
        choices=[
            "sqlite",
            "duckdb",
            "postgres",
            "mysql",
            "bigquery",
            "snowflake",
            "datafusion",
            "clickhouse",
        ],
        help="Generation type/SQL dialect (default: duckdb). Options: sqlite, duckdb, postgres, mysql, bigquery, snowflake, datafusion, clickhouse, or sqlite-{dialect} for cross-dialect transpilation",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="rits",
        help="Model engine/provider (default: openai). Options: openai, watsonx, rits, etc.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output (only summary)"
    )
    parser.add_argument(
        "--output", type=str, help="Optional: Save results to JSON file"
    )

    parser.add_argument(
        "--instructions-level",
        type=int,
        default=22,
        help="Instruction level: two-digit format XY (X=COT 1-2, Y=dialect 1-3). "
        "Valid: 11, 12, 13, 21, 22, 23. Default: 22.",
    )

    args = parser.parse_args()

    # Validate instruction level
    from polysql.evaluation.prompts.sql import parse_instruction_level

    try:
        parse_instruction_level(args.instructions_level)
    except ValueError as e:
        parser.error(str(e))

    results = run_evaluation_loop(
        dataset_path=args.dataset,
        model_name=args.model,
        n_examples=args.n,
        gen_type=args.gen_type,
        model_engine=args.engine,
        verbose=not args.quiet,
        instructions_level=args.instructions_level,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(results.model_dump_json(indent=2))

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
