"""Run experiments across multiple models and dialects in parallel.

This script runs evaluation experiments for all combinations of models and dialects,
saving results to JSON files for later analysis. Experiments are run in parallel
using multiprocessing for faster execution.

Usage:
    # Run with default settings (parallel, auto-detect CPU count)
    python examples/run_experiments.py

    # Run with specific number of workers
    python examples/run_experiments.py --workers 4

    # Run sequentially (no parallelization)
    python examples/run_experiments.py --workers 1

    # Run specific models and dialects
    python examples/run_experiments.py --models llama-4-maverick:rits --dialects sqlite duckdb
"""

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, cast

import pandas as pd
from diskcache import Cache
from dotenv import load_dotenv

from polysql.evaluation.analysis.results import load_experiment_data
from polysql.evaluation.config.types import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentSummary,
)
from polysql.evaluation.core.evaluation import (  # noqa: E402
    run_evaluation_loop,
)
from polysql.evaluation.utils.cache_paths import cache_subdirs, migrate_legacy_caches

# Add src to path for imports (must be before local imports)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load environment variables
load_dotenv()
migrate_legacy_caches()

# Suppress verbose logging from third-party libraries
logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("pyspark").setLevel(logging.WARNING)

# Initialize cache for experiments
CACHE_DIR = cache_subdirs()["experiments"]
experiments_cache = Cache(str(CACHE_DIR))


def create_experiment_configs(
    models: List[Tuple[str, str]],
    dialects: List[str],
    dataset_names: List[str],
    instruction_levels: List[int],
    n_examples: int,
) -> List[ExperimentConfig]:
    """Create all experiment configurations from the grid."""
    configs = []
    from polysql.evaluation.config.datasets import get_dataset_config

    for (model_name, model_engine), dialect, instruction_level, dataset_name in product(
        models, dialects, instruction_levels, dataset_names
    ):
        dataset_config = get_dataset_config(dataset_name)

        config = ExperimentConfig(
            model_name=model_name,
            model_engine=model_engine,
            gen_type=dialect,
            dataset_name=dataset_config.name,
            instructions_level=instruction_level,
            dataset_path=dataset_config.data_path,
            n_examples=n_examples,
        )
        configs.append(config)
    return configs


def run_single_experiment(
    config: ExperimentConfig,
    cache: Optional[Cache] = None,
    load_data: bool = True,
) -> Tuple[ExperimentSummary, Optional[EvaluationResult]]:
    """
    Run a single experiment, with optional caching.

    Args:
        config: Experiment configuration.
        cache: Optional diskcache.Cache object.
        load_data: Whether to load data into backends (default: True).

    Returns:
        Tuple of (ExperimentSummary, EvaluationResult or None if failed)
    """

    # Generate cache key from query, backend, db_path, and dataset_name
    import hashlib

    cache_key = hashlib.sha256(
        f"{config.model_name}|{config.model_engine}|{config.gen_type}|{config.dataset_path}|{config.dataset_name}|{config.n_examples}|{config.instructions_level}".encode()
    ).hexdigest()

    # Check cache
    if (
        cache is not None
        # and config.gen_type not in ["duckdb"]
        # and config.model_name not in ["llama-3-1-8b-instruct"]
    ):  # Skip caching for memory-intensive backends
        cached_result = cast(
            Optional[Tuple[ExperimentSummary, Optional[EvaluationResult]]],
            experiments_cache.get(cache_key),
        )
        if cached_result is not None:
            # Return cached result tuple
            return cached_result

    # If not in cache, run the experiment
    print(f"  - Running experiment (not found in cache): {config.experiment_id}")

    # Need to reload dotenv in each process

    from dotenv import load_dotenv

    load_dotenv()

    # Suppress verbose logging in worker processes
    logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
    logging.getLogger("google.cloud").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

    result = None
    try:
        # Get dataset configuration
        from polysql.evaluation.config.datasets import get_dataset_config

        dataset_config = get_dataset_config(config.dataset_name)

        # Run evaluation
        result = run_evaluation_loop(
            dataset_path=config.dataset_path,
            model_name=config.model_name,
            n_examples=config.n_examples,
            gen_type=config.gen_type,
            model_engine=config.model_engine,
            instructions_level=config.instructions_level,
            verbose=False,  # Quiet mode for parallel processing
            load_data=load_data,
            dataset_config=dataset_config,
        )

        result.exp_config = config

        # Create summary
        summary = ExperimentSummary(
            experiment_id=config.experiment_id,
            model_name=config.model_name,
            model_engine=config.model_engine,
            gen_type=config.gen_type,
            total=result.total,
            generated=result.generated,
            generation_failed=result.generation_failed,
            executed=result.executed,
            execution_failed=result.execution_failed,
            correct=result.correct,
            accuracy=result.accuracy,
            status="success",
            error=None,
            instructions_level=config.instructions_level,
        )

        # Store the result in the cache if caching is enabled
        experiments_cache.set(cache_key, (summary, result))

        # Force garbage collection to free memory
        return summary, result

    except Exception as e:
        # Return failure summary with no result
        summary = ExperimentSummary(
            experiment_id=config.experiment_id,
            model_name=config.model_name,
            model_engine=config.model_engine,
            gen_type=config.gen_type,
            total=0,
            generated=0,
            generation_failed=0,
            executed=0,
            execution_failed=0,
            correct=0,
            accuracy=0.0,
            status="failed",
            error=str(e),
            instructions_level=config.instructions_level,
        )

        # Don't cache failures - we want to retry them on next run
        # Only successful experiments should be cached

        # Force garbage collection even on error
        return summary, None


def run_experiments(
    configs: List[ExperimentConfig],
    output_dir: Path,
    max_workers: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    load_data: bool = True,
    save_results: bool = True,
    existing_results_dir: Optional[Path] = None,
) -> None:
    """
    Run all experiments in parallel and save results.

    Args:
        configs: List of experiment configurations
        output_dir: Directory to save results
        max_workers: Maximum number of parallel workers (default: number of CPUs)
        cache_dir: Directory to store cache
        load_data: Whether to load data into backends (default: True)
        save_results: Whether to save results to files (default: True)
        existing_results_dir: Directory with existing results to skip re-running
    """
    # Create output directory only if saving results
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cache
    cache = Cache(str(cache_dir)) if cache_dir else None

    # Filter out configs that already have results
    skipped_configs = []
    configs_to_run = []

    if existing_results_dir and existing_results_dir.exists():
        for config in configs:
            result_file = existing_results_dir / f"{config.experiment_id}.json"
            if result_file.exists():
                skipped_configs.append(config)
            else:
                configs_to_run.append(config)
    else:
        configs_to_run = configs

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(configs),
        "experiments_to_run": len(configs_to_run),
        "skipped_experiments": len(skipped_configs),
        "models": list(set((c.model_name, c.model_engine) for c in configs)),
        "dialects": list(set(c.gen_type for c in configs)),
        "dataset_paths": list(set(c.dataset_path for c in configs)),
        "n_examples": list(set(c.n_examples for c in configs)),
        "max_workers": max_workers,
    }

    if save_results:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    total_experiments = len(configs_to_run)
    summary_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    print(f"\n{'=' * 80}")
    print(f"Total experiments: {len(configs)}")
    if skipped_configs:
        print(f"  Skipping {len(skipped_configs)} (found in {existing_results_dir})")
        print(f"  Running {total_experiments} new experiments")
    else:
        print(f"  Running all {total_experiments} experiments")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {max_workers or 'auto (CPU count)'}")
    if cache:
        print(f"Cache directory: {cache_dir}")
    print(f"{'=' * 80}\n")

    # Track results for summary
    summary_list: List[ExperimentSummary] = []
    failed_experiments = []
    completed = 0

    # Load and copy skipped experiment results from existing directory
    if skipped_configs and existing_results_dir and save_results:
        import shutil

        for config in skipped_configs:
            result_file = existing_results_dir / f"{config.experiment_id}.json"
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)
                    # Copy result file to new output directory
                    shutil.copy(result_file, output_dir / result_file.name)
                    # Create summary from existing result
                    summary = ExperimentSummary(
                        experiment_id=config.experiment_id,
                        model_name=config.model_name,
                        model_engine=config.model_engine,
                        gen_type=config.gen_type,
                        total=result_data.get("total", 0),
                        generated=result_data.get("generated", 0),
                        generation_failed=result_data.get("generation_failed", 0),
                        executed=result_data.get("executed", 0),
                        execution_failed=result_data.get("execution_failed", 0),
                        correct=result_data.get("correct", 0),
                        accuracy=result_data.get("accuracy", 0.0),
                        status="success",
                        error=None,
                        instructions_level=config.instructions_level,
                    )
                    summary_list.append(summary)
                    completed += 1
                    print(
                        f"[{completed}/{len(configs)}] Loaded (existing): {config.experiment_id}"
                    )
                    print(
                        f"  ✓ Accuracy: {summary.accuracy * 100:.1f}% ({summary.correct}/{summary.executed})"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not load existing result {result_file}: {e}"
                    )

    def build_failure_summary(
        config: ExperimentConfig, error_message: str
    ) -> ExperimentSummary:
        return ExperimentSummary(
            experiment_id=config.experiment_id,
            model_name=config.model_name,
            model_engine=config.model_engine,
            gen_type=config.gen_type,
            total=0,
            generated=0,
            generation_failed=0,
            executed=0,
            execution_failed=0,
            correct=0,
            accuracy=0.0,
            status="failed",
            error=error_message,
            instructions_level=config.instructions_level,
        )

    def persist_summary_files() -> None:
        if not save_results:
            return
        summary_dicts = [summary.model_dump() for summary in summary_list]
        summary_df = pd.DataFrame(summary_dicts)
        summary_df.to_csv(summary_path, index=False)
        summary_json = {str(idx): record for idx, record in enumerate(summary_dicts)}
        summary_json_path.write_text(json.dumps(summary_json, indent=2))

    def clean_results(obj):
        if obj is pd.NaT:
            return None
        if isinstance(obj, dict):
            return {k: clean_results(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_results(v) for v in obj]
        return obj

    def handle_completion(
        config: ExperimentConfig,
        summary: ExperimentSummary,
        result: Optional[EvaluationResult],
    ) -> None:
        nonlocal completed
        completed += 1

        print(f"\n[{completed}/{total_experiments}] Completed: {config.experiment_id}")
        print(f"  Model: {config.model_name} ({config.model_engine})")
        print(f"  Dialect: {config.gen_type}")

        if summary.status == "success" and result is not None:
            print(
                f"  ✓ Accuracy: {summary.accuracy * 100:.1f}% ({summary.correct}/{summary.executed})"
            )
            if save_results:
                result_path = output_dir / f"{config.experiment_id}.json"
                results_dumped = clean_results(result.model_dump())
                result_path.write_text(json.dumps(results_dumped, indent=2, default=str))
                print(f"  ✓ Saved results to: {result_path.name}")
        else:
            print(f"  ✗ Experiment failed: {summary.error}")
            failed_experiments.append((config.experiment_id, summary.error))

        summary_list.append(summary)
        persist_summary_files()

    # Group configs so we can parallelize per model/instruction level
    grouped_configs = {}
    for config in configs_to_run:
        key = (config.model_name, config.model_engine, config.instructions_level)
        grouped_configs.setdefault(key, []).append(config)

    for group_index, (group_key, group_configs) in enumerate(
        grouped_configs.items(), start=1
    ):
        model_name, model_engine, instruction_level = group_key
        print(
            f"\n--- Group {group_index}/{len(grouped_configs)}: {model_name} ({model_engine}), instructions L{instruction_level} ---"
        )
        print(
            f"  Dialects queued: {', '.join(set(cfg.gen_type for cfg in group_configs))}"
        )

        worker_count = None
        if max_workers not in (None, 0):
            worker_count = max_workers

        run_in_parallel = len(group_configs) > 1 and (
            worker_count is None or worker_count > 1
        )

        if run_in_parallel:
            # Some sandboxed environments (notably macOS + MDS sandbox) block
            # `os.sysconf("SC_SEM_NSEMS_MAX")`, which the stdlib uses when
            # constructing ProcessPoolExecutor. Surface a clear, actionable
            # error instead of hanging on process startup.
            print(
                f"  Executing in parallel with {worker_count or 'auto'} worker(s) across dialects."
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_config = {
                    executor.submit(
                        run_single_experiment, config, cache, load_data
                    ): config
                    for config in group_configs
                }
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        # Timeout per experiment: 10 minutes (600 seconds)
                        # This covers all examples in the experiment
                        summary, result = future.result(timeout=600)
                    except TimeoutError:
                        error_msg = "Experiment timed out after 600 seconds"
                        summary = build_failure_summary(config, error_msg)
                        result = None
                    except Exception as exc:
                        summary = build_failure_summary(config, str(exc))
                        result = None
                    handle_completion(config, summary, result)
        else:
            for config in group_configs:
                try:
                    summary, result = run_single_experiment(config, cache, load_data)
                except Exception as exc:
                    summary = build_failure_summary(config, str(exc))
                    result = None
                handle_completion(config, summary, result)

    print(f"\n{'=' * 80}")
    print("Experiments Complete")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(configs)}")
    if skipped_configs:
        print(f"  - Skipped (loaded): {len(skipped_configs)}")
        print(f"  - Newly run: {len(configs_to_run)}")
    print(f"Successful: {len(summary_list) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print("\nFailed experiments:")
        for exp_id, error in failed_experiments:
            print(f"  - {exp_id}: {error}")

    if save_results:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Individual results: {output_dir}/*.json")
        print(f"  - Summary CSV: {summary_path}")
        print(f"  - Summary JSON: {summary_json_path}")
        print(f"  - Metadata: {metadata_path}")
    else:
        print("\nResults not saved (--no-save flag was set)")


def main():
    """Main entry point with argument parsing."""
    import argparse

    from polysql.evaluation.core.model import SUPPORTED_DIALECTS

    # All supported SQL dialects (no Ibis/Substrait support)
    ALL_SUPPORTED_DIALECTS = SUPPORTED_DIALECTS

    parser = argparse.ArgumentParser(
        description="Run NL2SQL experiments across multiple models and dialects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help='Models to test as "model_name:engine" pairs (e.g., "llama-4-maverick:rits").',
    )
    parser.add_argument(
        "--dialects",
        type=str,
        nargs="+",
        required=True,
        help=f"SQL dialects to test. Supported: {', '.join(ALL_SUPPORTED_DIALECTS)}",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        required=True,
        help="Number of examples to evaluate per experiment",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names (e.g., bird_dev, archer_dev_s_and_c, spider_dev, beaver_dw)",
    )
    parser.add_argument(
        "--instructions-levels",
        type=int,
        nargs="+",
        required=True,
        help="Instruction levels: two-digit format XY (X=COT 1-2, Y=dialect 1-3). "
             "Valid: 11, 12, 13, 21, 22, 23.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for memory safety).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of experiment results.",
    )
    parser.add_argument(
        "--skip-data-load",
        action="store_true",
        help="Skip loading data into backends (assumes data already exists).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files (useful for testing).",
    )
    parser.add_argument(
        "--existing-results-dir",
        type=Path,
        default=None,
        help="Directory with existing results to skip re-running. Result files will be copied to output directory.",
    )

    args = parser.parse_args()

    # Fail fast: DataFusion is in-memory and always needs data loading
    if args.skip_data_load and "datafusion" in args.dialects:
        raise ValueError(
            "--skip-data-load cannot be used with the datafusion backend because it "
            "stores no on-disk state; disable the flag or remove datafusion from "
            "--dialects."
        )

    # Validate and parse models
    models: List[Tuple[str, str]] = []
    for model_spec in args.models:
        assert ":" in model_spec, (
            f"Invalid model format '{model_spec}'. Expected 'model_name:engine'"
        )
        model_name, engine = model_spec.split(":", 1)
        models.append((model_name, engine))

    # Validate dialects
    for dialect in args.dialects:
        # Handle sqlite-{dialect} pattern (e.g., sqlite-postgres, sqlite-duckdb)
        if dialect.startswith("sqlite-"):
            target_dialect = dialect.split("-", 1)[1]
            assert target_dialect in SUPPORTED_DIALECTS, (
                f"Unsupported target dialect '{target_dialect}' in '{dialect}'. "
                f"Supported targets: {SUPPORTED_DIALECTS}"
            )
        else:
            assert dialect in ALL_SUPPORTED_DIALECTS, (
                f"Unsupported dialect '{dialect}'. Supported: {ALL_SUPPORTED_DIALECTS}"
            )

    # Validate instruction levels
    from polysql.evaluation.prompts.sql import parse_instruction_level

    instruction_levels: List[int] = []
    for level in args.instructions_levels:
        try:
            parse_instruction_level(level)
            instruction_levels.append(level)
        except ValueError as e:
            parser.error(str(e))

    # Create experiment configs
    configs = create_experiment_configs(
        models=models,
        dialects=args.dialects,
        dataset_names=args.datasets,
        instruction_levels=instruction_levels,
        n_examples=args.n_examples,
    )

    print("Configuration:")
    print(f"  Models: {len(models)}")
    print(f"  Dialects: {len(args.dialects)}")
    print(f"  Datasets: {len(args.datasets)}")
    print(f"  Instruction levels: {instruction_levels}")
    print(f"  Examples per experiment: {args.n_examples}")
    print(f"  Total experiments: {len(configs)}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    from names_generator import generate_name

    output_dir = (
        Path("results")
        / f"{timestamp}_{generate_name(seed=int(datetime.now().timestamp()))}"
    )

    # Determine cache directory
    cache_dir = CACHE_DIR if not args.no_cache else None

    # Run experiments
    run_experiments(
        configs,
        output_dir,
        max_workers=args.workers,
        cache_dir=cache_dir,
        load_data=not args.skip_data_load,
        save_results=not args.no_save,
        existing_results_dir=args.existing_results_dir,
    )

    if not args.no_save:
        load_experiment_data(output_dir).to_csv(output_dir / "all_results.csv")


if __name__ == "__main__":
    main()
