"""Error analysis for CSV result files using judge+aggregator pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from polysql.evaluation.analysis.insights import EvaluationInsightsGenerator


def load_csv_results(csv_path: Path) -> pd.DataFrame:
    """Load results CSV into dataframe."""
    return pd.read_csv(csv_path)


def prepare_predictions_from_df(
    df: pd.DataFrame, sample_size: Optional[int] = None
) -> list[dict]:
    """Convert dataframe rows to prediction dictionaries compatible with insights generator."""
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    predictions = []
    for idx, row in df.iterrows():
        prediction = {
            "question_id": int(idx),
            "db_id": "unknown",
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "predicted_sql": row["predicted_sql"],
            "results_equal": row["results_equal"]
            if pd.notna(row["results_equal"])
            else None,
            "gold_error": row["gold_error"] if pd.notna(row["gold_error"]) else None,
            "pred_error": row["pred_error"] if pd.notna(row["pred_error"]) else None,
            "full_prompt": row["full_prompt"] if pd.notna(row["full_prompt"]) else "",
        }
        predictions.append(prediction)

    return predictions


def analyze_errors_from_csv(
    csv_path: Path,
    output_path: Optional[Path] = None,
    *,
    sample_size: Optional[int] = None,
    filter_by_model: Optional[str] = None,
    filter_by_dialect: Optional[str] = None,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
    judge_max_tokens: int = 2048,
    aggregator_max_tokens: int = 4096,
    temperature: float = 0.0,
    instructions: Optional[str] = None,
) -> Path:
    """
    Generate error analysis from CSV results file.

    Args:
        csv_path: Path to CSV results file
        output_path: Where to write the markdown report (defaults to csv_path.with_suffix('.error_analysis.md'))
        sample_size: If provided, randomly sample this many rows
        filter_by_model: If provided, only analyze rows with this model_name
        filter_by_dialect: If provided, only analyze rows with this gen_type (dialect)
        model_name: LLM model to use for analysis
        provider: LLM provider
        judge_max_tokens: Max tokens for judge model
        aggregator_max_tokens: Max tokens for aggregator model
        temperature: Sampling temperature
        instructions: Custom aggregator template

    Returns:
        Path to the written insights file
    """
    df = load_csv_results(csv_path)

    if filter_by_model is not None:
        df = df[df["model_name"] == filter_by_model]

    if filter_by_dialect is not None:
        df = df[df["gen_type"] == filter_by_dialect]

    if len(df) == 0:
        raise ValueError("No rows match the specified filters")

    predictions = prepare_predictions_from_df(df, sample_size=sample_size)

    metadata = {
        "exp_config": {
            "csv_path": str(csv_path),
            "sample_size": sample_size,
            "filter_by_model": filter_by_model,
            "filter_by_dialect": filter_by_dialect,
        },
        "total": len(df),
        "executed": int(df["both_executed"].sum())
        if "both_executed" in df.columns
        else 0,
        "correct": int(df["results_equal"].sum())
        if "results_equal" in df.columns
        else 0,
        "accuracy": float(df["results_equal"].mean())
        if "results_equal" in df.columns
        else 0.0,
    }

    generator = EvaluationInsightsGenerator(
        model_name=model_name,
        provider=provider,
        judge_max_tokens=judge_max_tokens,
        aggregator_max_tokens=aggregator_max_tokens,
        temperature=temperature,
    )

    result_dict = {"exp_config": metadata["exp_config"], "predictions": predictions}
    metadata_for_aggregator = {
        "exp_config": metadata["exp_config"],
        "total": metadata["total"],
        "executed": metadata["executed"],
        "correct": metadata["correct"],
        "accuracy": metadata["accuracy"],
    }

    judgments = generator.judge_predictions(predictions)
    insights_text = generator._aggregate_judgments(
        judgments, metadata_for_aggregator, instructions
    )

    destination = (
        output_path
        if output_path is not None
        else csv_path.with_suffix(".error_analysis.md")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(insights_text)
    return destination


def analyze_by_model_and_dialect(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    *,
    sample_size_per_group: Optional[int] = None,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
    judge_max_tokens: int = 2048,
    aggregator_max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict[str, Path]:
    """
    Generate separate error analysis for each (model, dialect) combination.

    Returns:
        Dictionary mapping "{model}_{dialect}" to output file path
    """
    df = load_csv_results(csv_path)

    if "model_name" not in df.columns or "gen_type" not in df.columns:
        raise ValueError("CSV must contain 'model_name' and 'gen_type' columns")

    output_dir = output_dir or csv_path.parent / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for (model, dialect), group_df in df.groupby(["model_name", "gen_type"]):
        key = f"{model}_{dialect}"
        output_path = output_dir / f"{key}_error_analysis.md"

        analysis_path = analyze_errors_from_csv(
            csv_path,
            output_path=output_path,
            sample_size=sample_size_per_group,
            filter_by_model=model,
            filter_by_dialect=dialect,
            model_name=model_name,
            provider=provider,
            judge_max_tokens=judge_max_tokens,
            aggregator_max_tokens=aggregator_max_tokens,
            temperature=temperature,
        )
        results[key] = analysis_path

    return results


__all__ = [
    "load_csv_results",
    "prepare_predictions_from_df",
    "analyze_errors_from_csv",
    "analyze_by_model_and_dialect",
]
