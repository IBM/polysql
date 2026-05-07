"""Parse .insights.md files and extract structured data for analysis."""

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_summary_section(text: str) -> dict:
    """Extract summary statistics from the Summary section (handles multiple formats)."""
    summary = {}

    for pattern in [r"\*\*experiment_id\*\*", r"`experiment_id`"]:
        match = re.search(rf"{pattern}.*?`([^`]+)`", text)
        if match:
            summary["experiment_id"] = match.group(1)
            break

    for field in ["total", "executed", "correct"]:
        for pattern in [rf"\*\*{field}\*\*", rf"`{field}`"]:
            match = re.search(rf"{pattern}.*?\|\s*(\d+)", text)
            if match:
                summary[field] = int(match.group(1))
                break

    for pattern in [r"\*\*accuracy\*\*", r"`accuracy`"]:
        match = re.search(rf"{pattern}.*?\|\s*([\d.]+)", text)
        if match:
            summary["accuracy"] = float(match.group(1))
            break

    for field in ["model_name", "model_engine", "gen_type"]:
        for pattern in [rf"\*\*{field}\*\*", rf"`{field}`"]:
            match = re.search(rf"{pattern}.*?`([^`]+)`", text)
            if match:
                summary[field] = match.group(1)
                break

    for pattern in [r"\*\*instructions_level\*\*", r"`instructions_level`"]:
        match = re.search(rf"{pattern}.*?\|\s*(\d+)", text)
        if match:
            summary["instructions_level"] = int(match.group(1))
            break

    return summary


def clean_pattern_name(pattern: str) -> str:
    """Remove formatting characters from pattern names."""
    pattern = re.sub(r"\*\*", "", pattern)
    pattern = re.sub(r"`", "", pattern)
    return pattern.strip()


def parse_pattern_totals_section(text: str) -> list[dict]:
    """Extract pattern totals from the Pattern Totals section."""
    patterns = []

    pattern_totals_match = re.search(
        r"## \d+\.\s+Pattern Totals.*?\n\|.*?\|\s*Count\s*\|(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if not pattern_totals_match:
        return patterns

    table_text = pattern_totals_match.group(1)
    lines = table_text.strip().split("\n")

    for line in lines:
        if "|" not in line or line.strip().startswith("|---"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        pattern_name = clean_pattern_name(parts[1])
        count_str = clean_pattern_name(parts[2])  # Also clean formatting from count

        if not pattern_name or pattern_name == "Pattern":
            continue

        try:
            count = int(count_str)
        except ValueError:
            continue

        question_ids = []
        if len(parts) > 3:
            qid_str = parts[3].strip()
            if qid_str and qid_str != "–" and qid_str != "-":
                qid_parts = re.findall(r"\d+", qid_str)
                question_ids = [int(qid) for qid in qid_parts]

        patterns.append(
            {
                "pattern": pattern_name,
                "count": count,
                "question_ids": question_ids,
            }
        )

    return patterns


def parse_failure_patterns_section(text: str) -> list[dict]:
    """Extract failure pattern details from the Failure Patterns section."""
    patterns = []

    failure_section_match = re.search(
        r"## \d+\.\s+Failure Patterns(.*?)(?=\n## \d+\.|\Z)", text, re.DOTALL
    )

    if not failure_section_match:
        return patterns

    failure_text = failure_section_match.group(1)

    pattern_blocks = re.findall(
        r"### \d+\.\d+\.\s+\*\*([^*]+)\*\*\s+\*\(([^)]+)\)",
        failure_text,
    )

    for pattern_name, error_type in pattern_blocks:
        pattern_name = clean_pattern_name(pattern_name)
        error_type = error_type.strip().lower()

        patterns.append(
            {
                "pattern": pattern_name,
                "error_type": error_type,
            }
        )

    return patterns


def parse_insights_file(insights_path: Path) -> Optional[dict]:
    """Parse an insights.md file and return structured data."""
    if not insights_path.exists():
        return None

    try:
        text = insights_path.read_text()
    except Exception:
        return None

    summary = parse_summary_section(text)
    pattern_totals = parse_pattern_totals_section(text)
    failure_patterns = parse_failure_patterns_section(text)

    error_type_map = {p["pattern"]: p["error_type"] for p in failure_patterns}

    for pattern in pattern_totals:
        pattern["error_type"] = error_type_map.get(pattern["pattern"], "unknown")

    return {
        "summary": summary,
        "patterns": pattern_totals,
        "file_path": str(insights_path),
    }


def load_all_insights(results_dir: Path) -> pd.DataFrame:
    """Load and parse all insights files in a results directory."""
    all_data = []

    for insights_file in results_dir.glob("**/*.insights.md"):
        parsed = parse_insights_file(insights_file)
        if parsed is None:
            continue

        summary = parsed["summary"]

        for pattern in parsed["patterns"]:
            row = {
                "file_path": parsed["file_path"],
                "experiment_id": summary.get("experiment_id"),
                "model_name": summary.get("model_name"),
                "gen_type": summary.get("gen_type"),
                "instructions_level": summary.get("instructions_level"),
                "total": summary.get("total"),
                "executed": summary.get("executed"),
                "correct": summary.get("correct"),
                "accuracy": summary.get("accuracy"),
                "pattern": pattern["pattern"],
                "pattern_count": pattern["count"],
                "question_ids": pattern["question_ids"],
                "error_type": pattern["error_type"],
            }
            all_data.append(row)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    df = df.dropna(subset=["gen_type"])

    return df


__all__ = [
    "parse_insights_file",
    "load_all_insights",
]
