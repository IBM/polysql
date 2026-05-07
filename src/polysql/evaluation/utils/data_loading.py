"""
Shared evaluation utilities for NL2DSL model testing and benchmarking.

This module contains common functions used by both the test suite and
evaluation scripts to ensure consistency and reduce code duplication.
"""

from typing import Dict, List, Optional

from polysql.evaluation.config.datasets import DatasetConfig
from polysql.utils.data_loaders import load_nl2dsl_dataset


def load_standard_dataset(
    dataset_path: str = "tests/assets/dataset_train_100_0_none.json",
    sample_size: Optional[int] = None,
    random_state: int = 0,
    dataset_config: DatasetConfig = None,
) -> List[Dict]:
    """
    Load the standard NL2DSL dataset with consistent parameters.

    Args:
        dataset_path: Path to dataset file
        sample_size: Number of samples to load (None for all)
        random_state: Random seed for sampling
        dataset_config: Dataset configuration (required)

    Returns:
        List of dataset samples enriched with dataset metadata
    """
    assert (
        dataset_config is not None
    ), "dataset_config is required - no default config available"

    instances = load_nl2dsl_dataset(
        dataset_path=dataset_config.data_path,
        sample_size=sample_size,
        random_state=random_state,
    )

    # Enrich instances with dataset metadata for downstream use
    for instance in instances:
        instance["_dataset_config"] = {
            "name": dataset_config.name,
            "db_id": instance.get(dataset_config.db_id_field, ""),
            "gold_query_dialect": dataset_config.gold_query_dialect,
            "gold_query_type": dataset_config.gold_query_type,
            "source_db_type": dataset_config.source_db_type,
        }

    return instances
