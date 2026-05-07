"""Configuration and type definitions for evaluation."""

from polysql.evaluation.config.datasets import (
    DatasetConfig,
    DatasetRegistry,
    get_dataset_config,
    get_registry,
)
from polysql.evaluation.config.types import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentSummary,
    PredictionResult,
)

__all__ = [
    "DatasetConfig",
    "DatasetRegistry",
    "get_dataset_config",
    "get_registry",
    "EvaluationResult",
    "ExperimentConfig",
    "ExperimentSummary",
    "PredictionResult",
]
