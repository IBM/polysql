"""Dataset configuration system for managing multiple datasets.

This module provides a configuration-driven approach to handling
datasets with:
- Different database source types (SQLite, Postgres, MySQL, etc.)
- Different gold query dialects
- Different field naming conventions across datasets
"""

from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    name: str = Field(..., description="Unique dataset identifier")
    data_path: str = Field(..., description="Path to dataset JSON file")
    source_db_type: str = Field(
        default="sqlite",
        description="Type of source database (sqlite, postgres, mysql)",
    )
    gold_query_dialect: str = Field(
        default="sqlite",
        description="SQL dialect of gold queries",
    )
    gold_query_type: Literal["sql", "substrait"] = Field(
        default="sql", description="Type of gold query"
    )

    # Field name mappings to handle different dataset formats
    db_path_field: str = Field(
        default="db_path",
        description="Field name containing database path",
    )
    gold_query_field: str = Field(
        default="sql_query",
        description="Field name containing gold SQL query",
    )
    nl_query_field: str = Field(
        default="nl_query",
        description="Field name containing natural language query",
    )
    question_id_field: str = Field(
        default="question_id",
        description="Field name containing question ID",
    )
    db_id_field: str = Field(
        default="db_id",
        description="Field name containing database ID",
    )

    # Pre-computed gold results support
    use_precomputed_gold: bool = Field(
        default=False,
        description="Skip gold query execution and use pre-computed results from CSV files",
    )
    gold_results_path: Optional[str] = Field(
        default=None,
        description="Path to directory or file with pre-computed gold results (optional)",
    )


class DatasetRegistry:
    """Registry for managing dataset configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize registry from config file.

        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        if config_path is None:
            # Default to configs/datasets.yaml in project root
            # From src/nl2dsl/evaluation/config/datasets.py -> project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = project_root / "configs" / "datasets.yaml"

        self.config_path = config_path
        self.datasets: Dict[str, DatasetConfig] = {}

        if config_path.exists():
            self._load_configs()
        else:
            raise FileNotFoundError(
                f"Dataset config file not found: {config_path}. "
                "All dataset configurations must be defined in configs/datasets.yaml"
            )

    def _load_configs(self) -> None:
        """Load dataset configurations from YAML file."""
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "datasets" not in data:
            return

        for dataset_name, dataset_data in data["datasets"].items():
            self.datasets[dataset_name] = DatasetConfig(**dataset_data)

    def get(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name."""
        return self.datasets.get(name)

    def register(self, config: DatasetConfig) -> None:
        """Register a new dataset configuration."""
        self.datasets[config.name] = config

    def list_datasets(self) -> list[str]:
        """List all registered dataset names."""
        return list(self.datasets.keys())


# Global registry instance
_global_registry: Optional[DatasetRegistry] = None


def get_registry() -> DatasetRegistry:
    """Get or create global dataset registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DatasetRegistry()
    return _global_registry


def get_dataset_config(name_or_path: str) -> DatasetConfig:
    """
    Get dataset configuration by name or create default from path.

    Args:
        name_or_path: Dataset name (e.g., "bird_dev") or path to dataset file

    Returns:
        DatasetConfig object

    Raises:
        ValueError: If dataset name not found in registry
    """
    registry = get_registry()

    # Try to get from registry first
    config = registry.get(name_or_path)
    if config is not None:
        return config

    # Dataset name not found and not a path
    available = registry.list_datasets()
    raise ValueError(
        f"Dataset '{name_or_path}' not found in registry. "
        f"Available datasets: {available}. "
        f"Or provide a path to a dataset JSON file."
    )
