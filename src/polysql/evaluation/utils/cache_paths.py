"""Central cache path helpers for NL2DSL."""

import os
import shutil
from pathlib import Path
from typing import Dict


def get_cache_root() -> Path:
    """Return root cache directory."""
    root = Path(".cache") / "nl2dsl"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_subdirs() -> Dict[str, Path]:
    """Return all standard cache subdirectories."""
    root = get_cache_root()
    paths = {
        "converted_dbs": root / "converted_dbs",
        "schemas": root / "schemas",
        "executions": root / "executions",
        "experiments": root / "experiments",
        "inference": root / "inference",
        "tooling": root / "tooling",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def migrate_legacy_caches() -> None:
    """Move old cache locations into the unified .cache/nl2dsl layout once."""
    paths = cache_subdirs()
    migrations = {
        Path(".nl2dsl_cache"): paths["converted_dbs"],
        Path("evaluation_cache"): paths["experiments"],
        Path("inference_engine_cache"): paths["inference"],
        Path("src/nl2dsl/evaluation/backends/.cache/schemas"): paths["schemas"],
        Path("src/nl2dsl/evaluation/backends/.cache/executions"): paths["executions"],
        Path("src/nl2dsl/evaluation/core/.cache/experiments"): paths["experiments"],
    }

    for legacy_path, target_path in migrations.items():
        if not legacy_path.exists():
            continue

        # Avoid double moves if already consolidated
        if any(target_path.iterdir()):
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Move contents rather than nesting the folder name
            for child in legacy_path.iterdir():
                shutil.move(str(child), target_path)
            legacy_path.rmdir()
        except Exception as exc:
            raise RuntimeError(f"Failed migrating cache from {legacy_path} to {target_path}: {exc}") from exc


def ensure_inference_cache_env() -> None:
    """Point unitxt inference cache to the unified location."""
    paths = cache_subdirs()
    env_key = "UNITXT_INFERENCE_ENGINE_CACHE_PATH"
    if os.getenv(env_key):
        return
    os.environ[env_key] = str(paths["inference"])
