"""Unified cache management for converted databases.

This module provides a consistent approach to storing and retrieving converted
databases across all backends (SQLite, DuckDB, MySQL, PostgreSQL, etc.).

The cache structure mirrors the connector architecture:
- Native evaluation (source == target): No caching needed, uses original database
- Cross-dialect (source != target): Caches converted databases in .nl2dsl_cache/

Cache directory format: .nl2dsl_cache/{source_type}_to_{target_type}/{namespace}/{filename}
"""

import hashlib
from pathlib import Path
from typing import Optional

from polysql.evaluation.utils.cache_paths import cache_subdirs, migrate_legacy_caches


class DatabaseCache:
    """Manages caching of converted databases with consistent directory structure."""

    def __init__(self, base_cache_dir: Optional[Path] = None):
        """Initialize database cache.

        Args:
            base_cache_dir: Root directory for all cached databases
        """
        migrate_legacy_caches()
        self.base_cache_dir = (
            base_cache_dir
            if base_cache_dir is not None
            else cache_subdirs()["converted_dbs"]
        )
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self,
        source_db_path: Path,
        source_type: str,
        target_type: str,
        namespace: Optional[str] = None,
    ) -> Path:
        """Get the cache path for a converted database.

        Directory structure: .nl2dsl_cache/{source_type}_to_{target_type}/{namespace}/{filename}

        Args:
            source_db_path: Path or identifier of source database
            source_type: Source database type (sqlite, mysql, postgres, etc.)
            target_type: Target database type (sqlite, duckdb, mysql, etc.)
            namespace: Optional namespace for organizing related conversions (e.g., dataset name)

        Returns:
            Path where the converted database should be stored

        Examples:
            >>> cache = DatabaseCache()
            >>> # MySQL → SQLite conversion for MINIDEV dataset
            >>> path = cache.get_cache_path(
            ...     Path("minidev_mysql_card_games"),
            ...     source_type="mysql",
            ...     target_type="sqlite",
            ...     namespace="bird_mini_dev_mysql"
            ... )
            >>> # Returns: .nl2dsl_cache/mysql_to_sqlite/bird_mini_dev_mysql/mysql_import_abc123.sqlite

            >>> # SQLite → DuckDB conversion
            >>> path = cache.get_cache_path(
            ...     Path("data/BIRD/dev/card_games/card_games.sqlite"),
            ...     source_type="sqlite",
            ...     target_type="duckdb"
            ... )
            >>> # Returns: .nl2dsl_cache/sqlite_to_duckdb/sqlite_import_def456.duckdb

            >>> # PostgreSQL → MySQL conversion
            >>> path = cache.get_cache_path(
            ...     Path("minidev_postgres_financial"),
            ...     source_type="postgres",
            ...     target_type="mysql",
            ...     namespace="bird_mini_dev_postgres"
            ... )
            >>> # Returns: .nl2dsl_cache/postgres_to_mysql/bird_mini_dev_postgres/postgres_import_xyz789 (MySQL database name)
        """
        # Generate stable hash from source path
        hash_input = str(source_db_path)
        db_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Determine file extension (for file-based databases)
        extension = self._get_extension(target_type)

        # Construct filename/database name
        if extension:
            # File-based database (SQLite, DuckDB)
            filename = f"{source_type}_import_{db_hash}.{extension}"
        else:
            # Server-based database (MySQL, PostgreSQL) - return database name
            filename = f"{source_type}_import_{db_hash}"

        # Build directory structure: {source}_to_{target}/{namespace}/
        conversion_type = f"{source_type}_to_{target_type}"

        if namespace:
            cache_dir = self.base_cache_dir / conversion_type / namespace
        else:
            cache_dir = self.base_cache_dir / conversion_type

        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / filename

    def _get_extension(self, target_type: str) -> Optional[str]:
        """Get file extension for target database type.

        Returns None for server-based databases (MySQL, PostgreSQL).
        """
        file_based_extensions = {
            "sqlite": "sqlite",
            "duckdb": "duckdb",
        }
        return file_based_extensions.get(target_type)

    def cache_exists(self, cache_path: Path, target_type: str) -> bool:
        """Check if cached database exists.

        For file-based databases (SQLite, DuckDB), checks if file exists.
        For server-based databases (MySQL, PostgreSQL), checks if database exists on server.
        """
        if self._get_extension(target_type):
            # File-based: check file existence
            return cache_path.exists()
        else:
            # Server-based: caller should check via connection
            # This is handled by connectors themselves
            return False

    def get_all_cached_databases(
        self, source_type: Optional[str] = None, target_type: Optional[str] = None
    ) -> list[Path]:
        """Get all cached databases, optionally filtered by conversion type.

        Args:
            source_type: Filter by source database type
            target_type: Filter by target database type

        Returns:
            List of paths to cached database files
        """
        if source_type and target_type:
            conversion_type = f"{source_type}_to_{target_type}"
            search_dir = self.base_cache_dir / conversion_type
            if not search_dir.exists():
                return []
            extension = self._get_extension(target_type)
            if extension:
                pattern = f"*.{extension}"
                return list(search_dir.rglob(pattern))
            return []
        else:
            # Get all database files
            patterns = ["*.sqlite", "*.duckdb"]
            files = []
            for pattern in patterns:
                files.extend(self.base_cache_dir.rglob(pattern))
            return files

    def clear_cache(
        self, source_type: Optional[str] = None, target_type: Optional[str] = None
    ):
        """Clear cached databases.

        Args:
            source_type: Clear only caches from specific source type
            target_type: Clear only caches to specific target type
            If both specified, clears only that conversion type
            If neither specified, clears all caches
        """
        if source_type and target_type:
            conversion_type = f"{source_type}_to_{target_type}"
            cache_dir = self.base_cache_dir / conversion_type
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
        elif source_type or target_type:
            # Clear all conversions involving the specified type
            import shutil

            for conversion_dir in self.base_cache_dir.iterdir():
                if not conversion_dir.is_dir():
                    continue
                parts = conversion_dir.name.split("_to_")
                if len(parts) == 2:
                    src, tgt = parts
                    if (source_type and src == source_type) or (
                        target_type and tgt == target_type
                    ):
                        shutil.rmtree(conversion_dir)
        else:
            # Clear all caches
            if self.base_cache_dir.exists():
                import shutil

                shutil.rmtree(self.base_cache_dir)
                self.base_cache_dir.mkdir(parents=True, exist_ok=True)


# Global cache instance
_global_cache = DatabaseCache()


def get_cache_path(
    source_db_path: Path,
    source_type: str,
    target_type: str,
    namespace: Optional[str] = None,
) -> Path:
    """Get cache path using global cache instance.

    Args:
        source_db_path: Path or identifier of source database
        source_type: Source database type (sqlite, mysql, postgres, etc.)
        target_type: Target database type (sqlite, duckdb, mysql, etc.)
        namespace: Optional namespace for organizing related conversions

    Returns:
        Path where the converted database should be stored
    """
    return _global_cache.get_cache_path(
        source_db_path, source_type, target_type, namespace
    )


__all__ = ["DatabaseCache", "get_cache_path"]
