"""Factory for creating appropriate backend connectors."""

from pathlib import Path
from typing import List, Optional

from polysql.evaluation.backends.connectors.base import BaseBackendConnector

# Import native connectors
from polysql.evaluation.backends.connectors.native.sqlite import SQLiteConnector
from polysql.evaluation.backends.connectors.native.mysql import (
    MySQLConnector,
    MySQLDumpConnector,
)
from polysql.evaluation.backends.connectors.native.postgres import PostgresConnector

# Import cross-dialect connectors
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_duckdb import (
    SQLiteToDuckDBConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_datafusion import (
    SQLiteToDataFusionConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_mysql import (
    SQLiteToMySQLConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_postgres import (
    SQLiteToPostgresConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_bigquery import (
    SQLiteToBigQueryConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_snowflake import (
    SQLiteToSnowflakeConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_clickhouse import (
    SQLiteToClickHouseConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_databricks import (
    SQLiteToDatabricksConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_postgres import (
    MySQLToPostgresConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_sqlite import (
    MySQLToSQLiteConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_duckdb import (
    MySQLToDuckDBConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_snowflake import (
    MySQLToSnowflakeConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_bigquery import (
    MySQLToBigQueryConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_datafusion import (
    MySQLToDataFusionConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_mysql import (
    PostgresToMySQLConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_sqlite import (
    PostgresToSQLiteConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_duckdb import (
    PostgresToDuckDBConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_snowflake import (
    PostgresToSnowflakeConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_bigquery import (
    PostgresToBigQueryConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_datafusion import (
    PostgresToDataFusionConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.mysql_to_clickhouse import (
    MySQLToClickHouseConnector,
)
from polysql.evaluation.backends.connectors.cross_dialect.postgres_to_clickhouse import (
    PostgresToClickHouseConnector,
)


class BackendConnectorFactory:
    """Factory for creating appropriate backend connectors."""

    _connectors = {
        # Native connectors (no data conversion)
        "sqlite": SQLiteConnector,
        "mysql": MySQLConnector,
        "postgres": PostgresConnector,
        # SQLite source conversions
        "sqlite_to_duckdb": SQLiteToDuckDBConnector,
        "sqlite_to_datafusion": SQLiteToDataFusionConnector,
        "sqlite_to_mysql": SQLiteToMySQLConnector,
        "sqlite_to_postgres": SQLiteToPostgresConnector,
        "sqlite_to_bigquery": SQLiteToBigQueryConnector,
        "sqlite_to_snowflake": SQLiteToSnowflakeConnector,
        "sqlite_to_clickhouse": SQLiteToClickHouseConnector,
        "sqlite_to_databricks": SQLiteToDatabricksConnector,
        # Cross-dialect conversions
        "mysql_to_postgres": MySQLToPostgresConnector,
        "mysql_to_sqlite": MySQLToSQLiteConnector,
        "mysql_to_duckdb": MySQLToDuckDBConnector,
        "mysql_to_snowflake": MySQLToSnowflakeConnector,
        "mysql_to_bigquery": MySQLToBigQueryConnector,
        "mysql_to_datafusion": MySQLToDataFusionConnector,
        "mysql_to_clickhouse": MySQLToClickHouseConnector,
        "postgres_to_mysql": PostgresToMySQLConnector,
        "postgres_to_sqlite": PostgresToSQLiteConnector,
        "postgres_to_duckdb": PostgresToDuckDBConnector,
        "postgres_to_snowflake": PostgresToSnowflakeConnector,
        "postgres_to_bigquery": PostgresToBigQueryConnector,
        "postgres_to_datafusion": PostgresToDataFusionConnector,
        "postgres_to_clickhouse": PostgresToClickHouseConnector,
        # Special cases
        "mysql_dump": MySQLDumpConnector,
    }

    @classmethod
    def create(
        cls,
        backend: str,
        db_path: Optional[Path] = None,
        source_db_type: Optional[str] = None,
    ) -> BaseBackendConnector:
        """Create a connector for the specified backend."""
        # Special case: MySQL dump files
        if db_path is not None and backend == "mysql" and str(db_path).endswith(".sql"):
            return cls._connectors["mysql_dump"]()

        # Determine source type (default to SQLite for backward compatibility)
        source = source_db_type if source_db_type is not None else "sqlite"

        # Native connector: source matches target
        if source == backend:
            connector_class = cls._connectors.get(backend)
            if connector_class is None:
                raise ValueError(
                    f"Native connector not available for backend: {backend}. "
                    f"Available native backends: sqlite, mysql, postgres"
                )
            return connector_class()

        # Cross-dialect connector: source != target
        connector_key = f"{source}_to_{backend}"
        connector_class = cls._connectors.get(connector_key)

        if connector_class is None:
            raise ValueError(
                f"Cross-dialect connector not available: {source} → {backend}. "
                f"Available conversions: {[k for k in cls._connectors.keys() if '_to_' in k]}"
            )

        return connector_class()

    @classmethod
    def register(cls, backend: str, connector_class) -> None:
        """Register a new backend connector."""
        cls._connectors[backend] = connector_class

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all supported native backends."""
        native_backends = ["sqlite", "mysql", "postgres"]
        conversion_targets = {
            key.split("_to_")[1] for key in cls._connectors.keys() if "_to_" in key
        }
        return sorted(set(native_backends) | conversion_targets)
