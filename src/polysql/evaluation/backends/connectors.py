"""Backend connector abstractions for Ibis database connections.

DEPRECATED: This module is a re-export shim for backward compatibility.
Import from polysql.evaluation.backends.connectors package instead.

The connectors have been refactored into separate modules:
- nl2dsl.evaluation.backends.connectors.base: Base classes and utilities
- nl2dsl.evaluation.backends.connectors.native: Native backend connectors
- nl2dsl.evaluation.backends.connectors.cross_dialect: Cross-dialect connectors
- nl2dsl.evaluation.backends.connectors.factory: BackendConnectorFactory
"""

# Re-export everything from the new package structure
from polysql.evaluation.backends.connectors import (
    # Base classes and utilities
    BackendConnector,
    BaseBackendConnector,
    create_sqlite_connection,
    get_sqlite_type_map,
    get_mysql_credentials,
    get_postgres_credentials,
    get_snowflake_credentials,
    get_bigquery_credentials,
    # Factory
    BackendConnectorFactory,
    # Native connectors
    SQLiteConnector,
    MySQLConnector,
    MySQLDumpConnector,
    PostgresConnector,
    # SQLite cross-dialect
    SQLiteToDuckDBConnector,
    SQLiteToDataFusionConnector,
    SQLiteToMySQLConnector,
    SQLiteToPostgresConnector,
    SQLiteToBigQueryConnector,
    SQLiteToSnowflakeConnector,
    SQLiteToPySparkConnector,
    # MySQL cross-dialect
    MySQLToPostgresConnector,
    MySQLToSQLiteConnector,
    MySQLToDuckDBConnector,
    MySQLToSnowflakeConnector,
    MySQLToBigQueryConnector,
    # PostgreSQL cross-dialect
    PostgresToMySQLConnector,
    PostgresToSQLiteConnector,
    PostgresToDuckDBConnector,
    PostgresToSnowflakeConnector,
    PostgresToBigQueryConnector,
)

__all__ = [
    # Base
    "BackendConnector",
    "BaseBackendConnector",
    "create_sqlite_connection",
    "get_sqlite_type_map",
    "get_mysql_credentials",
    "get_postgres_credentials",
    "get_snowflake_credentials",
    "get_bigquery_credentials",
    # Factory
    "BackendConnectorFactory",
    # Native connectors
    "SQLiteConnector",
    "MySQLConnector",
    "MySQLDumpConnector",
    "PostgresConnector",
    # SQLite cross-dialect
    "SQLiteToDuckDBConnector",
    "SQLiteToDataFusionConnector",
    "SQLiteToMySQLConnector",
    "SQLiteToPostgresConnector",
    "SQLiteToBigQueryConnector",
    "SQLiteToSnowflakeConnector",
    "SQLiteToPySparkConnector",
    # MySQL cross-dialect
    "MySQLToPostgresConnector",
    "MySQLToSQLiteConnector",
    "MySQLToDuckDBConnector",
    "MySQLToSnowflakeConnector",
    "MySQLToBigQueryConnector",
    # PostgreSQL cross-dialect
    "PostgresToMySQLConnector",
    "PostgresToSQLiteConnector",
    "PostgresToDuckDBConnector",
    "PostgresToSnowflakeConnector",
    "PostgresToBigQueryConnector",
]
