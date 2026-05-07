"""Backend connector abstractions for Ibis database connections.

This package provides modular, testable connector classes for managing
different backend database connections.

Architecture:
- BackendConnector: Protocol defining the interface all connectors must implement
- BaseBackendConnector: Abstract base class with shared functionality
- Native connectors: SQLiteConnector, MySQLConnector, PostgresConnector
- Cross-dialect connectors: SQLiteToDuckDBConnector, MySQLToPostgresConnector, etc.
- BackendConnectorFactory: Factory for creating appropriate connectors
"""

# Base classes and utilities
from polysql.evaluation.backends.connectors.base import (
    BackendConnector,
    BaseBackendConnector,
    create_sqlite_connection,
    get_sqlite_type_map,
    get_mysql_credentials,
    get_postgres_credentials,
    get_snowflake_credentials,
    get_bigquery_credentials,
)

# Factory
from polysql.evaluation.backends.connectors.factory import BackendConnectorFactory

# Native connectors
from polysql.evaluation.backends.connectors.native import (
    SQLiteConnector,
    MySQLConnector,
    MySQLDumpConnector,
    PostgresConnector,
)

# Cross-dialect connectors
from polysql.evaluation.backends.connectors.cross_dialect import (
    # SQLite source
    SQLiteToDuckDBConnector,
    SQLiteToDataFusionConnector,
    SQLiteToMySQLConnector,
    SQLiteToPostgresConnector,
    SQLiteToBigQueryConnector,
    SQLiteToSnowflakeConnector,
    # MySQL source
    MySQLToPostgresConnector,
    MySQLToSQLiteConnector,
    MySQLToDuckDBConnector,
    MySQLToSnowflakeConnector,
    MySQLToBigQueryConnector,
    # PostgreSQL source
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
