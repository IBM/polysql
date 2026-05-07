"""Cross-dialect connectors for data conversion between backends."""

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
# PySpark is optional - import only if needed
# from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_pyspark import (
#     SQLiteToPySparkConnector,
# )
from polysql.evaluation.backends.connectors.cross_dialect.sqlite_to_clickhouse import (
    SQLiteToClickHouseConnector,
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

__all__ = [
    # SQLite source
    "SQLiteToDuckDBConnector",
    "SQLiteToDataFusionConnector",
    "SQLiteToMySQLConnector",
    "SQLiteToPostgresConnector",
    "SQLiteToBigQueryConnector",
    "SQLiteToSnowflakeConnector",
    "SQLiteToClickHouseConnector",
    # MySQL source
    "MySQLToPostgresConnector",
    "MySQLToSQLiteConnector",
    "MySQLToDuckDBConnector",
    "MySQLToSnowflakeConnector",
    "MySQLToBigQueryConnector",
    "MySQLToDataFusionConnector",
    "MySQLToClickHouseConnector",
    # PostgreSQL source
    "PostgresToMySQLConnector",
    "PostgresToSQLiteConnector",
    "PostgresToDuckDBConnector",
    "PostgresToSnowflakeConnector",
    "PostgresToBigQueryConnector",
    "PostgresToDataFusionConnector",
    "PostgresToClickHouseConnector",
]
