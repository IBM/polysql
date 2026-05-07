"""Base classes and shared utilities for backend connectors.

This module provides the foundation for all backend connectors:
- BackendConnector: Protocol defining the interface all connectors must implement
- BaseBackendConnector: Abstract base class with shared functionality
- Utility functions for credentials and common operations
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Protocol, Union

import ibis
from ibis import BaseBackend


def _set_mysql_session_timezone(cursor) -> None:
    """Ensure MySQL sessions always run in UTC."""
    try:
        cursor.execute("SET @@session.time_zone = 'UTC'")
        return
    except Exception:  # noqa: BLE001
        try:
            cursor.execute("SET @@session.time_zone = '+00:00'")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to set MySQL session time zone to UTC. "
                "Populate MySQL time zone tables or ensure the server "
                "supports '+00:00' offsets."
            ) from exc


def _patch_ibis_mysql_timezone() -> None:
    """Monkey patch Ibis MySQL backend to use our deterministic timezone helper."""
    try:
        from ibis.backends.mysql import Backend as MySQLBackend
    except Exception:
        return

    if getattr(MySQLBackend, "_nl2dsl_timezone_patch", False):
        return

    def _patched_post_connect(self) -> None:  # type: ignore[no-untyped-def]
        with self.con.cursor() as cursor:
            _set_mysql_session_timezone(cursor)

    MySQLBackend._post_connect = _patched_post_connect  # type: ignore[attr-defined]
    MySQLBackend._nl2dsl_timezone_patch = True  # type: ignore[attr-defined]


# Apply the patch when this module is imported
_patch_ibis_mysql_timezone()


def get_sqlite_type_map() -> dict[str, str]:
    """Return type map for SQLite connections to handle non-standard float declarations."""
    return {
        "float(10,2)": "float64",
        "float(3,1)": "float64",
        "float(4,1)": "float64",
    }


def create_sqlite_connection(db_path: Path) -> BaseBackend:
    """Create SQLite connection with proper encoding handling."""
    conn = ibis.sqlite.connect(str(db_path), type_map=get_sqlite_type_map())
    conn.con.text_factory = lambda x: x.decode("utf-8", errors="replace")
    return conn


def get_mysql_credentials() -> dict:
    """Get MySQL credentials from environment variables."""
    password = os.getenv("MYSQL_PASSWORD", "")
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": password if password else "",
    }


def get_postgres_credentials() -> dict:
    """Get PostgreSQL credentials from environment variables."""
    return {
        "host": os.getenv("DBHOST", "localhost"),
        "port": int(os.getenv("DBPORT", "5432")),
        "user": os.getenv("DBUSER", "postgres"),
        "password": os.getenv("DBPASSWORD", "postgres"),
    }


def get_snowflake_credentials() -> dict:
    """Get Snowflake credentials from environment variables."""
    return {
        "user": os.getenv("SFDBUSER"),
        "password": os.getenv("SFDBPASSWORD"),
        "account": os.getenv("SFDBACCOUNT"),
        "warehouse": os.getenv("SFDBWAREHOUSE"),
    }


def get_bigquery_credentials() -> dict:
    """Get BigQuery credentials from environment variables."""
    return {
        "project_id": os.getenv("BIGQUERY_PROJ"),
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def get_clickhouse_credentials() -> dict:
    """Get ClickHouse credentials from environment variables."""
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    return {
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "9000")),  # Native TCP port
        "http_port": int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),  # HTTP port
        "database": os.getenv("CLICKHOUSE_DATABASE", "default"),
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": password if password else "",
    }


def get_databricks_credentials() -> dict:
    """Get Databricks credentials from environment variables."""
    return {
        "server_hostname": os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        "http_path": os.getenv("DATABRICKS_HTTP_PATH"),
        "access_token": os.getenv("DATABRICKS_TOKEN"),
        "catalog": os.getenv("DATABRICKS_CATALOG", "workspace"),
    }


class BackendConnector(Protocol):
    """Protocol defining the interface for backend connectors."""

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Create connection and optionally load data from source."""
        ...

    def list_tables(self, conn: BaseBackend) -> List[str]:
        """List all non-system tables in the connection."""
        ...


class BaseBackendConnector(ABC):
    """Base class for backend connectors with shared functionality."""

    def __init__(self, backend_name: str):
        """Initialize connector with backend name."""
        self.backend_name = backend_name

    @abstractmethod
    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Create connection - must be implemented by subclasses."""
        pass

    def list_tables(self, conn: BaseBackend) -> List[str]:
        """List all non-system tables."""
        all_tables = conn.list_tables()
        return [t for t in all_tables if "_dlt_" not in t.lower()]

    def _validate_source_exists(self, db_path: Union[str, Path]) -> None:
        """Validate that source database file exists."""
        if isinstance(db_path, str):
            return
        if not db_path.suffix:
            return
        if not db_path.exists():
            raise FileNotFoundError(f"Source database not found: {db_path}")

    def _get_source_tables(self, db_path: Path) -> List[str]:
        """Get list of tables from source SQLite database."""
        sqlite_conn = create_sqlite_connection(db_path)
        table_names = sqlite_conn.list_tables()
        sqlite_conn = None  # Close connection
        return table_names
