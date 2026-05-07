"""PostgreSQL native connector."""

from pathlib import Path
from typing import Literal

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_postgres_credentials,
)
from polysql.evaluation.backends.connections import _ensure_schema_cache


class PostgresConnector(BaseBackendConnector):
    """Connector for native PostgreSQL databases.

    Connects directly to existing PostgreSQL databases without data conversion.
    """

    def __init__(self):
        """Initialize PostgreSQL connector."""
        super().__init__("postgres")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to an existing PostgreSQL database."""
        creds = get_postgres_credentials()
        database_name = str(db_path)

        try:
            conn = ibis.postgres.connect(
                host=creds["host"],
                port=creds["port"],
                database=database_name,
                user=creds["user"],
                password=creds["password"],
            )
            _ = conn.list_tables()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL database '{database_name}': {e}"
            )

        _ensure_schema_cache(conn, db_path, "postgres")
        return conn
