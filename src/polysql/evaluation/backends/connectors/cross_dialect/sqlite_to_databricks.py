"""SQLite to Databricks cross-dialect connector."""

import hashlib
from pathlib import Path
from typing import Literal
import os

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _ensure_schema_cache,
    _load_data_with_dlt,
    _normalize_table_names_for_matching,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
    get_databricks_credentials,
)


class SQLiteToDatabricksConnector(BaseBackendConnector):
    """Connector for loading SQLite data into Databricks backend."""

    def __init__(self):
        """Initialize SQLite to Databricks connector."""
        super().__init__("databricks")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to Databricks and optionally load data."""
        creds = get_databricks_credentials()
        server_hostname = creds["server_hostname"]
        http_path = creds["http_path"]
        access_token = creds["access_token"]
        catalog = creds["catalog"]

        if not server_hostname:
            raise ValueError(
                "DATABRICKS_SERVER_HOSTNAME environment variable not set. "
                "Please set it in .env file."
            )
        if not http_path:
            raise ValueError(
                "DATABRICKS_HTTP_PATH environment variable not set. "
                "Please set it in .env file."
            )
        if not access_token:
            raise ValueError(
                "DATABRICKS_TOKEN environment variable not set. "
                "Please set it in .env file."
            )

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        schema_name = f"sqlite_import_{db_hash}"

        should_reload = True
        table_names = []

        if not load_data:
            should_reload = False
        else:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            try:
                test_conn = ibis.connect(
                    f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}&catalog={catalog}&schema={schema_name}"
                )
                # Use raw SQL to check tables instead of list_tables()
                result = test_conn.raw_sql(
                    f"SHOW TABLES IN {catalog}.{schema_name}"
                )
                existing_table_rows = result.fetchall()
                existing_tables = _normalize_table_names_for_matching(
                    [row.tableName for row in existing_table_rows if not row.tableName.startswith("_dlt_")]
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    should_reload = False
                else:
                    print(
                        f"Databricks schema {schema_name} missing tables. Will reload."
                    )
                test_conn = None
            except Exception as e:
                print(
                    f"Databricks schema {schema_name} does not exist or error: {e}. Will create."
                )

        if should_reload:
            # Set environment variables for dlt Databricks destination
            os.environ["DESTINATION__DATABRICKS__CREDENTIALS__SERVER_HOSTNAME"] = (
                server_hostname
            )
            os.environ["DESTINATION__DATABRICKS__CREDENTIALS__HTTP_PATH"] = http_path
            os.environ["DESTINATION__DATABRICKS__CREDENTIALS__ACCESS_TOKEN"] = (
                access_token
            )
            os.environ["DESTINATION__DATABRICKS__CREDENTIALS__CATALOG"] = catalog

            sqlite_conn = create_sqlite_connection(db_path)
            _load_data_with_dlt(
                sqlite_conn=sqlite_conn,
                table_names=table_names,
                backend="databricks",
                dataset_name=schema_name,
                write_disposition=write_disposition,
            )
            sqlite_conn = None

        conn = ibis.connect(
            f"databricks://token:{access_token}@{server_hostname}?http_path={http_path}&catalog={catalog}&schema={schema_name}"
        )
        _ensure_schema_cache(conn, db_path, "databricks")
        return conn
