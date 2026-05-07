"""SQLite to BigQuery cross-dialect connector."""

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
    get_bigquery_credentials,
)


class SQLiteToBigQueryConnector(BaseBackendConnector):
    """Connector for loading SQLite data into BigQuery backend."""

    def __init__(self):
        """Initialize SQLite to BigQuery connector."""
        super().__init__("bigquery")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to BigQuery and optionally load data."""
        creds = get_bigquery_credentials()
        project_id = creds["project_id"]
        credentials_path = creds["credentials_path"]

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        dataset_name = f"sqlite_import_{db_hash}"

        if not project_id:
            raise ValueError(
                "BIGQUERY_PROJ environment variable not set. "
                "Please set it in .env file."
            )
        if not credentials_path or not Path(credentials_path).exists():
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS not set or file not found. "
                "Please set correct path in .env file."
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        should_reload = True
        table_names = []

        if not load_data:
            should_reload = False
        else:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            try:
                test_conn = ibis.bigquery.connect(
                    project_id=project_id, dataset_id=dataset_name
                )
                existing_tables = _normalize_table_names_for_matching(
                    [
                        tab
                        for tab in test_conn.list_tables()
                        if not tab.startswith("_dlt_")
                    ]
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    should_reload = False
                else:
                    print(
                        f"BigQuery dataset {dataset_name} missing tables. Will reload."
                    )
                test_conn = None
            except Exception as e:
                print(
                    f"BigQuery dataset {dataset_name} does not exist or error: {e}. Will create."
                )

        if should_reload:
            sqlite_conn = create_sqlite_connection(db_path)
            _load_data_with_dlt(
                sqlite_conn=sqlite_conn,
                table_names=table_names,
                backend="bigquery",
                dataset_name=dataset_name,
                write_disposition=write_disposition,
            )
            sqlite_conn = None

        conn = ibis.bigquery.connect(project_id=project_id, dataset_id=dataset_name)
        _ensure_schema_cache(conn, db_path, "bigquery")
        return conn
