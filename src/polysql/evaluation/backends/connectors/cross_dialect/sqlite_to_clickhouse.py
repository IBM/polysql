"""SQLite to ClickHouse cross-dialect connector."""

import hashlib
from pathlib import Path
from typing import List, Literal
import os

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _load_data_with_dlt,
    _normalize_table_names_for_matching,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
    get_clickhouse_credentials,
)


class SQLiteToClickHouseConnector(BaseBackendConnector):
    """Connector for loading SQLite data into ClickHouse backend."""

    def __init__(self):
        """Initialize SQLite to ClickHouse connector."""
        super().__init__("clickhouse")

    def _get_prefixed_tables(self, conn: BaseBackend, dataset_name: str) -> List[str]:
        """Get tables with the dlt dataset prefix, returning unprefixed names."""
        prefix = f"{dataset_name}___"
        all_tables = conn.list_tables()
        return [
            t[len(prefix) :]
            for t in all_tables
            if t.startswith(prefix) and not t.startswith(f"{prefix}_dlt_")
        ]

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to ClickHouse and optionally load data."""
        creds = get_clickhouse_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        dataset_name = f"sqlite_import_{db_hash}"

        # Set dlt credentials via environment variables
        # dlt uses native TCP port (9000) for data loading
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__HOST"] = creds["host"]
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__PORT"] = str(creds["port"])
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__HTTP_PORT"] = str(
            creds["http_port"]
        )
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__USERNAME"] = creds["user"]
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__PASSWORD"] = creds["password"]
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__DATABASE"] = creds["database"]
        os.environ["DESTINATION__CLICKHOUSE__CREDENTIALS__SECURE"] = (
            "0"  # No SSL for local
        )

        should_reload = True
        table_names = []

        if not load_data:
            should_reload = False
        else:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            try:
                # dlt creates tables with prefix {dataset_name}___ in default database
                test_conn = ibis.clickhouse.connect(
                    host=creds["host"],
                    port=creds["http_port"],
                    database=creds["database"],
                    user=creds["user"],
                    password=creds["password"],
                )
                existing_tables = _normalize_table_names_for_matching(
                    self._get_prefixed_tables(test_conn, dataset_name)
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    should_reload = False
                else:
                    print(
                        f"ClickHouse dataset {dataset_name} missing tables. Will reload."
                    )
                test_conn = None
            except Exception as e:
                print(f"ClickHouse connection error: {e}. Will create tables.")

        if should_reload:
            sqlite_conn = create_sqlite_connection(db_path)
            _load_data_with_dlt(
                sqlite_conn=sqlite_conn,
                table_names=table_names,
                backend="clickhouse",
                dataset_name=dataset_name,
                write_disposition=write_disposition,
            )
            sqlite_conn = None
            print("dlt pipeline completed successfully!")

        # Connect to default database where dlt created the tables
        conn = ibis.clickhouse.connect(
            host=creds["host"],
            port=creds["http_port"],
            database=creds["database"],
            user=creds["user"],
            password=creds["password"],
        )

        # Store dataset name for table access
        conn._clickhouse_dataset_name = dataset_name  # type: ignore[attr-defined]

        return conn
