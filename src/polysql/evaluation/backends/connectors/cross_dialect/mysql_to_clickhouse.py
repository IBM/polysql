"""MySQL to ClickHouse cross-dialect connector."""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Literal, Union

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _create_ibis_dlt_source,
    _normalize_table_names_for_matching,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_clickhouse_credentials,
    get_mysql_credentials,
)


class MySQLToClickHouseConnector(BaseBackendConnector):
    """Connector for loading MySQL data into ClickHouse backend."""

    def __init__(self):
        """Initialize MySQL to ClickHouse connector."""
        super().__init__("clickhouse")

    def _get_source_tables(self, db_path: Union[str, Path]) -> List[str]:
        """Get list of tables from source MySQL database."""
        creds = get_mysql_credentials()

        mysql_conn = ibis.mysql.connect(
            host=creds["host"],
            port=creds["port"],
            database=str(db_path),
            user=creds["user"],
            password=creds["password"],
        )
        table_names = mysql_conn.list_tables()
        mysql_conn = None
        return table_names

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
        """Connect to ClickHouse and optionally load data from MySQL."""
        creds = get_clickhouse_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        dataset_name = f"mysql_import_{db_hash}"

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
            table_names = [
                t for t in self._get_source_tables(db_path) if not t.startswith("_dlt_")
            ]

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
            self._load_data_with_dlt(db_path, table_names, dataset_name, write_disposition)
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

    def _load_data_with_dlt(
        self,
        source_db_path: Path,
        table_names: List[str],
        dataset_name: str,
        write_disposition: str,
    ) -> None:
        """Load data from MySQL source to ClickHouse using dlt."""
        logging.getLogger("dlt").setLevel(logging.ERROR)

        creds = get_mysql_credentials()

        mysql_conn = ibis.mysql.connect(
            host=creds["host"],
            port=creds["port"],
            database=str(source_db_path),
            user=creds["user"],
            password=creds["password"],
        )

        print(f"Running dlt pipeline for clickhouse with {len(table_names)} tables...")

        pipeline = dlt.pipeline(
            pipeline_name=f"mysql_to_clickhouse_{dataset_name}",
            destination="clickhouse",
            dataset_name=dataset_name,
        )

        mysql_source = _create_ibis_dlt_source(mysql_conn, table_names, write_disposition)
        pipeline.run(mysql_source())
        mysql_conn = None
