"""MySQL to BigQuery cross-dialect connector."""

import hashlib
from pathlib import Path
from typing import List, Literal, Union
import os

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _create_ibis_dlt_source,
    _ensure_schema_cache,
    _normalize_table_names_for_matching,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_bigquery_credentials,
    get_mysql_credentials,
)


class MySQLToBigQueryConnector(BaseBackendConnector):
    """Connector that loads data from native MySQL to BigQuery using dlt."""

    def __init__(self):
        """Initialize MySQL to BigQuery connector."""
        super().__init__("bigquery")

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

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to BigQuery and load data from MySQL database using dlt."""
        mysql_creds = get_mysql_credentials()
        bq_creds = get_bigquery_credentials()

        project_id = bq_creds["project_id"]
        credentials_path = bq_creds["credentials_path"]

        if not project_id:
            raise ValueError("BIGQUERY_PROJ environment variable not set.")
        if not credentials_path or not Path(credentials_path).exists():
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set or file not found.")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        dataset_name = f"mysql_import_{db_hash}"

        table_names = []

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            try:
                test_conn = ibis.bigquery.connect(
                    project_id=project_id, dataset_id=dataset_name
                )
                existing_tables = _normalize_table_names_for_matching(
                    [t for t in test_conn.list_tables() if not t.startswith("_dlt_")]
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    _ensure_schema_cache(test_conn, db_path, "bigquery")
                    return test_conn
                else:
                    print(f"BigQuery dataset {dataset_name} missing tables. Will reload.")
                test_conn = None
            except Exception as e:
                print(f"BigQuery dataset {dataset_name} does not exist: {e}. Will create.")

            mysql_conn = ibis.mysql.connect(
                host=mysql_creds["host"],
                port=mysql_creds["port"],
                database=str(db_path),
                user=mysql_creds["user"],
                password=mysql_creds["password"],
            )

            print(f"Loading {len(table_names)} tables from MySQL to BigQuery...")

            source = _create_ibis_dlt_source(mysql_conn, table_names, write_disposition)

            pipeline = dlt.pipeline(
                pipeline_name="mysql_to_bigquery",
                destination="bigquery",
                dataset_name=dataset_name,
            )
            pipeline.run(source())

            mysql_conn = None

            print(f"Successfully loaded {len(table_names)} tables to BigQuery")

        bigquery_conn = ibis.bigquery.connect(
            project_id=project_id, dataset_id=dataset_name
        )

        _ensure_schema_cache(bigquery_conn, db_path, "bigquery")
        return bigquery_conn
