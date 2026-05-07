"""SQLite to PostgreSQL cross-dialect connector."""

import hashlib
import logging
from pathlib import Path
from typing import Literal
import os

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _check_sql_database_exists,
    _create_postgres_database,
    _create_sqlite_dlt_source,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
    get_postgres_credentials,
)


class SQLiteToPostgresConnector(BaseBackendConnector):
    """Connector for loading SQLite data into PostgreSQL backend."""

    def __init__(self):
        """Initialize SQLite to Postgres connector."""
        super().__init__("postgres")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to PostgreSQL and optionally load data."""
        creds = get_postgres_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        database_name = f"sqlite_import_{db_hash}"

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            conn_kwargs = {
                "host": creds["host"],
                "port": creds["port"],
                "user": creds["user"],
                "password": creds["password"],
                "database": database_name,
                "schema": database_name,
            }
            existing_conn = _check_sql_database_exists(
                ibis.postgres.connect,
                conn_kwargs,
                table_names,
                f"PostgreSQL '{database_name}'",
            )
            if existing_conn:
                return existing_conn

            _create_postgres_database(
                database_name, creds["host"], creds["port"], creds["user"], creds["password"]
            )

            os.environ["DESTINATION__POSTGRES__CREDENTIALS__DATABASE"] = database_name
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__USERNAME"] = creds["user"]
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__PASSWORD"] = creds["password"]
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__HOST"] = creds["host"]
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__PORT"] = str(creds["port"])

            logging.getLogger("dlt").setLevel(logging.ERROR)

            print(f"Running dlt pipeline for PostgreSQL with {len(table_names)} tables...")

            pipeline = dlt.pipeline(
                pipeline_name=f"sqlite_to_postgres_{database_name}",
                destination="postgres",
                dataset_name=database_name,
            )

            sqlite_conn = create_sqlite_connection(db_path)
            sqlite_source = _create_sqlite_dlt_source(
                sqlite_conn, table_names, write_disposition
            )
            pipeline.run(sqlite_source())
            sqlite_conn = None

            print("dlt pipeline for PostgreSQL completed successfully!")

        conn = ibis.postgres.connect(
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            database=database_name,
            schema=database_name,
        )
        return conn
