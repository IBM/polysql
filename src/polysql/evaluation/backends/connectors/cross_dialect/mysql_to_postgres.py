"""MySQL to PostgreSQL cross-dialect connector."""

import hashlib
import subprocess
from pathlib import Path
from typing import List, Literal, Union
import os

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _check_sql_database_exists,
    _create_ibis_dlt_source,
    _ensure_schema_cache,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_mysql_credentials,
    get_postgres_credentials,
)


class MySQLToPostgresConnector(BaseBackendConnector):
    """Connector that loads data from native MySQL to PostgreSQL using dlt."""

    def __init__(self):
        """Initialize MySQL to PostgreSQL connector."""
        super().__init__("postgres")

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
        """Connect to PostgreSQL and load data from MySQL database using dlt."""
        mysql_creds = get_mysql_credentials()
        pg_creds = get_postgres_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        pg_database = f"mysql_import_{db_hash}"

        table_names = []

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            pg_conn_kwargs = {
                "host": pg_creds["host"],
                "port": pg_creds["port"],
                "database": pg_database,
                "user": pg_creds["user"],
                "password": pg_creds["password"],
                "schema": pg_database,
            }
            existing_conn = _check_sql_database_exists(
                ibis.postgres.connect, pg_conn_kwargs, table_names, pg_database
            )

            if existing_conn is not None:
                _ensure_schema_cache(existing_conn, db_path, "postgres")
                return existing_conn

            mysql_conn = ibis.mysql.connect(
                host=mysql_creds["host"],
                port=mysql_creds["port"],
                database=str(db_path),
                user=mysql_creds["user"],
                password=mysql_creds["password"],
            )

            # Create database - connect to 'postgres' system database first
            # We need -d postgres because connecting without a database defaults to user's name
            env = os.environ.copy()
            if pg_creds["password"]:
                env["PGPASSWORD"] = pg_creds["password"]

            result = subprocess.run(
                [
                    "psql",
                    "-U",
                    pg_creds["user"],
                    "-h",
                    pg_creds["host"],
                    "-d",
                    "postgres",
                    "-c",
                    f"CREATE DATABASE {pg_database};",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create PostgreSQL database {pg_database}: {result.stderr}"
                )

            os.environ["DESTINATION__POSTGRES__CREDENTIALS__DATABASE"] = pg_database
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__HOST"] = pg_creds["host"]
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__PORT"] = str(pg_creds["port"])
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__USERNAME"] = pg_creds["user"]
            os.environ["DESTINATION__POSTGRES__CREDENTIALS__PASSWORD"] = pg_creds["password"]

            source = _create_ibis_dlt_source(mysql_conn, table_names, write_disposition)

            pipeline = dlt.pipeline(
                pipeline_name="mysql_to_postgres",
                destination="postgres",
                dataset_name=pg_database,
            )
            pipeline.run(source())

            mysql_conn = None

        pg_conn = ibis.postgres.connect(
            host=pg_creds["host"],
            port=pg_creds["port"],
            database=pg_database,
            user=pg_creds["user"],
            password=pg_creds["password"],
            schema=pg_database,
        )

        _ensure_schema_cache(pg_conn, db_path, "postgres")
        return pg_conn
