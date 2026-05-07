"""PostgreSQL to MySQL cross-dialect connector."""

import hashlib
from pathlib import Path
from typing import List, Literal, Union

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _check_sql_database_exists,
    _create_ibis_dlt_source,
    _create_mysql_database,
    _ensure_mysql_dlt_metadata_columns_longtext,
    _ensure_schema_cache,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_mysql_credentials,
    get_postgres_credentials,
)


class PostgresToMySQLConnector(BaseBackendConnector):
    """Connector that loads data from native PostgreSQL to MySQL using dlt."""

    def __init__(self):
        """Initialize PostgreSQL to MySQL connector."""
        super().__init__("mysql")

    def _get_source_tables(self, db_path: Union[str, Path]) -> List[str]:
        """Get list of tables from source PostgreSQL database."""
        creds = get_postgres_credentials()

        pg_conn = ibis.postgres.connect(
            host=creds["host"],
            port=creds["port"],
            database=str(db_path),
            user=creds["user"],
            password=creds["password"],
        )
        table_names = pg_conn.list_tables()
        pg_conn = None
        return table_names

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to MySQL and load data from PostgreSQL database using dlt."""
        pg_creds = get_postgres_credentials()
        mysql_creds = get_mysql_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        mysql_database = f"postgres_import_{db_hash}"

        password = mysql_creds["password"] if mysql_creds["password"] else None

        table_names = []

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            mysql_conn_kwargs = {
                "host": mysql_creds["host"],
                "port": mysql_creds["port"],
                "database": mysql_database,
                "user": mysql_creds["user"],
                "password": mysql_creds["password"],
            }

            existing_conn = _check_sql_database_exists(
                ibis.mysql.connect, mysql_conn_kwargs, table_names, mysql_database
            )

            if existing_conn is not None:
                _ensure_mysql_dlt_metadata_columns_longtext(
                    mysql_database,
                    mysql_creds["host"],
                    mysql_creds["port"],
                    mysql_creds["user"],
                    password,
                )
                _ensure_schema_cache(existing_conn, db_path, "mysql")
                return existing_conn

            pg_conn = ibis.postgres.connect(
                host=pg_creds["host"],
                port=pg_creds["port"],
                database=str(db_path),
                user=pg_creds["user"],
                password=pg_creds["password"],
            )

            _create_mysql_database(
                mysql_database,
                mysql_creds["host"],
                mysql_creds["port"],
                mysql_creds["user"],
                password,
            )

            if password:
                connection_string = f"mysql+pymysql://{mysql_creds['user']}:{password}@{mysql_creds['host']}:{mysql_creds['port']}/{mysql_database}"
            else:
                connection_string = f"mysql+pymysql://{mysql_creds['user']}@{mysql_creds['host']}:{mysql_creds['port']}/{mysql_database}"

            source = _create_ibis_dlt_source(pg_conn, table_names, write_disposition)

            pipeline = dlt.pipeline(
                pipeline_name="postgres_to_mysql",
                destination=dlt.destinations.sqlalchemy(credentials=connection_string),
                dataset_name=mysql_database,
            )
            _ensure_mysql_dlt_metadata_columns_longtext(
                mysql_database,
                mysql_creds["host"],
                mysql_creds["port"],
                mysql_creds["user"],
                password,
            )
            try:
                pipeline.run(source())
            except Exception as exc:
                message = str(exc)
                if "Data too long for column 'schema'" in message:
                    print("Detected MySQL metadata overflow. Expanding and retrying...")
                    _ensure_mysql_dlt_metadata_columns_longtext(
                        mysql_database,
                        mysql_creds["host"],
                        mysql_creds["port"],
                        mysql_creds["user"],
                        password,
                    )
                    pipeline.run(source())
                else:
                    raise

            pg_conn = None

        mysql_conn_kwargs = {
            "host": mysql_creds["host"],
            "port": mysql_creds["port"],
            "database": mysql_database,
            "user": mysql_creds["user"],
            "password": mysql_creds["password"],
        }

        mysql_conn = ibis.mysql.connect(**mysql_conn_kwargs)

        _ensure_schema_cache(mysql_conn, db_path, "mysql")
        return mysql_conn
