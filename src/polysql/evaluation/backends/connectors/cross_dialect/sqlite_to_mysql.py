"""SQLite to MySQL cross-dialect connector."""

import hashlib
import logging
from pathlib import Path
from typing import Literal

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _check_sql_database_exists,
    _create_mysql_database,
    _create_sqlite_dlt_source,
    _ensure_mysql_dlt_metadata_columns_longtext,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
    get_mysql_credentials,
)


class SQLiteToMySQLConnector(BaseBackendConnector):
    """Connector for loading SQLite data into MySQL backend."""

    def __init__(self):
        """Initialize SQLite to MySQL connector."""
        super().__init__("mysql")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to MySQL and optionally load data."""
        creds = get_mysql_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        database_name = f"sqlite_import_{db_hash}"

        password = creds["password"] if creds["password"] else None

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            conn_kwargs = {
                "host": creds["host"],
                "port": creds["port"],
                "user": creds["user"],
                "password": creds["password"],
                "database": database_name,
            }
            existing_conn = _check_sql_database_exists(
                ibis.mysql.connect,
                conn_kwargs,
                table_names,
                f"MySQL '{database_name}'",
            )
            if existing_conn:
                return existing_conn

            _create_mysql_database(
                database_name, creds["host"], creds["port"], creds["user"], password
            )

            if password is not None:
                connection_string = f"mysql+pymysql://{creds['user']}:{password}@{creds['host']}:{creds['port']}/{database_name}"
            else:
                connection_string = f"mysql+pymysql://{creds['user']}@{creds['host']}:{creds['port']}/{database_name}"

            logging.getLogger("dlt").setLevel(logging.ERROR)

            print(f"Running dlt pipeline for MySQL with {len(table_names)} tables...")

            pipeline = dlt.pipeline(
                pipeline_name=f"sqlite_to_mysql_{database_name}",
                destination=dlt.destinations.sqlalchemy(credentials=connection_string),
                dataset_name=database_name,
            )

            sqlite_conn = create_sqlite_connection(db_path)
            sqlite_source = _create_sqlite_dlt_source(
                sqlite_conn, table_names, write_disposition
            )

            _ensure_mysql_dlt_metadata_columns_longtext(
                database_name, creds["host"], creds["port"], creds["user"], password
            )

            try:
                pipeline.run(sqlite_source())
            except Exception as exc:
                message = str(exc)
                if "Data too long for column 'schema'" in message:
                    print(
                        "Detected MySQL metadata overflow. Expanding columns and retrying..."
                    )
                    _ensure_mysql_dlt_metadata_columns_longtext(
                        database_name,
                        creds["host"],
                        creds["port"],
                        creds["user"],
                        password,
                    )
                    pipeline.run(sqlite_source())
                else:
                    raise

            sqlite_conn = None
            print("dlt pipeline for MySQL completed successfully!")

        conn_kwargs = {
            "host": creds["host"],
            "port": creds["port"],
            "user": creds["user"],
            "password": creds["password"],
            "database": database_name,
            "connect_timeout": 60,
            "read_timeout": 60,
            "write_timeout": 60,
        }

        conn = ibis.mysql.connect(**conn_kwargs)
        return conn
