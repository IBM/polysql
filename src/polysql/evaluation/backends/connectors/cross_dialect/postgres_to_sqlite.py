"""PostgreSQL to SQLite cross-dialect connector."""

from pathlib import Path
from typing import List, Literal, Union
import os

import dlt
import ibis
from ibis import BaseBackend
from dlt.common.normalizers.naming.snake_case import NamingConvention

from polysql.evaluation.backends.cache import get_cache_path
from polysql.evaluation.backends.connections import (
    _create_ibis_dlt_source,
    _ensure_schema_cache,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_postgres_credentials,
)


class PostgresToSQLiteConnector(BaseBackendConnector):
    """Connector that loads data from native PostgreSQL to SQLite using dlt."""

    def __init__(self):
        """Initialize PostgreSQL to SQLite connector."""
        super().__init__("sqlite")

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
        """Connect to SQLite and optionally load data from PostgreSQL using dlt."""
        creds = get_postgres_credentials()

        sqlite_path = get_cache_path(
            source_db_path=Path(db_path),
            source_type="postgres",
            target_type="sqlite",
            namespace=namespace if namespace else None,
        )

        if sqlite_path.exists() and not load_data:
            sqlite_conn = ibis.sqlite.connect(str(sqlite_path))
            sqlite_conn._converted_db_path = sqlite_path
            _ensure_schema_cache(sqlite_conn, db_path, "sqlite")
            return sqlite_conn

        if load_data:
            pg_conn_temp = ibis.postgres.connect(
                host=creds["host"],
                port=creds["port"],
                database=str(db_path),
                user=creds["user"],
                password=creds["password"],
            )
            table_names = [
                t for t in pg_conn_temp.list_tables() if not t.startswith("_dlt_")
            ]
            pg_conn_temp = None

            if sqlite_path.exists():
                sqlite_conn = ibis.sqlite.connect(str(sqlite_path))
                existing_tables = set(sqlite_conn.list_tables())
                required_tables = set(
                    t for t in table_names if not t.startswith("_dlt_")
                )

                naming = NamingConvention()
                existing_normalized = {
                    naming.normalize_identifier(t): t for t in existing_tables
                }
                required_normalized = {
                    naming.normalize_identifier(t): t for t in required_tables
                }

                # print(f"  Cache check for {sqlite_path.name}:")
                # print(f"    Required tables ({len(required_tables)}): {sorted(required_tables)}")
                # print(f"    Existing tables ({len(existing_tables)}): {sorted(existing_tables)}")

                if required_normalized.keys() <= existing_normalized.keys():
                    # print("  ✓ Using cached SQLite database (skipping dlt)")
                    sqlite_conn._converted_db_path = sqlite_path
                    _ensure_schema_cache(sqlite_conn, db_path, "sqlite")
                    return sqlite_conn

                missing_normalized = required_normalized.keys() - existing_normalized.keys()
                missing_original = [required_normalized[n] for n in missing_normalized]
                # print(f"  ✗ Cache incomplete, missing {len(missing_original)} tables")
                # print("  Removing incomplete cache and reloading...")
                sqlite_path.unlink()

            pg_conn = ibis.postgres.connect(
                host=creds["host"],
                port=creds["port"],
                database=str(db_path),
                user=creds["user"],
                password=creds["password"],
            )

            os.environ["LOAD__WORKERS"] = "1"

            print(f"Loading {len(table_names)} tables from PostgreSQL to SQLite...")

            pipeline_name = f"postgres_to_sqlite_{sqlite_path.stem.split('_')[-1]}"
            pipeline = dlt.pipeline(
                pipeline_name=pipeline_name,
                destination=dlt.destinations.sqlalchemy(f"sqlite:///{sqlite_path}"),
                dataset_name="main",
            )

            source = _create_ibis_dlt_source(pg_conn, table_names, write_disposition)
            pipeline.run(source())

            pg_conn = None

            print(f"Successfully loaded {len(table_names)} tables to {sqlite_path}")

        sqlite_conn = ibis.sqlite.connect(str(sqlite_path))
        sqlite_conn._converted_db_path = sqlite_path
        _ensure_schema_cache(sqlite_conn, db_path, "sqlite")
        return sqlite_conn
