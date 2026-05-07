"""MySQL to DuckDB cross-dialect connector."""

import logging
from pathlib import Path
from typing import List, Literal, Union
import os

import dlt
import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.cache import get_cache_path
from polysql.evaluation.backends.connections import (
    _create_ibis_dlt_source,
    _ensure_schema_cache,
    _normalize_table_names_for_matching,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_mysql_credentials,
)


class MySQLToDuckDBConnector(BaseBackendConnector):
    """Connector that loads data from native MySQL to DuckDB using dlt."""

    def __init__(self):
        """Initialize MySQL to DuckDB connector."""
        super().__init__("duckdb")

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
        """Connect to DuckDB and optionally load data from MySQL source via dlt."""
        duckdb_path = get_cache_path(
            source_db_path=db_path,
            source_type="mysql",
            target_type="duckdb",
            namespace=namespace if namespace else None,
        )
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        if not load_data:
            if duckdb_path.exists():
                conn = ibis.duckdb.connect(str(duckdb_path))
                conn._converted_db_path = duckdb_path  # type: ignore[attr-defined]
                _ensure_schema_cache(conn, db_path, "duckdb")
                return conn
            raise FileNotFoundError(
                f"DuckDB cache not found at {duckdb_path}. "
                "Re-run without --skip-data-load to build the cache."
            )

        self._validate_source_exists(db_path)

        table_names = [
            t for t in self._get_source_tables(db_path) if not t.startswith("_dlt_")
        ]

        if self._can_reuse_existing(duckdb_path, table_names):
            conn = ibis.duckdb.connect(str(duckdb_path))
            conn._converted_db_path = duckdb_path  # type: ignore[attr-defined]
            _ensure_schema_cache(conn, db_path, "duckdb")
            return conn

        self._load_data_with_dlt(db_path, duckdb_path, table_names, write_disposition)

        conn = ibis.duckdb.connect(str(duckdb_path))
        conn._converted_db_path = duckdb_path  # type: ignore[attr-defined]
        _ensure_schema_cache(conn, db_path, "duckdb")
        return conn

    def _can_reuse_existing(
        self, duckdb_path: Path, required_tables: List[str]
    ) -> bool:
        """Return True if cached DuckDB has all required tables."""
        if not duckdb_path.exists():
            return False

        test_conn = ibis.duckdb.connect(str(duckdb_path))
        existing_tables = _normalize_table_names_for_matching(
            [t for t in test_conn.list_tables() if "_dlt_" not in t]
        )
        required_normalized = _normalize_table_names_for_matching(required_tables)
        return required_normalized.issubset(existing_tables)

    def _load_data_with_dlt(
        self,
        source_db_path: Path,
        target_db_path: Path,
        table_names: List[str],
        write_disposition: str,
    ) -> None:
        """Load data from MySQL source to DuckDB using dlt."""
        logging.getLogger("dlt").setLevel(logging.ERROR)

        creds = get_mysql_credentials()

        mysql_conn = ibis.mysql.connect(
            host=creds["host"],
            port=creds["port"],
            database=str(source_db_path),
            user=creds["user"],
            password=creds["password"],
        )

        print(f"Loading {len(table_names)} tables from MySQL to DuckDB using dlt...")

        pipeline = dlt.pipeline(
            pipeline_name=f"mysql_to_duckdb_{target_db_path.stem}",
            destination=dlt.destinations.duckdb(str(target_db_path), schema="main"),
            dataset_name="main",
        )

        mysql_source = _create_ibis_dlt_source(mysql_conn, table_names, write_disposition)
        pipeline.run(mysql_source())
        mysql_conn = None

        print(f"Successfully loaded {len(table_names)} tables to {target_db_path}")
