"""SQLite to Snowflake cross-dialect connector."""

from pathlib import Path
from typing import Literal
import os

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _ensure_schema_cache,
    _load_data_with_dlt,
    _normalize_table_names_for_matching,
    get_snowflake_names,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
    get_snowflake_credentials,
)


class SQLiteToSnowflakeConnector(BaseBackendConnector):
    """Connector for loading SQLite data into Snowflake backend."""

    def __init__(self):
        """Initialize SQLite to Snowflake connector."""
        super().__init__("snowflake")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to Snowflake and optionally load data."""
        creds = get_snowflake_credentials()

        missing_creds = [
            k
            for k, v in {
                "SFDBUSER": creds["user"],
                "SFDBPASSWORD": creds["password"],
                "SFDBACCOUNT": creds["account"],
            }.items()
            if not v
        ]
        if missing_creds:
            raise ValueError(
                f"Snowflake credentials missing: {', '.join(sorted(missing_creds))}. "
                "Set them in .env."
            )

        if not namespace:
            raise ValueError(
                "namespace parameter is required for SQLite to Snowflake connector."
            )

        dataset_name = namespace.split("_", 1)[1] if "_" in namespace else namespace

        (
            database_name,
            schema_name,
            effective_database_name,
            effective_schema_name,
        ) = get_snowflake_names(db_path, dataset_name)

        should_reload = True
        table_names = []

        if not load_data:
            should_reload = False
        else:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            # Ensure database exists
            try:
                admin_conn = ibis.snowflake.connect(
                    user=creds["user"],
                    password=creds["password"],
                    account=creds["account"],
                    warehouse=creds["warehouse"] or "COMPUTE_WH",
                    database=effective_database_name,
                    schema_name=effective_schema_name,
                )
                admin_conn.raw_sql(
                    f"CREATE DATABASE IF NOT EXISTS {effective_database_name}"
                )
                admin_conn.raw_sql(f"USE DATABASE {effective_database_name}")
                result = admin_conn.raw_sql("SELECT CURRENT_DATABASE()")
                current_db = result.fetchone()[0]
                if current_db != effective_database_name:
                    raise ValueError(
                        f"Failed to USE database {effective_database_name}"
                    )
                print(f"✓ Created and verified Snowflake database: {effective_database_name}")
                admin_conn = None
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create/access Snowflake database {effective_database_name}: {exc}"
                )

            try:
                test_conn = ibis.snowflake.connect(
                    user=creds["user"],
                    password=creds["password"],
                    account=creds["account"],
                    warehouse=creds["warehouse"] or "COMPUTE_WH",
                    database=effective_database_name,
                    schema=effective_schema_name,
                )

                existing_tables = _normalize_table_names_for_matching(
                    [
                        tab
                        for tab in test_conn.list_tables()
                        if not tab.startswith("dlt")
                    ]
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    print(f"Reusing existing Snowflake schema {effective_schema_name}")
                    should_reload = False
                else:
                    print(f"Schema {effective_schema_name} missing tables. Will reload.")
                    test_conn.raw_sql(
                        f"DROP SCHEMA IF EXISTS {effective_schema_name} CASCADE"
                    )
                test_conn = None
            except Exception as e:
                print(f"Schema {effective_schema_name} does not exist: {e}")

        if should_reload:
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__DATABASE"] = database_name
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__USERNAME"] = creds["user"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__PASSWORD"] = creds["password"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__HOST"] = creds["account"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__WAREHOUSE"] = (
                creds["warehouse"] or "COMPUTE_WH"
            )

            print(f"Loading {len(table_names)} tables from SQLite to Snowflake...")
            sqlite_conn = create_sqlite_connection(db_path)
            _load_data_with_dlt(
                sqlite_conn=sqlite_conn,
                table_names=table_names,
                backend="snowflake",
                dataset_name=schema_name,
                write_disposition=write_disposition,
            )
            sqlite_conn = None

        conn = ibis.snowflake.connect(
            user=creds["user"],
            password=creds["password"],
            account=creds["account"],
            warehouse=creds["warehouse"] or "COMPUTE_WH",
            database=effective_database_name,
            schema=effective_schema_name,
        )
        _ensure_schema_cache(
            conn, db_path, "snowflake", schema_name=effective_schema_name
        )
        return conn
