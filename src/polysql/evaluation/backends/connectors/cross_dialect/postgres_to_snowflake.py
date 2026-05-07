"""PostgreSQL to Snowflake cross-dialect connector."""

import logging
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
    get_snowflake_names,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_postgres_credentials,
    get_snowflake_credentials,
)


class PostgresToSnowflakeConnector(BaseBackendConnector):
    """Connector that loads data from native PostgreSQL to Snowflake using dlt."""

    def __init__(self):
        """Initialize PostgreSQL to Snowflake connector."""
        super().__init__("snowflake")

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
        """Connect to Snowflake and load data from PostgreSQL database using dlt."""
        pg_creds = get_postgres_credentials()
        sf_creds = get_snowflake_credentials()

        if not all([sf_creds["user"], sf_creds["password"], sf_creds["account"]]):
            raise ValueError(
                "Snowflake credentials not found. Set SFDBUSER, SFDBPASSWORD, SFDBACCOUNT."
            )

        if not namespace:
            raise ValueError(
                "namespace parameter is required for PostgresToSnowflake connector."
            )

        dataset_name = namespace.split("_", 1)[1] if "_" in namespace else namespace

        db_id = ""
        db_path_str = str(db_path)
        if db_path_str.startswith("minidev_postgres_"):
            db_id = db_path_str.replace("minidev_postgres_", "", 1)

        (
            database_name,
            schema_name,
            effective_database_name,
            effective_schema_name,
        ) = get_snowflake_names(db_path, dataset_name, db_id=db_id)

        table_names = []

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            try:
                test_conn = ibis.snowflake.connect(
                    user=sf_creds["user"],
                    password=sf_creds["password"],
                    account=sf_creds["account"],
                    warehouse=sf_creds["warehouse"],
                    database=effective_database_name,
                    schema=effective_schema_name,
                )

                existing_tables = _normalize_table_names_for_matching(
                    [t for t in test_conn.list_tables() if not t.startswith("_dlt_")]
                )
                required_tables = _normalize_table_names_for_matching(table_names)

                if required_tables.issubset(existing_tables):
                    _ensure_schema_cache(
                        test_conn, db_path, "snowflake", schema_name=effective_schema_name
                    )
                    return test_conn
                else:
                    print(f"Schema {effective_schema_name} missing tables. Will reload.")
                    test_conn.raw_sql(
                        f"DROP SCHEMA IF EXISTS {effective_schema_name} CASCADE"
                    )
                test_conn = None
            except Exception:
                print(f"Schema {effective_schema_name} does not exist. Will create.")

            pg_conn = ibis.postgres.connect(
                host=pg_creds["host"],
                port=pg_creds["port"],
                database=str(db_path),
                user=pg_creds["user"],
                password=pg_creds["password"],
            )

            try:
                admin_conn = ibis.snowflake.connect(
                    user=sf_creds["user"],
                    password=sf_creds["password"],
                    account=sf_creds["account"],
                    warehouse=sf_creds["warehouse"] or "COMPUTE_WH",
                    database=effective_database_name,
                    schema=effective_schema_name,
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
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create/access Snowflake database: {e}"
                )

            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__DATABASE"] = effective_database_name
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__USERNAME"] = sf_creds["user"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__PASSWORD"] = sf_creds["password"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__HOST"] = sf_creds["account"]
            os.environ["DESTINATION__SNOWFLAKE__CREDENTIALS__WAREHOUSE"] = (
                sf_creds["warehouse"] or "COMPUTE_WH"
            )

            print(f"Loading {len(table_names)} tables from PostgreSQL to Snowflake...")

            logging.getLogger("dlt").setLevel(logging.ERROR)

            source = _create_ibis_dlt_source(pg_conn, table_names, write_disposition)

            pipeline = dlt.pipeline(
                pipeline_name=f"postgres_to_snowflake_{effective_schema_name.lower()}",
                destination=dlt.destinations.snowflake(
                    enable_dataset_name_normalization=False
                ),
                dataset_name=schema_name,
            )

            print(f"Running dlt pipeline for Snowflake with {len(table_names)} tables...")
            pipeline.run(source())
            print("dlt pipeline completed successfully!")

            pg_conn = None

        snowflake_conn = ibis.snowflake.connect(
            user=sf_creds["user"],
            password=sf_creds["password"],
            account=sf_creds["account"],
            warehouse=sf_creds["warehouse"],
            database=effective_database_name,
            schema=effective_schema_name,
        )

        _ensure_schema_cache(
            snowflake_conn, db_path, "snowflake", schema_name=effective_schema_name
        )
        return snowflake_conn
