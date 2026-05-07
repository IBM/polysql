"""PostgreSQL to DataFusion cross-dialect connector."""

from pathlib import Path
from typing import List, Literal, Union

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _clean_dataframe_for_load,
    _dataframe_to_arrow_table_no_metadata,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_postgres_credentials,
)


class PostgresToDataFusionConnector(BaseBackendConnector):
    """Connector for loading PostgreSQL data into DataFusion backend."""

    def __init__(self):
        """Initialize PostgreSQL to DataFusion connector."""
        super().__init__("datafusion")

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
        """Connect to DataFusion in-memory backend and load data from PostgreSQL."""
        conn = ibis.datafusion.connect()

        if load_data:
            self._validate_source_exists(db_path)
            table_names = [
                t for t in self._get_source_tables(db_path) if not t.startswith("_dlt_")
            ]

            creds = get_postgres_credentials()
            pg_conn = ibis.postgres.connect(
                host=creds["host"],
                port=creds["port"],
                database=str(db_path),
                user=creds["user"],
                password=creds["password"],
            )

            for table_name in table_names:
                df = pg_conn.table(table_name).execute()
                df = _clean_dataframe_for_load(df, backend="datafusion")
                table = _dataframe_to_arrow_table_no_metadata(df)
                conn.create_table(table_name, table)

            pg_conn = None

        return conn
