"""SQLite to DataFusion cross-dialect connector."""

from pathlib import Path
from typing import Literal

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connections import (
    _clean_dataframe_for_load,
    _dataframe_to_arrow_table_no_metadata,
)
from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
)


class SQLiteToDataFusionConnector(BaseBackendConnector):
    """Connector for loading SQLite data into DataFusion backend."""

    def __init__(self):
        """Initialize SQLite to DataFusion connector."""
        super().__init__("datafusion")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to DataFusion in-memory backend."""
        conn = ibis.datafusion.connect()

        if load_data:
            self._validate_source_exists(db_path)
            table_names = self._get_source_tables(db_path)

            sqlite_conn = create_sqlite_connection(db_path)
            for table_name in table_names:
                df = sqlite_conn.table(table_name).execute()
                df = _clean_dataframe_for_load(df, backend="datafusion")
                table = _dataframe_to_arrow_table_no_metadata(df)
                conn.create_table(table_name, table)
            sqlite_conn = None  # Close connection

        return conn
