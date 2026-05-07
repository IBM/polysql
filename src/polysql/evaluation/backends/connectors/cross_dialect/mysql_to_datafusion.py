"""MySQL to DataFusion cross-dialect connector."""

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
    get_mysql_credentials,
)


class MySQLToDataFusionConnector(BaseBackendConnector):
    """Connector for loading MySQL data into DataFusion backend."""

    def __init__(self):
        """Initialize MySQL to DataFusion connector."""
        super().__init__("datafusion")

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
        """Connect to DataFusion in-memory backend and load data from MySQL."""
        conn = ibis.datafusion.connect()

        if load_data:
            self._validate_source_exists(db_path)
            table_names = [
                t for t in self._get_source_tables(db_path) if not t.startswith("_dlt_")
            ]

            creds = get_mysql_credentials()
            mysql_conn = ibis.mysql.connect(
                host=creds["host"],
                port=creds["port"],
                database=str(db_path),
                user=creds["user"],
                password=creds["password"],
            )

            for table_name in table_names:
                df = mysql_conn.table(table_name).execute()
                df = _clean_dataframe_for_load(df, backend="datafusion")
                table = _dataframe_to_arrow_table_no_metadata(df)
                conn.create_table(table_name, table)

            mysql_conn = None

        return conn
