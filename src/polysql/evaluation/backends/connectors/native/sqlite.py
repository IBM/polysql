"""SQLite native connector."""

from pathlib import Path
from typing import Literal

from ibis import BaseBackend

from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    create_sqlite_connection,
)


class SQLiteConnector(BaseBackendConnector):
    """Connector for SQLite backend."""

    def __init__(self):
        """Initialize SQLite connector."""
        super().__init__("sqlite")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to SQLite database directly."""
        self._validate_source_exists(db_path)
        return create_sqlite_connection(db_path)
