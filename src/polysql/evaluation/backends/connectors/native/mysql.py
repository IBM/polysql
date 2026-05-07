"""MySQL native connectors."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

import ibis
from ibis import BaseBackend

from polysql.evaluation.backends.connectors.base import (
    BaseBackendConnector,
    get_mysql_credentials,
)
from polysql.evaluation.backends.connections import (
    _check_sql_database_exists,
    _create_mysql_database,
    _ensure_schema_cache,
)


class MySQLConnector(BaseBackendConnector):
    """Connector for native MySQL databases.

    Connects directly to existing MySQL databases without data conversion.
    """

    def __init__(self):
        """Initialize MySQL connector."""
        super().__init__("mysql")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to an existing MySQL database."""
        creds = get_mysql_credentials()
        database_name = str(db_path)

        conn_kwargs = {
            "host": creds["host"],
            "port": creds["port"],
            "database": database_name,
            "user": creds["user"],
            "password": creds["password"],
        }

        try:
            conn = ibis.mysql.connect(**conn_kwargs)
            _ = conn.list_tables()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MySQL database '{database_name}': {e}"
            )

        _ensure_schema_cache(conn, db_path, "mysql")
        return conn


class MySQLDumpConnector(BaseBackendConnector):
    """Connector for MySQL backend with native dump file loading.

    Designed for datasets that provide MySQL dump files (.sql).
    """

    def __init__(self):
        """Initialize MySQL dump connector."""
        super().__init__("mysql")

    def connect(
        self,
        db_path: Path,
        load_data: bool = True,
        write_disposition: Literal["replace", "append", "merge"] = "replace",
        namespace: str = "",
    ) -> BaseBackend:
        """Connect to MySQL and optionally load data from MySQL dump file."""
        import hashlib

        self._validate_source_exists(db_path)
        creds = get_mysql_credentials()

        db_hash = hashlib.md5(str(db_path).encode()).hexdigest()[:8]
        database_name = f"beaver_mysql_{db_hash}"

        password = creds["password"] if creds["password"] else None

        if load_data:
            table_names = self._get_table_names_from_dump(db_path)

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

            print(f"Loading MySQL dump into database '{database_name}'...")
            self._load_mysql_dump(
                db_path,
                database_name,
                creds["host"],
                creds["port"],
                creds["user"],
                password,
            )
            print(f"MySQL dump loaded successfully into '{database_name}'")

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

    def _get_table_names_from_dump(self, dump_path: Path) -> List[str]:
        """Extract table names from MySQL dump file."""
        table_names = []
        with open(dump_path, "r") as f:
            for line in f:
                match = re.match(
                    r"CREATE TABLE [IF NOT EXISTS ]*[`]?(\w+)[`]?", line, re.IGNORECASE
                )
                if match:
                    table_names.append(match.group(1))
        return table_names

    def _load_mysql_dump(
        self,
        dump_path: Path,
        database_name: str,
        host: str,
        port: int,
        user: str,
        password: Optional[str],
    ) -> None:
        """Load MySQL dump file using mysql CLI."""
        import os

        filtered_sql_lines = []
        with open(dump_path, "r") as f:
            for line in f:
                line_upper = line.strip().upper()
                if line_upper.startswith("CREATE DATABASE") or line_upper.startswith(
                    "USE "
                ):
                    continue
                filtered_sql_lines.append(line)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as tmp_file:
            tmp_file.writelines(filtered_sql_lines)
            tmp_path = tmp_file.name

        cmd = [
            "mysql",
            f"--host={host}",
            f"--port={port}",
            f"--user={user}",
            database_name,
        ]

        if password is not None:
            cmd.insert(4, f"--password={password}")

        with open(tmp_path, "r") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
            )

        os.unlink(tmp_path)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to load MySQL dump: {result.stderr}")
