"""Dialect-invariant query execution engine using Ibis.

This module provides execution engines for:
- SQL queries across different backends (GenericExecutionEngine)
- Substrait plans using DuckDB (SubstraitExecutionEngine)
"""

import base64
import hashlib
import os
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd
from diskcache import Cache
from dotenv import load_dotenv
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from polysql.evaluation.backends.connections import get_ibis_connection
from polysql.evaluation.utils.cache_paths import (
    cache_subdirs,
    ensure_inference_cache_env,
    migrate_legacy_caches,
)

# Load environment variables for backend credentials
load_dotenv()
migrate_legacy_caches()
ensure_inference_cache_env()

# Initialize cache for query executions
CACHE_DIR = cache_subdirs()["executions"]
execution_cache = Cache(str(CACHE_DIR))


# ============================================================================
# Dialect Handlers
# ============================================================================


class DialectHandler:
    """Base class for backend-specific query rewriting."""

    def rewrite_query(self, query: str, tables: List[str]) -> str:
        """
        Rewrite query for backend-specific requirements.

        Args:
            query: Original SQL query
            tables: List of table names available in the database

        Returns:
            Rewritten query
        """
        raise NotImplementedError("Subclasses must implement rewrite_query")


class StandardDialect(DialectHandler):
    """Standard SQL dialect - no rewriting needed."""

    def rewrite_query(self, query: str, tables: List[str]) -> str:
        """Return query unchanged."""
        return query


class BigQueryDialect(DialectHandler):
    """BigQuery dialect - requires dataset-qualified table names."""

    def __init__(self, dataset: str = "sqlite_import"):
        """
        Initialize BigQuery dialect handler.

        Args:
            dataset: Dataset name to use for table qualification
        """
        self.dataset = dataset

    def rewrite_query(self, query: str, tables: List[str]) -> str:
        """Add dataset prefix to table names if not already present."""
        import re

        rewritten = query
        for table in tables:
            # Check if table already has dataset prefix
            if (
                f"{self.dataset}.{table}" in query
                or f"`{self.dataset}.{table}`" in query
            ):
                # Already has prefix, skip
                continue

            # Match table name as whole word (not part of another word)
            # Avoid matching if it's already prefixed with any dataset
            pattern = rf"(?<!\.)\b{table}\b(?!\s*\.)"
            replacement = f"{self.dataset}.{table}"
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        return rewritten


class SnowflakeDialect(DialectHandler):
    """Snowflake dialect - no rewriting needed if LLM quotes identifiers."""

    def rewrite_query(self, query: str, tables: List[str]) -> str:
        """
        No rewriting needed for Snowflake.

        The LLM prompt instructs the model to quote all identifiers with double quotes,
        so we don't need to add quotes here. If we did, we'd double-quote already
        quoted identifiers.
        """
        return query


class MySQLDialect(DialectHandler):
    """MySQL dialect - rewrites identifiers to match dlt-normalized names."""

    @staticmethod
    def _normalize_identifier(name: str) -> str:
        """Normalize identifiers to match MySQL loader behavior."""
        import re

        normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        normalized = normalized.strip("_")
        return normalized.lower()

    def rewrite_query(self, query: str, tables: List[str]) -> str:
        """
        Rewrite query identifiers to align with MySQL loader naming.

        The dlt pipeline lowercases identifiers and replaces non-alphanumeric
        characters with underscores. We mirror that behavior here so that
        queries authored against the original SQLite schema can run against the
        loaded MySQL tables.
        """

        try:
            expression = parse_one(query, read="mysql")
        except ParseError:
            # Fallback to original query if parsing fails
            return query

        def transform(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Table):
                table_name = node.name
                if table_name:
                    node.set(
                        "this",
                        exp.Identifier(this=self._normalize_identifier(table_name)),
                    )
                if node.db:
                    node.set(
                        "db", exp.Identifier(this=self._normalize_identifier(node.db))
                    )
            elif isinstance(node, exp.Column):
                column_name = node.name
                if column_name:
                    node.set(
                        "this",
                        exp.Identifier(this=self._normalize_identifier(column_name)),
                    )
                if node.table:
                    node.set(
                        "table",
                        exp.Identifier(this=self._normalize_identifier(node.table)),
                    )
            return node

        transformed = expression.transform(transform)
        return transformed.sql(dialect="mysql")


class GenericExecutionEngine:
    """Dialect-invariant execution engine using Ibis backends."""

    def __init__(
        self,
        db_path: Union[str, Path],
        backend: str = "duckdb",
        load_data: bool = True,
        namespace: str = "",
        dataset_name: str = "",
        source_db_type: str = "sqlite",
    ):
        """
        Initialize the execution engine with a specific backend.

        Args:
            db_path: Path to the database file or database name
            backend: Ibis backend to use (default: "duckdb")
                     Options: "duckdb", "sqlite",
                     "datafusion", "pyspark", "bigquery", "snowflake",
                     "postgres", "mysql"
            load_data: Whether to load data into the backend
            namespace: Optional namespace to avoid collisions between dialects
            dataset_name: Dataset identifier (bird, beaver, archer, etc.)
            source_db_type: Source database type (sqlite, mysql, postgres)
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.backend = backend
        self.namespace = namespace
        self.dataset_name = dataset_name
        self.source_db_type = source_db_type
        self._data_loaded = False
        self.conn = None
        self._connect(load_data=load_data)
        # Set up dialect handler
        self.dialect = self._get_dialect_handler(backend)

    def close_connection(self):
        """Close the connection and clean up resources."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        import gc

        gc.collect()

    def _connect(self, load_data: bool) -> None:
        """Create a fresh backend connection, optionally loading data."""
        self.conn = get_ibis_connection(
            self.db_path,
            backend=self.backend,
            load_data=load_data,
            namespace=self.namespace,
            source_db_type=self.source_db_type,
        )
        if load_data:
            self._data_loaded = True

    def ensure_connection(self, load_data: bool = False) -> None:
        """Re-open the backend connection if it was previously closed."""
        if self.conn is not None:
            return

        # If connection was closed and load_data is requested, we must reload
        # Otherwise the reconnected backend will have no tables
        self._connect(load_data=load_data)

    def _get_dialect_handler(self, backend: str) -> DialectHandler:
        """
        Get the appropriate dialect handler for the backend.

        No query rewriting - LLM must generate correct SQL per dialect instructions.

        Args:
            backend: Backend name

        Returns:
            DialectHandler instance (always StandardDialect - no rewriting)
        """
        return StandardDialect()

    def _execute_query_internal(self, rewritten_query: str) -> pd.DataFrame:
        """Execute query and handle errors."""
        try:
            return self.conn.sql(rewritten_query).execute()
        except Exception as e:
            # print(f"Error executing query: {rewritten_query}\nException: {e}")
            # Improve error message for empty result sets
            error_msg = str(e)
            if "Failed to infer types for columns" in error_msg:
                raise ValueError(
                    f"Query returned an empty result set, cannot infer types for computed columns. "
                    f"Original error: {error_msg}"
                ) from e
            raise

    def execute(self, query: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Execute a SQL query using the Ibis connection.

        Automatically rewrites queries for backend-specific requirements:
        - BigQuery: Adds dataset prefix to table names
        - Snowflake: No rewriting (transpiler handles identifier quoting)
        - Others: No rewriting needed

        Args:
            query: SQL query string (standard SQL format)
            use_cache: Whether to use disk cache for query results

        Returns:
            pandas DataFrame with query results

        Example:
            >>> engine = GenericExecutionEngine("db.sqlite", backend="duckdb")
            >>> result = engine.execute("SELECT * FROM table WHERE x > 10")
            >>> print(result)
        """
        # Generate cache key from query, backend, db_path, and dataset_name
        cache_key = hashlib.sha256(
            f"{query}|{self.backend}|{self.db_path}|{self.dataset_name}".encode()
        ).hexdigest()

        # Check cache
        if use_cache:
            cached_result = execution_cache.get(cache_key)
            if cached_result is not None:
                # Return cached DataFrame
                return cached_result

        # Rewrite query using dialect handler
        tables = self.list_tables()
        rewritten_query = self.dialect.rewrite_query(query, tables)

        # Use Ibis sql() method to execute the query with timeout

        result = self._execute_query_internal(rewritten_query)

        # Store in cache
        if use_cache:
            execution_cache.set(cache_key, result)

        return result

    def list_tables(self) -> List[str]:
        """List all available tables in the backend."""
        return self.conn.list_tables()

    def get_table(self, table_name: str):
        """
        Get an Ibis table reference.

        Args:
            table_name: Name of the table

        Returns:
            Ibis table expression
        """
        return self.conn.table(table_name)

    def close(self):
        """Close the connection (if applicable for the backend)."""
        if self.conn is None:
            import gc

            self.conn = None
            gc.collect()
            return
        # For truly in-memory backends, drop all tables to release memory
        # DuckDB is excluded because it uses persistent files in this codebase
        if self.backend in ("datafusion",):
            try:
                for table in self.list_tables():
                    self.conn.drop_table(table)
            except Exception:
                pass  # Ignore errors if tables can't be dropped

        # Some backends may need explicit cleanup
        if hasattr(self.conn, "close"):
            try:
                self.conn.close()
            except Exception:
                pass  # Ignore cleanup errors
        # Force cleanup of references
        self.conn = None
        import gc

        gc.collect()

    def __enter__(self):
        """Context manager entry.

        WARNING: Avoid using engines from ExecutionEngineFactory as context managers.
        The factory manages the connection lifecycle. Using 'with' will close
        connections that may be reused from the pool.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ExecutionEngineFactory:
    """Factory for creating execution engines with connection pooling.

    Connection pooling is process-local to prevent sharing database connections
    across multiprocessing workers (which causes segfaults with MySQL/PostgreSQL).
    """

    _engine_pool: dict = {}
    _pool_pid: int = os.getpid()

    @classmethod
    def _ensure_pool_is_valid(cls) -> None:
        """Clear the pool if we're in a different process.

        When multiprocessing spawns/forks a child process, it inherits the
        parent's _engine_pool. Database connections are not safe to share
        across processes, so we clear the pool in child processes to force
        creation of new connections.

        This prevents segfaults with MySQL/PostgreSQL connections.
        """
        current_pid = os.getpid()
        if current_pid != cls._pool_pid:
            # We're in a child process - clear parent's connections
            cls._engine_pool.clear()
            cls._pool_pid = current_pid

    @classmethod
    def create(
        cls,
        db_path: Union[str, Path],
        engine_type: Literal["sql", "substrait"] = "sql",
        backend: str = "duckdb",
        load_data: bool = True,
        dataset_name: str = "",
        source_db_type: str = "sqlite",
    ):
        """
        Create an execution engine of the specified type.

        Engines are pooled and reused based on (db_path, engine_type, backend, dataset_name).
        This avoids repeatedly loading data into backends like DuckDB.

        NOTE: MySQL and PostgreSQL engines are NOT pooled due to multiprocessing
        safety concerns. Fresh connections are created for each call when using
        these backends.

        Args:
            db_path: Path to the database file or database name
            engine_type: Type of engine - "sql" or "substrait"
            backend: Backend to use (only for SQL engine)
            load_data: Whether to load data into the backend
            dataset_name: Dataset identifier (bird, beaver, archer, etc.)
            source_db_type: Source database type (sqlite, mysql, postgres)

        Returns:
            Appropriate execution engine instance

        Example:
            >>> # SQL engine with SQLite source
            >>> engine = ExecutionEngineFactory.create(
            ...     "db.sqlite", engine_type="sql", backend="duckdb",
            ...     dataset_name="bird", source_db_type="sqlite"
            ... )
            >>> # SQL engine with MySQL source
            >>> engine = ExecutionEngineFactory.create(
            ...     "minidev_mysql_card_games", engine_type="sql", backend="mysql",
            ...     dataset_name="bird_minidev", source_db_type="mysql"
            ... )
        """
        # Ensure pool is valid for current process (clears pool in child processes)
        cls._ensure_pool_is_valid()

        # Disable pooling for MySQL/PostgreSQL to prevent multiprocessing segfaults
        # These backends use C-level connection state that isn't fork-safe
        disable_pooling = backend in ("mysql", "postgres", "postgresql", "sqlite")

        # Create cache key including dataset_name and source_db_type to avoid collisions
        db_path_str = str(
            Path(db_path).resolve() if isinstance(db_path, Path) else db_path
        )
        cache_key = (
            db_path_str,
            engine_type,
            backend,
            load_data,
            dataset_name,
            source_db_type,
        )

        # Check if engine exists in pool (skip for MySQL/PostgreSQL)
        if not disable_pooling and cache_key in cls._engine_pool:
            engine = cls._engine_pool[cache_key]
            # Only reconnect for SQL engines, not Substrait (which manages its own connections)
            if isinstance(engine, GenericExecutionEngine) and not isinstance(
                engine, SubstraitExecutionEngine
            ):
                engine.ensure_connection(load_data=load_data)
            return engine

        # Create namespace combining engine_type and dataset_name to avoid file collisions
        namespace = f"{engine_type}_{dataset_name}" if dataset_name else engine_type

        # Create new engine
        if engine_type == "sql":
            engine = GenericExecutionEngine(
                db_path,
                backend=backend,
                load_data=load_data,
                namespace=namespace,
                dataset_name=dataset_name,
                source_db_type=source_db_type,
            )
        elif engine_type == "substrait":
            engine = SubstraitExecutionEngine(
                db_path, load_data=load_data, dataset_name=dataset_name
            )
        else:
            raise ValueError(
                f"Unknown engine_type: {engine_type}. Must be 'sql' or 'substrait'"
            )

        # Store in pool (skip for MySQL/PostgreSQL - they're not pooled)
        if not disable_pooling:
            cls._engine_pool[cache_key] = engine

        return engine

    @classmethod
    def clear_pool(cls):
        """Close all pooled connections and clear the pool."""
        for engine in cls._engine_pool.values():
            try:
                engine.close_connection()
            except Exception:
                pass
        cls._engine_pool.clear()


class SubstraitExecutionEngine(GenericExecutionEngine):
    """Execution engine for Substrait plans using DuckDB backend."""

    def __init__(
        self, db_path: Union[str, Path], load_data: bool = True, dataset_name: str = ""
    ):
        """
        Initialize the Substrait execution engine.

        Note: This engine uses the transpilers package's execute_substrait_duckdb_sqlite
        function for execution, which handles all DuckDB connection setup and extension loading.

        Args:
            db_path: Path to the SQLite database file
            load_data: Ignored for Substrait engine (data is loaded on-demand)
            dataset_name: Dataset identifier (bird, beaver, archer, etc.)

        Raises:
            EnvironmentError: If DUCKDB_SUBSTRAIT_EXTENSION_PATH not set
        """
        # Check for required environment variable
        extension_path = os.getenv("DUCKDB_SUBSTRAIT_EXTENSION_PATH")
        if not extension_path:
            raise EnvironmentError(
                "DUCKDB_SUBSTRAIT_EXTENSION_PATH environment variable "
                "not set. Substrait execution requires DuckDB Substrait "
                "extension. Set it in your .env file."
            )

        # Store the database path
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.backend = "duckdb"
        self.dataset_name = dataset_name
        self.namespace = (
            "substrait"  # Namespace for DuckDB file caching (unused for Substrait)
        )
        self.dialect_handler = StandardDialect()
        self.conn = None  # We don't maintain a persistent connection
        self._data_loaded = False  # Match parent class interface (unused for Substrait)
        # load_data parameter is ignored - data is loaded on-demand by execute_substrait_duckdb_sqlite

    def execute_substrait(self, plan: Union[str, bytes]) -> pd.DataFrame:
        """
        Execute a Substrait plan against the database.

        Uses the transpilers package's execute_substrait_duckdb_sqlite function.

        Args:
            plan: Substrait plan as:
                  - base64-encoded string
                  - raw bytes (protobuf serialized Plan)

        Returns:
            pandas DataFrame with query results

        Raises:
            RuntimeError: If Substrait execution fails

        Example:
            >>> engine = SubstraitExecutionEngine("db.sqlite")
            >>> result = engine.execute_substrait(plan_b64)
            >>> print(result)
        """
        from transpilers.utils.query_execution import execute_substrait_duckdb_sqlite

        # Convert base64 string to bytes if needed
        if isinstance(plan, str):
            plan_bytes = base64.b64decode(plan)
        else:
            plan_bytes = plan

        # Use the transpilers package's function
        try:
            return execute_substrait_duckdb_sqlite(str(self.db_path), plan_bytes)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute Substrait plan: {e}\n"
                "Ensure DUCKDB_SUBSTRAIT_EXTENSION_PATH is set correctly "
                "and points to the DuckDB Substrait extension."
            )

    def close_connection(self):
        """Close connection (no-op for SubstraitExecutionEngine)."""
        pass  # We don't maintain a persistent connection
