"""
Utility for creating Ibis connections to SQLite databases across multiple backends.

Supports loading SQLite data into various backends including:
- Local: DuckDB, SQLite, DataFusion, PySpark
- Cloud: BigQuery, Snowflake, PostgreSQL, MySQL

Uses dlt (data load tool) for efficient data transfer to remote databases.

Note: Databricks support is currently disabled due to incompatibility between
dlt (requires databricks-sql-connector v3) and Ibis (requires v4).
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import dlt
import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv
from ibis import BaseBackend

from polysql.evaluation.utils.cache_paths import cache_subdirs, migrate_legacy_caches

# Load environment variables from .env file
load_dotenv()
migrate_legacy_caches()

TIMEOUT = 300  # 5 minutes for cloud databases like Snowflake

# Backends supported by dlt
# Note: databricks disabled due to incompatibility between dlt and Ibis connector versions
DLT_SUPPORTED_BACKENDS = {"duckdb", "postgres", "bigquery", "snowflake", "mysql"}

# Directory for schema cache files
SCHEMA_CACHE_DIR = cache_subdirs()["schemas"]
# Only cache schemas for truly persistent remote backends
# PySpark requires connection to load data, even though it uses warehouse
_CACHEABLE_BACKENDS = {"bigquery", "snowflake"}

# ============================================================================
# Helper Functions (RE-INTRODUCED)
# ============================================================================


def _clean_dataframe_for_load(
    df: pd.DataFrame, backend: str = "generic"
) -> pd.DataFrame:
    """
    Apply standard data cleaning to a Pandas DataFrame before loading
    into an Ibis backend.

    - Converts datetime columns to ISO 8601 strings.
    - Converts timedelta columns to total seconds (float).
    - Handles all-NULL columns (backend-specific handling).
    - Converts empty strings to None in object columns (for DataFusion/PyArrow).
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            # Convert to ISO 8601 string, handling NaT (null)
            df_copy[col] = df_copy[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )
        elif pd.api.types.is_timedelta64_dtype(df_copy[col]):
            # Convert timedelta to total seconds, handling NaT (null)
            df_copy[col] = df_copy[col].apply(
                lambda x: x.total_seconds() if pd.notna(x) else None
            )
        elif pd.api.types.is_object_dtype(df_copy[col]):
            # Replace empty strings with None to avoid PyArrow conversion issues
            # DataFusion/PyArrow cannot infer types for columns with empty strings
            df_copy[col] = df_copy[col].replace("", None)

    # Identify all-NULL columns
    all_null_columns = [col for col in df_copy.columns if df_copy[col].isnull().all()]

    if all_null_columns:
        if backend == "pyspark":
            # PySpark/Parquet doesn't support VOID types - drop all-NULL columns
            print(
                f"Dropping {len(all_null_columns)} all-NULL columns for PySpark: {all_null_columns}"
            )
            df_copy = df_copy.drop(columns=all_null_columns)
        else:
            # Other backends: cast to 'object' (becomes string/TEXT)
            for col in all_null_columns:
                df_copy[col] = df_copy[col].astype("object")

    return df_copy


def _dataframe_to_arrow_table_no_metadata(df: pd.DataFrame) -> pa.Table:
    """
    Convert pandas DataFrame to PyArrow Table without index or metadata.

    Removing pandas metadata avoids DataFusion optimizer schema-mismatch errors.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    stripped = table.replace_schema_metadata({})

    # Fail fast if column order changes
    assert stripped.schema.names == list(df.columns), (
        "Column names changed during pandas->arrow conversion: "
        f"{stripped.schema.names} vs {list(df.columns)}"
    )
    return stripped


def _get_schema_cache_path(db_path: Path, backend: str) -> Path:
    """Return cache file path for a backend/db combination."""
    cache_key_source = f"{backend}::{Path(db_path).resolve()}"
    cache_key = hashlib.md5(cache_key_source.encode()).hexdigest()
    return SCHEMA_CACHE_DIR / f"{backend}_{cache_key}.sql"


def _load_cached_schema(db_path: Path, backend: str) -> Optional[str]:
    """Return cached schema text if available."""
    if backend not in _CACHEABLE_BACKENDS:
        return None

    cache_path = _get_schema_cache_path(db_path, backend)
    if cache_path.exists():
        schema_text = cache_path.read_text(encoding="utf-8")

        # Guard against empty/invalid cached schemas
        if not schema_text.strip():
            cache_path.unlink(missing_ok=True)
            return None

        if backend == "bigquery":
            # Invalidate legacy caches that lacked dataset-qualified table names
            # e.g., CREATE TABLE `patient` (...) instead of `dataset.patient`
            has_dataset_prefix = re.search(r"CREATE TABLE `[^`]*\.[^`]*`", schema_text)
            if not has_dataset_prefix:
                return None

        return schema_text
    return None


def _write_schema_cache(db_path: Path, backend: str, schema_text: str) -> None:
    """Persist schema text to cache."""
    if backend not in _CACHEABLE_BACKENDS:
        return

    cache_path = _get_schema_cache_path(db_path, backend)
    cache_path.write_text(schema_text, encoding="utf-8")


def sanitize_column_names(columns: List[str], backend: str) -> List[str]:
    """
    Sanitize column names based on backend-specific requirements.
    - BigQuery: Allows letters, numbers, underscores. Must start with letter/underscore.
    - PostgreSQL: Folds to lowercase. Best to be explicit.
    - Snowflake/DuckDB/SQLite: Preserve original names (support quoted identifiers).
    - Unknown backends: Preserve original names (conservative approach).
    """
    # Known backends that need sanitization
    sanitize_backends = {"bigquery", "postgres"}

    # Preserve original column names for unknown backends or those that support quoted identifiers
    if backend not in sanitize_backends:
        return columns

    sanitized_columns = []
    for col in columns:
        # General sanitization: replace spaces and special chars with underscores
        new_col = re.sub(r"\W+", "_", col)

        # Remove leading/trailing underscores
        new_col = new_col.strip("_")

        # Ensure not empty
        if not new_col:
            new_col = "unnamed_col"

        # BigQuery: ensure starts with letter or underscore
        if backend == "bigquery":
            if not re.match(r"^[a-zA-Z_]", new_col):
                new_col = f"_{new_col}"

        # PostgreSQL: fold to lowercase (matches Postgres default behavior)
        if backend == "postgres":
            new_col = new_col.lower()

        sanitized_columns.append(new_col)
    return sanitized_columns


# ============================================================================
# DLT-based Data Loading
# ============================================================================


def _create_sqlite_dlt_source(
    sqlite_conn: BaseBackend,
    table_names: List[str],
    write_disposition: Literal["replace", "append", "merge"] = "replace",
):
    """Create a dlt source that yields tables from SQLite."""

    @dlt.source
    def sqlite_source():
        for table_name in table_names:
            yield dlt.resource(
                sqlite_conn.table(table_name).execute().to_dict("records"),
                name=table_name,
                write_disposition=write_disposition,
            )

    return sqlite_source


def _create_ibis_dlt_source(
    source_conn: BaseBackend,
    table_names: List[str],
    write_disposition: Literal["replace", "append", "merge"] = "replace",
):
    """
    Create a dlt source that yields tables from any Ibis backend connection.

    This works with SQLite, MySQL, PostgreSQL, or any other Ibis backend.

    Args:
        source_conn: Ibis connection to source database
        table_names: List of table names to load
        write_disposition: How to write data (replace, append, merge)

    Returns:
        dlt source function
    """

    @dlt.source
    def ibis_source():
        for table_name in table_names:
            yield dlt.resource(
                source_conn.table(table_name).execute().to_dict("records"),
                name=table_name,
                write_disposition=write_disposition,
            )

    return ibis_source


def _normalize_table_names_for_matching(names: List[str]) -> set:
    """Normalize a list of table names for cross-database matching."""
    return {name.lower().replace("_", "") for name in names}


def _check_sql_database_exists(
    connection_func,
    conn_kwargs: dict,
    required_tables: List[str],
    database_name: str,
) -> Optional[BaseBackend]:
    """
    Check if a SQL database exists with all required tables.

    Args:
        connection_func: Function to create connection (e.g., ibis.postgres.connect)
        conn_kwargs: Connection parameters as dict
        required_tables: List of required table names
        database_name: Name of the database for logging

    Returns:
        Connection object if database exists with all tables, None otherwise
    """
    try:
        test_conn = connection_func(**conn_kwargs)

        # Normalize table names for matching
        existing_tables = _normalize_table_names_for_matching(test_conn.list_tables())
        required_normalized = _normalize_table_names_for_matching(required_tables)

        if required_normalized.issubset(existing_tables):
            # print(
            #     f"Reusing existing {database_name} database with {len(required_tables)} tables"
            # )
            return test_conn
        else:
            missing = required_normalized - existing_tables
            print(
                f"{database_name} database exists but missing tables: {missing}. Will reload."
            )
            return None
    except Exception:
        print(f"{database_name} database not found. Will create.")
        return None


def _load_data_with_dlt(
    sqlite_conn: BaseBackend,
    table_names: List[str],
    backend: str,
    dataset_name: str,
    write_disposition: Literal["replace", "append", "merge"] = "replace",
) -> None:
    """
    Load data from SQLite to supported backends using dlt.

    Args:
        sqlite_conn: SQLite Ibis connection
        table_names: List of table names to load
        backend: Target backend name
        dataset_name: Dataset/schema name for the target
        write_disposition: How to write data (replace, append, merge)
    """
    import logging
    import signal
    from contextlib import contextmanager

    @contextmanager
    def timeout_context(seconds: int):
        """Context manager that raises TimeoutError after specified seconds."""

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        # Set signal handler and alarm
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            # Restore original handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

    # Create pipeline with appropriate destination
    destination = backend
    if backend == "snowflake":
        destination = dlt.destinations.snowflake(
            enable_dataset_name_normalization=False
        )

    pipeline = dlt.pipeline(
        pipeline_name=f"sqlite_to_{backend}_{dataset_name}",
        destination=destination,
        dataset_name=dataset_name,
    )

    # Suppress dlt warnings
    logging.getLogger("dlt").setLevel(logging.ERROR)

    # Run the pipeline with timeout
    print(f"Running dlt pipeline for {backend} with {len(table_names)} tables...")
    sqlite_source = _create_sqlite_dlt_source(
        sqlite_conn, table_names, write_disposition
    )
    try:
        with timeout_context(TIMEOUT):
            pipeline.run(sqlite_source())
        print("dlt pipeline completed successfully!")
    except TimeoutError as e:
        print(f"dlt pipeline timed out after {TIMEOUT} seconds!")
        raise RuntimeError(f"dlt pipeline loading timed out: {e}")


def _create_postgres_database(
    database_name: str, host: str, port: int, user: str, password: str
) -> None:
    """Create a PostgreSQL database if it doesn't exist."""
    import psycopg2

    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",  # Connect to default postgres database
        )
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE {database_name}")
        cursor.close()
        connection.close()
        print(f"Created PostgreSQL database '{database_name}'")
    except psycopg2.errors.DuplicateDatabase:
        print(f"PostgreSQL database '{database_name}' already exists")
    except Exception as e:
        print(f"Note: {e}")


def _create_mysql_database(
    database_name: str, host: str, port: int, user: str, password: Optional[str]
) -> None:
    """Create a MySQL database if it doesn't exist."""
    import pymysql

    try:
        create_conn_kwargs = {
            "host": host,
            "port": port,
            "user": user,
        }
        if password is not None:
            create_conn_kwargs["password"] = password

        connection = pymysql.connect(**create_conn_kwargs)
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        cursor.close()
        connection.close()
        print(f"Created MySQL database '{database_name}'")
    except Exception as e:
        print(f"Error creating MySQL database: {e}")
        raise


def _ensure_mysql_dlt_metadata_columns_longtext(
    database_name: str,
    host: str,
    port: int,
    user: str,
    password: Optional[str],
) -> None:
    """Ensure MySQL dlt metadata columns can store large serialized schemas."""
    import pymysql

    conn_kwargs = {
        "host": host,
        "port": port,
        "user": user,
        "database": database_name,
        "charset": "utf8mb4",
        "autocommit": True,
    }
    if password is not None:
        conn_kwargs["password"] = password

    try:
        connection = pymysql.connect(**conn_kwargs)
    except Exception:
        return

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = '_dlt_version'
                """,
                (database_name,),
            )
            exists = cursor.fetchone()[0] > 0

            if not exists:
                return

            for column in ("schema", "schema_name", "version_hash"):
                try:
                    cursor.execute(
                        f"ALTER TABLE `{database_name}`.`_dlt_version` "
                        f"MODIFY COLUMN `{column}` LONGTEXT"
                    )
                except Exception:
                    # Ignore if column already has sufficient size or does not exist
                    break
    finally:
        connection.close()


# ============================================================================
# Main Connection Function
# ============================================================================


def get_ibis_connection(
    db_path: Union[str, Path],
    backend: str = "duckdb",
    load_data: bool = True,
    write_disposition: Literal["replace", "append", "merge"] = "replace",
    namespace: str = "",
    source_db_type: Optional[str] = None,
) -> BaseBackend:
    """
    Create an Ibis connection to a database and load its data into a target backend.

    Args:
        db_path: Path to the database file or database name
        backend: Target backend for data loading
                 Supported: duckdb, sqlite, datafusion, pyspark,
                           bigquery, snowflake, postgres, mysql
        load_data: Whether to load data into the backend
                   If False, only the connection is created
        write_disposition: Data loading strategy for dlt-supported backends
                          - "replace": Full reload (default)
                          - "append": Incremental append
                          - "merge": Upsert based on primary key
        namespace: Optional namespace to avoid collisions between different dialects
                   (e.g., "sql" vs "substrait" when both use duckdb backend)
        source_db_type: Source database type (sqlite, mysql, postgres)
                       Used to determine if a native connector should be used

    Returns:
        Ibis connection object

    Raises:
        FileNotFoundError: If the database file doesn't exist
        ValueError: If an unsupported backend is specified

    Examples:
        >>> # Load SQLite data into DuckDB
        >>> conn = get_ibis_connection("data.sqlite", backend="duckdb")
        >>> tables = conn.list_tables()

        >>> # Connect to native MySQL database
        >>> conn = get_ibis_connection("minidev_mysql_card_games",
        ...                           backend="mysql", source_db_type="mysql")

        >>> # Connect to PostgreSQL with incremental loading
        >>> conn = get_ibis_connection("data.sqlite", backend="postgres",
        ...                           write_disposition="append")
    """
    db_path = Path(db_path) if isinstance(db_path, str) else db_path

    # Use connector architecture for all backends
    from polysql.evaluation.backends.connectors import BackendConnectorFactory

    connector = BackendConnectorFactory.create(backend, db_path, source_db_type)
    return connector.connect(db_path, load_data, write_disposition, namespace)


def _normalize_name_for_matching(name: str) -> str:
    """Normalize table/column names for cross-database matching."""
    # Remove underscores and lowercase: CDSCode -> cdscode, cds_code -> cdscode
    return name.lower().replace("_", "")


def _extract_foreign_keys_from_sqlite(
    sqlite_path: Path, target_tables: list, target_columns: dict
) -> dict:
    """
    Extract FK relationships from SQLite, mapped to target database names.

    Args:
        sqlite_path: Path to SQLite database
        target_tables: List of table names from target database
        target_columns: Dict mapping table names to their column lists

    Returns:
        Dict mapping target table name -> {target_col_name: {ref_table, ref_column}}
    """
    import sqlite3

    # Build normalized name mappings for reverse lookup
    # target_normalized -> target_actual
    table_name_map = {_normalize_name_for_matching(t): t for t in target_tables}

    column_name_maps = {}  # target_table -> {normalized -> actual}
    for table, cols in target_columns.items():
        column_name_maps[table] = {_normalize_name_for_matching(c): c for c in cols}

    fk_map = {}
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        for (sqlite_table,) in tables:
            # Find corresponding target table
            normalized_sqlite_table = _normalize_name_for_matching(sqlite_table)
            if normalized_sqlite_table not in table_name_map:
                continue
            target_table = table_name_map[normalized_sqlite_table]

            fks = cursor.execute(f"PRAGMA foreign_key_list({sqlite_table})").fetchall()
            if fks:
                table_fks = {}
                for fk in fks:
                    _, _, sqlite_ref_table, sqlite_from_col, sqlite_to_col, _, _, _ = fk

                    # Map SQLite column to target column
                    normalized_from = _normalize_name_for_matching(sqlite_from_col)
                    if normalized_from in column_name_maps.get(target_table, {}):
                        target_from_col = column_name_maps[target_table][
                            normalized_from
                        ]

                        # Map SQLite ref table to target ref table
                        normalized_ref = _normalize_name_for_matching(sqlite_ref_table)
                        if normalized_ref in table_name_map:
                            target_ref_table = table_name_map[normalized_ref]

                            table_fks[target_from_col] = {
                                "ref_table": target_ref_table,
                                "ref_column": sqlite_to_col,  # Keep original for display
                            }

                fk_map[target_table] = table_fks

        conn.close()
    except Exception:
        # If we can't extract FKs, return empty dict
        pass

    return fk_map


def _filter_dlt_columns_from_ddl(ddl: str) -> str:
    """
    Remove _dlt_* column definitions from CREATE TABLE DDL.

    Handles various DDL formats and cleans up trailing commas after removal.

    Args:
        ddl: CREATE TABLE statement with potential _dlt_* columns

    Returns:
        Cleaned DDL without _dlt_* columns
    """
    import re

    # Split DDL into lines
    lines = ddl.split("\n")
    filtered_lines = []

    for line in lines:
        # Skip lines that define _dlt_ columns
        # Pattern matches: column_name starting with _dlt_ followed by type definition
        # Examples:
        #   `_dlt_load_id` TEXT NOT NULL,
        #   "_dlt_id" VARCHAR NOT NULL,
        #   _dlt_load_id TEXT NOT NULL,
        if re.search(r'[`"]?_dlt_\w+[`"]?\s+\w+', line):
            continue
        filtered_lines.append(line)

    # Join lines back
    result = "\n".join(filtered_lines)

    # Fix trailing commas before closing parenthesis
    # Pattern: comma followed by optional whitespace and closing paren
    result = re.sub(r",(\s*\))", r"\1", result)

    return result


def _extract_ddl_from_mysql(conn, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from MySQL database.

    Args:
        conn: Ibis connection to MySQL database
        target_tables: List of table names from target database

    Returns:
        Dict mapping target table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    import re

    ddl_map = {}
    for table in target_tables:
        try:
            # Use SHOW CREATE TABLE to get actual MySQL DDL
            result = conn.raw_sql(f"SHOW CREATE TABLE `{table}`")
            row = result.fetchone()
            if row is None or len(row) < 2:
                raise ValueError(
                    f"SHOW CREATE TABLE returned no results for table {table}"
                )

            # row is (table_name, create_statement)
            ddl = row[1]

            # Clean up DDL to make it more consistent
            # Remove MySQL-specific options at end (ENGINE=InnoDB, AUTO_INCREMENT, etc.)
            ddl = re.sub(
                r"\)\s*(ENGINE|DEFAULT CHARSET|AUTO_INCREMENT|COLLATE|ROW_FORMAT)[^;]*",
                ")",
                ddl,
                flags=re.IGNORECASE,
            )

            # Ensure it ends with semicolon
            if not ddl.strip().endswith(";"):
                ddl = ddl.strip() + ";"

            # Filter out dlt metadata columns
            ddl = _filter_dlt_columns_from_ddl(ddl)

            ddl_map[table] = ddl

        except Exception as e:
            raise ValueError(f"Failed to extract DDL for table {table} from MySQL: {e}")

    return ddl_map


def _extract_ddl_from_postgres(conn, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from PostgreSQL database.

    Args:
        conn: Ibis connection to PostgreSQL database
        target_tables: List of table names from target database

    Returns:
        Dict mapping table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    ddl_map = {}
    for table in target_tables:
        try:
            # Query information_schema to build CREATE TABLE DDL
            query = f"""
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
            """
            result = conn.raw_sql(query)
            # BigQuery returns RowIterator, convert to list
            rows = list(result)

            if not rows:
                raise ValueError(f"No columns found for table {table}")

            # Build column definitions
            col_defs = []
            for col_name, data_type, max_length, is_nullable, col_default in rows:
                # Build column definition
                col_def = f'"{col_name}" '

                # Add data type with length if applicable
                if data_type.upper() in ("CHARACTER VARYING", "VARCHAR"):
                    if max_length:
                        col_def += f"VARCHAR({max_length})"
                    else:
                        col_def += "VARCHAR"
                elif data_type.upper() == "CHARACTER":
                    if max_length:
                        col_def += f"CHAR({max_length})"
                    else:
                        col_def += "CHAR"
                else:
                    col_def += data_type.upper()

                # Add NOT NULL constraint if applicable
                if is_nullable == "NO":
                    col_def += " NOT NULL"

                # Add DEFAULT clause if present
                if col_default:
                    col_def += f" DEFAULT {col_default}"

                col_defs.append(col_def)

            # Build CREATE TABLE statement
            cols = ",\n    ".join(col_defs)
            ddl = f'CREATE TABLE "{table}" (\n    {cols}\n);'

            # Filter out dlt metadata columns
            ddl = _filter_dlt_columns_from_ddl(ddl)

            ddl_map[table] = ddl

        except Exception as e:
            raise ValueError(
                f"Failed to extract DDL for table {table} from PostgreSQL: {e}"
            )

    return ddl_map


def _extract_ddl_from_sqlite(sqlite_path: Path, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from SQLite database.

    Args:
        sqlite_path: Path to SQLite database
        target_tables: List of table names from target database

    Returns:
        Dict mapping target table name -> DDL string with proper SQL types
    """
    import re
    import sqlite3

    # Build normalized name mapping
    table_name_map = {_normalize_name_for_matching(t): t for t in target_tables}

    ddl_map = {}
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall()

        for sqlite_table, ddl in tables:
            if ddl is None:
                continue

            # Find corresponding target table
            normalized_sqlite_table = _normalize_name_for_matching(sqlite_table)
            if normalized_sqlite_table not in table_name_map:
                continue
            target_table = table_name_map[normalized_sqlite_table]

            # Clean up DDL: remove SQLite-specific syntax
            # Replace table name with target table name (in case of case differences)
            # Pattern: CREATE TABLE "original_name" or CREATE TABLE original_name
            ddl = re.sub(
                rf'CREATE\s+TABLE\s+["`]?{re.escape(sqlite_table)}["`]?',
                f"CREATE TABLE {target_table}",
                ddl,
                flags=re.IGNORECASE,
            )

            # Filter out dlt metadata columns
            ddl = _filter_dlt_columns_from_ddl(ddl)

            ddl_map[target_table] = ddl

        conn.close()
    except Exception:
        # If we can't extract DDL, return empty dict
        pass

    return ddl_map


def _extract_ddl_from_duckdb(conn, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from DuckDB database.

    Args:
        conn: Ibis DuckDB connection
        target_tables: List of table names from target database

    Returns:
        Dict mapping table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    ddl_map = {}
    for table in target_tables:
        try:
            # Use PRAGMA table_info to reconstruct DDL with DuckDB-native types
            info = conn.raw_sql(f"PRAGMA table_info('{table}')").fetchall()
            if not info:
                raise ValueError(f"PRAGMA table_info returned no rows for {table}")

            col_defs = []
            for _, name, col_type, notnull, default, _ in info:
                # Skip dlt metadata columns
                if str(name).lower().startswith("_dlt_"):
                    continue

                col_def = f'"{name}" {col_type}'
                if notnull:
                    col_def += " NOT NULL"
                if default is not None:
                    col_def += f" DEFAULT {default}"
                col_defs.append(col_def)

            if not col_defs:
                raise ValueError(f"No non-metadata columns found for {table}")

            ddl_text = (
                f'CREATE TABLE "{table}" (\n    ' + ",\n    ".join(col_defs) + "\n);"
            )
            ddl_map[table] = ddl_text
        except Exception as exc:
            raise ValueError(
                f"Failed to extract DDL for table {table} from DuckDB: {exc}"
            )

    return ddl_map


def _extract_ddl_from_datafusion(conn, target_tables: list) -> dict:
    """
    Extract CREATE TABLE DDL statements from DataFusion by building from Ibis schema.

    DataFusion doesn't support SHOW CREATE TABLE, so we construct DDL from schema info.

    Args:
        conn: Ibis DataFusion connection
        target_tables: List of table names from target database

    Returns:
        Dict mapping table name -> DDL string with DataFusion SQL types
    """
    # Map Ibis type names to DataFusion SQL types
    ibis_to_datafusion = {
        "Int8": "TINYINT",
        "Int16": "SMALLINT",
        "Int32": "INT",
        "Int64": "BIGINT",
        "UInt8": "TINYINT UNSIGNED",
        "UInt16": "SMALLINT UNSIGNED",
        "UInt32": "INT UNSIGNED",
        "UInt64": "BIGINT UNSIGNED",
        "Float16": "FLOAT",
        "Float32": "FLOAT",
        "Float64": "DOUBLE",
        "Decimal": "DECIMAL",
        "String": "TEXT",
        "Binary": "BYTEA",
        "Boolean": "BOOLEAN",
        "Date": "DATE",
        "Time": "TIME",
        "Timestamp": "TIMESTAMP",
        "Interval": "INTERVAL",
        "UUID": "TEXT",
        "JSON": "TEXT",
        "GeoSpatial": "TEXT",
        "Array": "TEXT",
        "Map": "TEXT",
        "Struct": "TEXT",
    }

    ddl_map = {}
    for table in target_tables:
        schema = conn.table(table).schema()
        col_defs = []

        for col_name, ibis_type in schema.items():
            if str(col_name).lower().startswith("_dlt_"):
                continue

            type_class_name = type(ibis_type).__name__
            sql_type = ibis_to_datafusion.get(type_class_name, "TEXT")
            col_defs.append(f'"{col_name}" {sql_type}')

        if not col_defs:
            raise ValueError(f"No non-metadata columns found for {table}")

        ddl_text = f'CREATE TABLE "{table}" (\n    ' + ",\n    ".join(col_defs) + "\n);"
        ddl_map[table] = ddl_text

    return ddl_map


def _extract_ddl_from_snowflake(
    conn, target_tables: list, schema_name: Optional[str]
) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from Snowflake database.

    Args:
        conn: Ibis Snowflake connection
        target_tables: List of table names from target database
        schema_name: Schema name to query (required for Snowflake)

    Returns:
        Dict mapping table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    ddl_map = {}

    # Determine database and schema
    result = conn.raw_sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
    current_database, current_schema = result.fetchone()

    database = current_database
    schema = schema_name if schema_name is not None else current_schema
    assert schema is not None, "Snowflake schema is required for DDL extraction"

    for table in target_tables:
        try:
            query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM {database}.information_schema.columns
            WHERE UPPER(table_schema) = '{schema.upper()}'
              AND table_name = '{table.upper()}'
            ORDER BY ordinal_position
            """
            rows = conn.raw_sql(query).fetchall()

            if not rows:
                raise ValueError(f"No columns found for table {table}")

            col_defs = []
            for col_name, data_type, is_nullable in rows:
                if str(col_name).lower().startswith("_dlt_"):
                    continue

                col_def = f'"{col_name}" {data_type}'
                if is_nullable == "NO":
                    col_def += " NOT NULL"
                col_defs.append(col_def)

            if not col_defs:
                raise ValueError(f"No non-metadata columns found for {table}")

            table_identifier = f'"{database}"."{schema}"."{table}"'
            ddl_text = (
                f"CREATE TABLE {table_identifier} (\n    "
                + ",\n    ".join(col_defs)
                + "\n);"
            )
            ddl_map[table] = ddl_text
        except Exception as exc:
            raise ValueError(
                f"Failed to extract DDL for table {table} from Snowflake: {exc}"
            )

    return ddl_map


def _extract_ddl_from_bigquery(conn, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from BigQuery database.

    Args:
        conn: Ibis BigQuery connection
        target_tables: List of table names from target database

    Returns:
        Dict mapping table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    ddl_map = {}
    for table in target_tables:
        try:
            # Query INFORMATION_SCHEMA to get column information
            # BigQuery uses project.dataset.table naming
            project_id = conn.project_id
            dataset_id = conn.dataset_id

            query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
            """
            result = conn.raw_sql(query)
            # BigQuery returns RowIterator, convert to list
            rows = list(result)

            if not rows:
                raise ValueError(f"No columns found for table {table}")

            # Build column definitions
            col_defs = []
            for col_name, data_type, is_nullable in rows:
                # Skip dlt metadata columns
                if str(col_name).lower().startswith("_dlt_"):
                    continue

                col_def = f"`{col_name}` {data_type}"

                # Add NOT NULL constraint if applicable
                if is_nullable == "NO":
                    col_def += " NOT NULL"

                col_defs.append(col_def)

            if not col_defs:
                raise ValueError(f"No non-metadata columns found for {table}")

            # Build CREATE TABLE statement with dataset-qualified name
            table_identifier = f"{dataset_id}.{table}"
            ddl_text = (
                f"CREATE TABLE `{table_identifier}` (\n    "
                + ",\n    ".join(col_defs)
                + "\n);"
            )
            ddl_map[table] = ddl_text

        except Exception as exc:
            raise ValueError(
                f"Failed to extract DDL for table {table} from BigQuery: {exc}"
            )

    return ddl_map


def _extract_ddl_from_clickhouse(conn, target_tables: list) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from ClickHouse database.

    Args:
        conn: Ibis ClickHouse connection
        target_tables: List of table names from target database

    Returns:
        Dict mapping table name -> DDL string with proper SQL types
    """
    ddl_map = {}
    database = conn.current_database

    for table in target_tables:
        # Query system.columns to get column information
        query = f"""
        SELECT
            name,
            type,
            default_kind,
            default_expression
        FROM system.columns
        WHERE database = '{database}'
          AND table = '{table}'
        ORDER BY position
        """
        result = conn.raw_sql(query)
        # ClickHouse driver uses result_rows instead of fetchall()
        rows = result.result_rows

        if not rows:
            raise ValueError(f"No columns found for table {table}")

        col_defs = []
        for col_name, col_type, default_kind, default_expr in rows:
            if str(col_name).lower().startswith("_dlt_"):
                continue

            col_def = f"`{col_name}` {col_type}"
            if default_kind and default_expr:
                col_def += f" {default_kind} {default_expr}"
            col_defs.append(col_def)

        if not col_defs:
            raise ValueError(f"No non-metadata columns found for {table}")

        ddl_text = (
            f"CREATE TABLE `{database}`.`{table}` (\n    "
            + ",\n    ".join(col_defs)
            + "\n) ENGINE = MergeTree();"
        )
        ddl_map[table] = ddl_text

    return ddl_map


def _extract_ddl_from_databricks(
    conn, target_tables: list, catalog: str, schema: str
) -> dict:
    """
    Extract actual CREATE TABLE DDL statements from Databricks database.

    Args:
        conn: Ibis Databricks connection
        target_tables: List of table names from target database
        catalog: Databricks catalog name
        schema: Databricks schema name

    Returns:
        Dict mapping table name -> DDL string with proper SQL types

    Raises:
        ValueError: If DDL extraction fails for any table
    """
    ddl_map = {}

    for table in target_tables:
        try:
            # Query information_schema to get column information
            query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM {catalog}.information_schema.columns
            WHERE table_catalog = '{catalog}'
              AND table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
            """
            result = conn.raw_sql(query)
            rows = result.fetchall()

            if not rows:
                raise ValueError(f"No columns found for table {table}")

            col_defs = []
            for col_name, data_type, is_nullable in rows:
                if str(col_name).lower().startswith("_dlt_"):
                    continue

                col_def = f"`{col_name}` {data_type}"
                if is_nullable == "NO":
                    col_def += " NOT NULL"
                col_defs.append(col_def)

            if not col_defs:
                raise ValueError(f"No non-metadata columns found for {table}")

            table_identifier = f"`{catalog}`.`{schema}`.`{table}`"
            ddl_text = (
                f"CREATE TABLE {table_identifier} (\n    "
                + ",\n    ".join(col_defs)
                + "\n);"
            )
            ddl_map[table] = ddl_text
        except Exception as exc:
            raise ValueError(
                f"Failed to extract DDL for table {table} from Databricks: {exc}"
            )

    return ddl_map


def render_schema_from_tables(
    conn, sqlite_path: Optional[Path] = None, schema_name: Optional[str] = None
):
    """
    Render schema from database tables, filtering out dlt metadata.

    Args:
        conn: Ibis connection to the database
        sqlite_path: Optional path to original SQLite file for FK extraction
        schema_name: Optional schema name for Snowflake (when CURRENT_SCHEMA() may return None)
    """
    # First pass: collect actual table and column names from target database
    target_tables = []
    target_columns = {}

    # Special handling for Snowflake and Databricks to query tables
    backend_name = conn.name
    if backend_name == "databricks":
        # For Databricks, conn.list_tables() has a bug, use SHOW TABLES instead
        result = conn.raw_sql("SELECT current_catalog(), current_schema()")
        catalog, schema = result.fetchone()

        result = conn.raw_sql(f"SHOW TABLES IN {catalog}.{schema}")
        table_list = [
            row.tableName
            for row in result.fetchall()
            if not row.tableName.startswith("_dlt_")
        ]
    elif backend_name == "snowflake":
        # For Snowflake, conn.list_tables() doesn't properly filter by schema
        # Try provided schema, its normalized variant, and current schema
        from dlt.common.normalizers.naming.snake_case import NamingConvention

        result = conn.raw_sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        database, current_schema = result.fetchone()

        candidates = []
        if schema_name is not None:
            candidates.append(schema_name)
            norm = NamingConvention().normalize_identifier(schema_name)
            if norm != schema_name:
                candidates.append(norm)
        if current_schema:
            candidates.append(current_schema)

        tried = []
        table_list = []
        effective_schema = None
        for cand in candidates:
            if cand is None:
                continue
            tried.append(cand)
            query = f"""
            SELECT table_name
            FROM {database}.information_schema.tables
            WHERE UPPER(table_schema) = '{cand.upper()}'
            AND table_type = 'BASE TABLE'
            AND NOT table_name LIKE '_DLT_%'
            ORDER BY table_name
            """
            rows = conn.raw_sql(query).fetchall()
            if rows:
                table_list = [row[0] for row in rows]
                effective_schema = cand
                # Set session schema so conn.table() resolves correctly
                conn.raw_sql(f"USE SCHEMA {effective_schema}")
                break

        if not table_list:
            raise ValueError(
                f"No tables found in Snowflake schemas: {tried or 'none'}. "
                "Schema extraction would be empty."
            )
    else:
        # For other backends, use standard list_tables()
        table_list = conn.list_tables()

    # ClickHouse can host many imported datasets in one database.
    # When the connector sets _clickhouse_dataset_name, restrict tables to that prefix
    if backend_name == "clickhouse" and hasattr(conn, "_clickhouse_dataset_name"):
        dataset_prefix = f"{getattr(conn, '_clickhouse_dataset_name')}___"
        table_list = [t for t in table_list if t.startswith(dataset_prefix)]

    for table in table_list:
        # Filter out dlt metadata tables - check both prefix and infix patterns
        # dlt uses prefix (_dlt_*) for most backends, but infix (___dlt_*) for ClickHouse
        if (
            table.startswith("_dlt_")
            or table.startswith("_DLT_")
            or "_dlt_" in table.lower()
        ):
            continue
        target_tables.append(table)
        schema = conn.table(table).schema()
        target_columns[table] = [
            col for col in schema.keys() if not col.startswith("_dlt_")
        ]

    # Extract DDL based on backend type
    ddl_map = {}
    if not target_tables:
        raise ValueError(
            f"No tables found for backend {backend_name}; schema extraction would be empty"
        )
    if backend_name == "mysql":
        ddl_map = _extract_ddl_from_mysql(conn, target_tables)
    elif backend_name == "postgres":
        ddl_map = _extract_ddl_from_postgres(conn, target_tables)
    elif backend_name == "duckdb":
        ddl_map = _extract_ddl_from_duckdb(conn, target_tables)
    elif backend_name == "bigquery":
        ddl_map = _extract_ddl_from_bigquery(conn, target_tables)
    elif backend_name == "snowflake":
        ddl_map = _extract_ddl_from_snowflake(conn, target_tables, schema_name)
    elif backend_name == "datafusion":
        ddl_map = _extract_ddl_from_datafusion(conn, target_tables)
    elif backend_name == "clickhouse":
        ddl_map = _extract_ddl_from_clickhouse(conn, target_tables)
    elif backend_name == "databricks":
        # Extract catalog and schema from connection
        # Use raw SQL to get current catalog and schema
        result = conn.raw_sql("SELECT current_catalog(), current_schema()")
        catalog, schema = result.fetchone()
        ddl_map = _extract_ddl_from_databricks(conn, target_tables, catalog, schema)
    elif sqlite_path and sqlite_path.exists():
        ddl_map = _extract_ddl_from_sqlite(sqlite_path, target_tables)
    else:
        raise ValueError(
            f"Cannot extract DDL for backend {backend_name}: "
            f"no sqlite_path provided and backend does not support native DDL extraction"
        )

    # Render schema - all tables must have DDL
    create_blocks = []
    for table in target_tables:
        if table not in ddl_map:
            raise ValueError(
                f"DDL extraction failed for table {table} on backend {backend_name}"
            )
        create_blocks.append(ddl_map[table])

    return "\n\n".join(create_blocks)


def _ensure_schema_cache(
    conn: BaseBackend, db_path: Path, backend: str, schema_name: Optional[str] = None
) -> None:
    """Populate schema cache for supported backends if missing."""
    if backend not in _CACHEABLE_BACKENDS:
        return

    cache_path = _get_schema_cache_path(db_path, backend)
    if cache_path.exists():
        return

    schema_text = render_schema_from_tables(
        conn, sqlite_path=db_path, schema_name=schema_name
    )
    _write_schema_cache(db_path, backend, schema_text)


def get_snowflake_names(
    db_path: Path, dataset_name: str, db_id: str = ""
) -> tuple[str, str, str, str]:
    """
    Compute Snowflake database and schema names with dlt normalization.

    Args:
        db_path: Path to source database (file path for SQLite, db name for MySQL/PostgreSQL)
        dataset_name: Dataset name (e.g., "bird_mini_dev_sqlite")
        db_id: Optional database ID from dataset (e.g., "california_schools")
                If provided, used as schema name. Otherwise extracted from db_path.

    Returns:
        Tuple of (database_name, schema_name, effective_database_name, effective_schema_name)
        where effective_* are the normalized uppercase versions that Snowflake actually uses
    """
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    # Extract schema name: use db_id if provided, otherwise extract from path
    if db_id:
        # Use the original db_id from dataset (works for all source types)
        db_name = db_id
    else:
        # Fallback: extract from file path (for SQLite)
        # e.g., data/MINIDEV/dev_databases/california_schools/california_schools.sqlite
        # -> california_schools
        db_name = db_path.parent.name

    database_name = dataset_name.upper()
    schema_name = db_name.upper()

    # dlt normalizes even with enable_dataset_name_normalization=False
    naming = NamingConvention()
    normalized_database_name = naming.normalize_identifier(database_name)
    normalized_schema_name = naming.normalize_identifier(schema_name)
    # Snowflake stores identifiers in uppercase
    effective_database_name = normalized_database_name.upper()
    effective_schema_name = normalized_schema_name.upper()

    return database_name, schema_name, effective_database_name, effective_schema_name


def schema_getter(instance, gen_type):
    """Get schema representation for LLM prompt, with FK information."""
    db_path = Path(instance["db_path"])
    dataset_meta = instance.get("_dataset_config", {})
    source_db_type = dataset_meta.get("source_db_type", "sqlite")
    dataset_name = dataset_meta.get("name", "")
    db_id = dataset_meta.get("db_id", "")

    backend = gen_type
    if backend.startswith("ibis-"):
        backend = backend[len("ibis-") :]
    elif backend.startswith("sqlite-"):
        # For sqlite-{dialect} pattern (e.g., sqlite-postgres), extract target dialect
        # The model should see the converted target schema (e.g., PostgreSQL schema)
        backend = backend.split("-", 1)[1]

    # For substrait, use DuckDB's schema representation
    if backend == "substrait":
        backend = "duckdb"

    cached_schema = _load_cached_schema(db_path, backend)
    if cached_schema is not None:
        if not cached_schema.strip():
            raise ValueError(
                f"Cached schema for backend {backend} at {db_path} is empty; cache invalid"
            )
        return cached_schema

    # Create namespace for cross-backend data loading
    namespace = f"sql_{dataset_name}" if dataset_name else "sql"

    # Get connection to the target backend; this may trigger data loading and caching.
    conn = get_ibis_connection(
        db_path, backend=backend, source_db_type=source_db_type, namespace=namespace
    )

    if backend in _CACHEABLE_BACKENDS:
        cached_schema = _load_cached_schema(db_path, backend)
        if cached_schema is not None:
            return cached_schema

    # Get the target database path for schema extraction
    # For cross-dialect conversions, use the converted database path if available
    target_db_path = getattr(conn, "_converted_db_path", None)
    if target_db_path is None:
        # Native evaluation or native SQLite: use db_path directly
        target_db_path = db_path

    # Render schema with FK information from target database
    # For Snowflake, we need to compute the schema name to pass it
    schema_name = None
    if backend == "snowflake":
        # Use shared helper to compute Snowflake names consistently
        _, _, _, schema_name = get_snowflake_names(db_path, dataset_name, db_id)

    schema = render_schema_from_tables(
        conn, sqlite_path=target_db_path, schema_name=schema_name
    )
    if not schema.strip():
        raise ValueError(
            f"Schema extraction returned empty for backend {backend} at {target_db_path}"
        )
    if backend in _CACHEABLE_BACKENDS:
        _write_schema_cache(db_path, backend, schema)
    return schema


_IBIS_SCHEMA_CACHE: Dict[Tuple[str, str], str] = {}


def _list_tables_for_ibis(conn, schema_name: Optional[str] = None) -> List[str]:
    """List tables for Ibis schema rendering, filtering dlt metadata."""
    backend_name = conn.name

    if backend_name == "snowflake":
        # Reuse the Snowflake-specific logic used in render_schema_from_tables
        from dlt.common.normalizers.naming.snake_case import NamingConvention

        result = conn.raw_sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        database, current_schema = result.fetchone()

        candidates: List[str] = []
        if schema_name is not None:
            candidates.append(schema_name)
            norm = NamingConvention().normalize_identifier(schema_name)
            if norm != schema_name:
                candidates.append(norm)
        if current_schema:
            candidates.append(current_schema)

        tried: List[str] = []
        table_list: List[str] = []
        for cand in candidates:
            if cand is None:
                continue
            tried.append(cand)
            query = f"""
            SELECT table_name
            FROM {database}.information_schema.tables
            WHERE UPPER(table_schema) = '{cand.upper()}'
            AND table_type = 'BASE TABLE'
            AND NOT table_name LIKE '_DLT_%'
            ORDER BY table_name
            """
            rows = conn.raw_sql(query).fetchall()
            if rows:
                table_list = [row[0] for row in rows]
                # Set session schema so conn.table() resolves correctly
                conn.raw_sql(f'USE SCHEMA "{database}"."{cand}"')
                break

        if not table_list:
            raise ValueError(
                f"No Snowflake tables found in schemas {tried}; unable to render Ibis schema"
            )
        return table_list

    # Default path for other backends
    tables = [
        table
        for table in conn.list_tables()
        if not str(table).lower().startswith("_dlt_")
    ]
    if not tables:
        raise ValueError(f"No tables found for backend {backend_name}")
    return tables


def _render_ibis_schema_from_backend(conn, schema_name: Optional[str] = None) -> str:
    """Build Ibis table definitions from a live backend connection."""
    table_names = _list_tables_for_ibis(conn, schema_name=schema_name)
    ibis_defs: List[str] = []

    for table_name in table_names:
        table_obj = conn.table(table_name)
        schema = table_obj.schema()

        columns = {
            str(col_name): str(dtype).lstrip("!").replace("string", "str")
            for col_name, dtype in zip(schema.names, schema.types)
            if not str(col_name).lower().startswith("_dlt_")
        }

        if not columns:
            raise ValueError(f"Table {table_name} has no non-metadata columns")

        # Compact format: remove spaces to save tokens
        columns_str = (
            str(columns).replace("'", "'").replace(": ", ":").replace(", ", ",")
        )
        ibis_defs.append(
            f"{table_name}=ibis.table(name='{table_name}',schema={columns_str})"
        )

    return "import ibis\n" + "\n".join(ibis_defs)


def ibis_schema_getter(instance, gen_type=None):
    """Get Ibis schema representation for LLM prompt based on target backend.

    For cross-dialect runs the schema must reflect the converted target
    database, not the original SQLite source, otherwise generated code will not
    align with execution.

    For sqlite-{dialect} transpilation experiments, the model should see the
    converted target schema (e.g., PostgreSQL schema) even though it generates
    SQLite code that will be transpiled.
    """
    if gen_type is None:
        raise ValueError("gen_type is required to build Ibis schema")

    db_path = Path(instance["db_path"])
    dataset_meta = instance.get("_dataset_config", {})
    source_db_type = dataset_meta.get("source_db_type", "sqlite")
    dataset_name = dataset_meta.get("name", "")
    db_id = dataset_meta.get("db_id", "")

    backend = gen_type
    if backend.startswith("ibis-"):
        backend = backend[len("ibis-") :]
    elif backend.startswith("sqlite-"):
        # For sqlite-{dialect} pattern (e.g., sqlite-postgres), extract target dialect
        # The model should see the converted target schema (e.g., PostgreSQL schema)
        backend = backend.split("-", 1)[1]
    backend = backend.replace("-ss", "")
    if backend == "substrait":
        backend = "duckdb"

    cache_key = (str(db_path), backend)
    if cache_key in _IBIS_SCHEMA_CACHE:
        cached_schema = _IBIS_SCHEMA_CACHE[cache_key]
        if not cached_schema.strip():
            raise ValueError(
                f"Cached Ibis schema for backend {backend} at {db_path} is empty"
            )
        return cached_schema

    namespace = f"sql_{dataset_name}" if dataset_name else "sql"
    conn = get_ibis_connection(
        db_path, backend=backend, source_db_type=source_db_type, namespace=namespace
    )

    schema_name = None
    if backend == "snowflake":
        _, _, _, schema_name = get_snowflake_names(db_path, dataset_name, db_id)

    ibis_schema = _render_ibis_schema_from_backend(conn, schema_name=schema_name)
    if not ibis_schema.strip():
        raise ValueError(
            f"Ibis schema extraction returned empty for backend {backend} at {db_path}"
        )

    _IBIS_SCHEMA_CACHE[cache_key] = ibis_schema
    return ibis_schema
