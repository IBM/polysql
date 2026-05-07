"""Generic dialect metric for comparing query results across execution engines.

This module provides a simple metric that executes two queries (possibly in
different dialects) and compares their results using pandas DataFrame equality.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
from pydantic import BaseModel, Field
from sqlglot import transpile

from polysql.evaluation.backends.execution import ExecutionEngineFactory

# Query comparison timeout (10 seconds)
COMPARISON_TIMEOUT = 10

# ============================================================================
# Pydantic Models
# ============================================================================


class QueryInput(BaseModel):
    """Input model for a query with its execution backend.

    Supports both SQL queries and Substrait plans (passed in query field).
    """

    query: str = Field(
        ..., description="SQL query string or Substrait plan (base64-encoded or JSON)"
    )
    backend: str = Field(
        default="duckdb", description="Backend to execute the query on"
    )
    query_type: Literal["sql", "substrait"] = Field(
        default="sql", description="Type of query (sql or substrait)"
    )

    class Config:
        """Pydantic config."""

        frozen = True


class MetricResult(BaseModel):
    """Result model for generic dialect metric comparison."""

    # Execution status
    query1_executed: bool = Field(
        ..., description="Whether query1 executed successfully"
    )
    query2_executed: bool = Field(
        ..., description="Whether query2 executed successfully"
    )
    both_executed: bool = Field(..., description="Whether both queries executed")

    # Comparison results
    results_equal: Optional[bool] = Field(None, description="Whether results are equal")

    # Error information
    query1_error: Optional[str] = Field(None, description="Error from query1 execution")
    query2_error: Optional[str] = Field(None, description="Error from query2 execution")

    # Result metadata
    query1_shape: Optional[Tuple[int, int]] = Field(
        None, description="Shape of query1 result (rows, cols)"
    )
    query2_shape: Optional[Tuple[int, int]] = Field(
        None, description="Shape of query2 result (rows, cols)"
    )

    # Actual query results
    query1_result: Optional[Any] = Field(
        None, description="Actual DataFrame result from query1 execution"
    )
    query2_result: Optional[Any] = Field(
        None, description="Actual DataFrame result from query2 execution"
    )

    # Mismatch details
    df1_values_on_mismatch: Optional[List[List[Any]]] = Field(
        None, description="DataFrame1 values as list of lists if results are not equal"
    )
    df2_values_on_mismatch: Optional[List[List[Any]]] = Field(
        None, description="DataFrame2 values as list of lists if results are not equal"
    )
    df1_dtypes_on_mismatch: Optional[Dict[str, str]] = Field(
        None, description="DataFrame1 dtypes as dict if results are not equal"
    )
    df2_dtypes_on_mismatch: Optional[Dict[str, str]] = Field(
        None, description="DataFrame2 dtypes as dict if results are not equal"
    )

    # Backend information
    query1_backend: str = Field(..., description="Backend used for query1")
    query2_backend: str = Field(..., description="Backend used for query2")

    class Config:
        """Pydantic config."""

        frozen = True
        arbitrary_types_allowed = True


class QueryTranspiler:
    """Handles SQL query transpilation with dialect-specific adjustments."""

    def transpile_query(
        self,
        query: str,
        read_dialect: str,
        write_dialect: str,
        target_backend: str = None,
    ) -> str:
        """


        Transpiles a SQL query from one dialect to another, with adjustments
        for target dialect specific requirements.

        Args:
            query: The SQL query string to transpile.
            read_dialect: The dialect of the input query (e.g., "sqlite").
            write_dialect: The target dialect (e.g., "postgres", "snowflake").

        Returns:
            The transpiled query string.


        """

        # Dialect-specific transpilation options
        # Note: identify=True quotes all identifiers

        # Determine whether to quote identifiers based on dialect and backend
        # - Postgres backend: Don't quote (allows case folding to lowercase)
        # - DataFusion backend with postgres dialect: Quote (DataFusion is case-sensitive)
        # - Snowflake: Quote (Ibis creates tables with original case, Snowflake uppercases unquoted)
        # - Others: Quote by default

        if target_backend == "postgres" and write_dialect == "postgres":
            # Real Postgres backend - don't quote to allow case folding
            identify = False
        elif write_dialect == "postgres" and target_backend != "postgres":
            # Other backend using postgres dialect (e.g., DataFusion) - quote to preserve case
            identify = True
        elif write_dialect == "snowflake":
            # Snowflake - quote to preserve case from SQLite
            identify = True
        else:
            # Default - quote identifiers
            identify = True

        transpile_kwargs = {
            "read": read_dialect,
            "write": write_dialect,
            "identify": identify,
        }

        return transpile(query, **transpile_kwargs)[0]


# ============================================================================
# Helper Functions
# ============================================================================


def _compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, order_matters: bool = True) -> bool:
    """Compare two DataFrames for equality.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        order_matters: If True, row order must match. If False, rows are sorted before comparison.
                      Default True for backward compatibility.

    Returns:
        True if DataFrames are equal, False otherwise
    """
    # Check shapes
    if df1.shape != df2.shape:
        return False

    # Sort columns by name (case-insensitive)
    df1_sorted = df1.reindex(sorted(df1.columns, key=str.lower), axis=1)
    df2_sorted = df2.reindex(sorted(df2.columns, key=str.lower), axis=1)

    # if df2 has more columns than df1, remove extra columns, keep only those in df1
    if df1_sorted.shape[1] < df2_sorted.shape[1]:
        df2_sorted = df2_sorted[df1_sorted.columns]

    # Now compare column by column
    try:
        for i in range(df1_sorted.shape[1]):
            s1 = df1_sorted.iloc[:, i]
            s2 = df2_sorted.iloc[:, i]

            if not pd.api.types.is_numeric_dtype(
                s1
            ) or not pd.api.types.is_numeric_dtype(s2):
                # For non-numeric, fall back to string comparison
                s1_str = s1.astype(str).str.strip()
                s2_str = s2.astype(str).str.strip()
                if not s1_str.equals(s2_str):
                    return False
            else:
                # For numeric, compare values (handles int64 vs uint32 etc)
                if not np.allclose(s1.values, s2.values, equal_nan=True):
                    return False
        return True
    except Exception:
        return False


# ============================================================================
# Metric Implementation
# ============================================================================


def generic_dialect_metric(
    query1: QueryInput,
    query2: QueryInput,
    db_path: Union[str, Path],
    load_data: bool = True,
    dataset_name: str = "",
    source_db_type: str = "sqlite",
) -> MetricResult:
    """
    Compare two queries by executing them and comparing results.

    Args:
        query1: First query with backend specification
        query2: Second query with backend specification
        db_path: Path to the SQLite database
        load_data: Whether to load data into backends (default: True)
        dataset_name: Dataset identifier (bird, beaver, archer, etc.)
        source_db_type: Source database type for connectors (sqlite, mysql, postgres)

    Returns:
        MetricResult with comparison results

    Example:
        >>> q1 = QueryInput(query="SELECT * FROM schools LIMIT 5", backend="duckdb")
        >>> q2 = QueryInput(query="SELECT * FROM schools LIMIT 5", backend="datafusion")
        >>> result = generic_dialect_metric(q1, q2, "db.sqlite", dataset_name="bird")
        >>> print(result.results_equal)
        True
    """

    def _run_comparison() -> MetricResult:
        """Run the actual comparison with timeout protection."""
        # Execute query1
        result1 = None
        query1_executed = False
        query1_error = None
        query1_shape = None

        try:
            if query1.query_type == "substrait":
                # Don't use context manager - factory manages connection lifecycle
                engine = ExecutionEngineFactory.create(
                    db_path,
                    engine_type="substrait",
                    backend=query1.backend,
                    load_data=load_data,
                    dataset_name=dataset_name,
                    source_db_type=source_db_type,
                )
                result1 = engine.execute_substrait(query1.query)  # type: ignore
                query1_executed = True
                query1_shape = result1.shape
            else:

                def execute_query1():
                    # Don't use context manager - factory manages connection lifecycle
                    engine = ExecutionEngineFactory.create(
                        db_path,
                        engine_type="sql",
                        backend=query1.backend,
                        load_data=load_data,
                        dataset_name=dataset_name,
                        source_db_type=source_db_type,
                    )
                    return engine.execute(query1.query)

                result1 = func_timeout(COMPARISON_TIMEOUT, execute_query1)
                query1_executed = True
                query1_shape = result1.shape
        except (FunctionTimedOut, Exception) as e:
            query1_error = str(e)

        # Execute query2
        result2 = None
        query2_executed = False
        query2_error = None
        query2_shape = None

        try:
            if query2.query_type == "substrait":
                # Don't use context manager - factory manages connection lifecycle
                engine = ExecutionEngineFactory.create(
                    db_path,
                    engine_type="substrait",
                    backend=query2.backend,
                    load_data=load_data,
                    dataset_name=dataset_name,
                    source_db_type=source_db_type,
                )
                result2 = engine.execute_substrait(query2.query)  # type: ignore
                query2_executed = True
                query2_shape = result2.shape
            else:

                def execute_query2():
                    # Don't use context manager - factory manages connection lifecycle
                    engine = ExecutionEngineFactory.create(
                        db_path,
                        engine_type="sql",
                        backend=query2.backend.replace("ibis-", ""),
                        load_data=load_data,
                        dataset_name=dataset_name,
                        source_db_type=source_db_type,
                    )
                    return engine.execute(query2.query)

                result2 = func_timeout(COMPARISON_TIMEOUT, execute_query2)
                query2_executed = True
                query2_shape = result2.shape
        except (FunctionTimedOut, Exception) as e:
            query2_error = str(e)

        # Compare results if both executed
        both_executed = query1_executed and query2_executed
        results_equal = None
        df1_values = None
        df2_values = None
        df1_dtypes = None
        df2_dtypes = None

        if both_executed:
            try:
                # Compare values, ignoring dtype differences
                # Different backends may return different dtypes for the same data
                results_equal = _compare_dataframes(result1, result2)
                # if not results_equal:
                #     df1_values = result1.values.tolist()
                #     df2_values = result2.values.tolist()
                #     df1_dtypes = result1.dtypes.astype(str).to_dict()
                #     df2_dtypes = result2.dtypes.astype(str).to_dict()
            except Exception as e:
                results_equal = False
                query2_error = f"Comparison error: {e}"

        # Return Pydantic model
        return MetricResult(
            query1_executed=query1_executed,
            query2_executed=query2_executed,
            both_executed=both_executed,
            results_equal=results_equal,
            query1_error=query1_error,
            query2_error=query2_error,
            query1_shape=query1_shape,
            query2_shape=query2_shape,
            query1_result=result1,
            query2_result=result2,
            df1_values_on_mismatch=df1_values,
            df2_values_on_mismatch=df2_values,
            df1_dtypes_on_mismatch=df1_dtypes,
            df2_dtypes_on_mismatch=df2_dtypes,
            query1_backend=query1.backend,
            query2_backend=query2.backend,
        )

    return _run_comparison()
