"""Core transpilation evaluation logic for SQL cross-dialect analysis.

This module provides utilities for:
1. Transpiling SQL queries using sqlglot (syntax-based)
2. Transpiling SQL queries using LLMs (semantic-aware)
3. Evaluating transpilation correctness
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import ibis
import pandas as pd
import sqlglot
import sqlglot.errors
from sqlglot import exp

from polysql.evaluation.backends.connectors.base import (
    get_mysql_credentials,
    get_postgres_credentials,
)
from polysql.evaluation.core.model import CrossProviderInferenceEngineWithMoreRISTModels
from polysql.evaluation.metrics.dialect_comparison import (
    QueryInput,
    generic_dialect_metric,
)


# ============================================================================
# Identifier Normalization Utilities
# ============================================================================


def normalize_identifier(name: str) -> str:
    """Normalize identifier to snake_case for matching.

    Handles: CamelCase, PascalCase, mixed_Case_Names, and acronyms (CDSCode → cds_code)
    """
    # Insert underscore between lowercase/digit and uppercase: "customerId" → "customer_Id"
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    # Insert underscore between multiple uppercase and lowercase: "CDSCode" → "CDS_Code"
    name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    # Convert to lowercase
    return name.lower()


def normalize_for_fuzzy_match(name: str) -> str:
    """Normalize identifier by removing underscores, spaces, and hyphens for fuzzy matching.

    Used to match 'bordercolor' with 'border_color', 'School Name' with 'school_name', etc.
    """
    return name.replace('_', '').replace(' ', '').replace('-', '').lower()


# ============================================================================
# Sqlglot Utilities
# ============================================================================


def strip_qualifiers(expression):
    """Remove database/catalog qualifiers from SQL AST."""
    for table in expression.find_all(exp.Table):
        table.set("db", None)
        table.set("catalog", None)
        table.set("this", exp.Identifier(this=table.name, quoted=True))
    return expression


def extract_identifiers(sql: str, dialect: str) -> Dict[str, Set[str]]:
    """Extract table and column names from SQL query.

    Returns dict with 'tables' and 'columns' sets.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return {"tables": set(), "columns": set()}

    tables = set()
    columns = set()

    for node in parsed.walk():
        if isinstance(node, exp.Table):
            tables.add(node.name.lower())
        elif isinstance(node, exp.Column):
            col_name = node.name if isinstance(node.name, str) else str(node.this)
            columns.add(col_name.lower())

    return {"tables": tables, "columns": columns}


def get_target_schema(
    target_db_name: str,
    target_dialect: str,
    table_names: Set[str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Get schema from target database for specific tables.

    Returns:
        - schema_map: {table_fuzzy: {column_fuzzy: actual_column_name}}
        - table_map: {table_fuzzy: actual_table_name}
    """
    schema = {}
    table_map = {}

    try:
        # Connect directly to target database
        if target_dialect == "mysql":
            creds = get_mysql_credentials()
            conn = ibis.mysql.connect(
                host=creds["host"],
                port=creds["port"],
                user=creds["user"],
                password=creds["password"],
                database=target_db_name
            )
        elif target_dialect in ["postgres", "postgresql"]:
            creds = get_postgres_credentials()
            conn = ibis.postgres.connect(
                host=creds["host"],
                port=creds["port"],
                user=creds["user"],
                password=creds["password"],
                database=target_db_name,
                schema=target_db_name
            )
        else:
            return schema, table_map

        # Get tables (normalized matching for CamelCase → snake_case)
        available_tables = conn.list_tables()

        for requested_table in table_names:
            # Find matching table (normalize both sides, with fuzzy fallback)
            matched_table = None
            requested_normalized = normalize_identifier(requested_table)
            requested_fuzzy = normalize_for_fuzzy_match(requested_table)

            # First try exact normalized match
            for avail_table in available_tables:
                avail_normalized = normalize_identifier(avail_table)
                if avail_normalized == requested_normalized:
                    matched_table = avail_table
                    break

            # Fallback to fuzzy match (strip underscores)
            if not matched_table:
                for avail_table in available_tables:
                    avail_fuzzy = normalize_for_fuzzy_match(avail_table)
                    if avail_fuzzy == requested_fuzzy:
                        matched_table = avail_table
                        break

            if matched_table:
                # Store table name mapping
                table_map[requested_fuzzy] = matched_table

                # Get columns for this table
                table_obj = conn.table(matched_table)
                columns = table_obj.columns
                # Build schema map with fuzzy keys for flexible matching
                col_map = {}
                for col in columns:
                    col_fuzzy = normalize_for_fuzzy_match(col)
                    col_map[col_fuzzy] = col
                schema[requested_fuzzy] = col_map

        conn.disconnect()

    except Exception as e:
        print(f"[Schema extraction] Failed for database '{target_db_name}': {e}")

    return schema, table_map


def apply_schema_mapping(
    sql: str,
    dialect: str,
    schema_mapping: Dict[str, Dict[str, str]],
    table_mapping: Dict[str, str]
) -> str:
    """Apply schema mapping to SQL by replacing table and column names.

    Args:
        schema_mapping: {table_fuzzy: {column_fuzzy: actual_column}}
        table_mapping: {table_fuzzy: actual_table_name}

    Uses fuzzy matching (strips underscores, spaces, hyphens) to match identifiers.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)

        # First pass: Replace table names
        for node in parsed.walk():
            if isinstance(node, exp.Table):
                table_name = node.name if isinstance(node.name, str) else str(node.this)
                table_fuzzy = normalize_for_fuzzy_match(table_name)

                if table_fuzzy in table_mapping:
                    actual_table = table_mapping[table_fuzzy]
                    node.set("this", exp.Identifier(this=actual_table, quoted=True))

        # Second pass: Replace column names
        for node in parsed.walk():
            if isinstance(node, exp.Column):
                col_name = node.name if isinstance(node.name, str) else str(node.this)
                col_fuzzy = normalize_for_fuzzy_match(col_name)

                # Try to find mapping using fuzzy matching
                if node.table:
                    table_fuzzy = normalize_for_fuzzy_match(node.table)
                    if table_fuzzy in schema_mapping:
                        if col_fuzzy in schema_mapping[table_fuzzy]:
                            actual_col = schema_mapping[table_fuzzy][col_fuzzy]
                            node.set("this", exp.Identifier(this=actual_col))
                            continue

                # Otherwise check all tables for this column
                for table_schema in schema_mapping.values():
                    if col_fuzzy in table_schema:
                        actual_col = table_schema[col_fuzzy]
                        node.set("this", exp.Identifier(this=actual_col))
                        break

        return parsed.sql(dialect=dialect)

    except Exception as e:
        print(f"[Schema mapping] Failed: {e}")
        return sql


# ============================================================================
# Database Path Resolution
# ============================================================================


def get_db_path(dataset_path: str, db_id: str) -> Path:
    """Resolve the SQLite database path based on dataset path conventions.

    Priority: MINIDEV > BIRD > fallback
    """
    dataset_path_str = str(dataset_path).lower()

    # Check MINIDEV first (for mini_dev datasets)
    if "minidev" in dataset_path_str or "mini_dev" in dataset_path_str:
        minidev_path = Path(f"data/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite")
        if minidev_path.exists():
            return minidev_path

    # Check BIRD next
    if "bird" in dataset_path_str:
        bird_path = Path(f"data/BIRD/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite")
        if bird_path.exists():
            return bird_path

    # Fallback to standard
    return Path(f"data/databases/{db_id}/{db_id}.sqlite")


# ============================================================================
# LLM Transpilation
# ============================================================================


def extract_schema_from_prompt(full_prompt: str) -> str:
    """Extract the Database Schema section from full_prompt.

    Returns schema section starting from 'Database Schema:' until the next major section.
    """
    if "Database Schema:" not in full_prompt:
        return ""

    # Extract from "Database Schema:" onwards
    schema_start = full_prompt.find("Database Schema:")
    prompt_after_schema = full_prompt[schema_start:]

    # Find where schema section ends (usually at "Question:" or end of string)
    schema_end_markers = ["\nQuestion:", "\nQUESTION:", "\n\n\n", "\nUser:"]
    schema_end = len(prompt_after_schema)
    for marker in schema_end_markers:
        pos = prompt_after_schema.find(marker)
        if pos != -1:
            schema_end = min(schema_end, pos)

    schema = prompt_after_schema[:schema_end].strip()
    return schema


def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from LLM response (handles markdown code blocks)."""
    # Try to extract from ```sql ... ``` block
    sql_block_pattern = r"```sql\s+(.*?)\s+```"
    match = re.search(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code block ``` ... ```
    code_block_pattern = r"```\s+(.*?)\s+```"
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no code block found
    return response.strip()


def transpile_with_llm(
    gold_sql: str,
    schema: str,
    source_dialect: str,
    target_dialect: str,
    inference_engine: CrossProviderInferenceEngineWithMoreRISTModels
) -> Tuple[Optional[str], Optional[str]]:
    """Transpile SQL query using LLM.

    Returns:
        (transpiled_sql, error_message)
        If successful: (sql_string, None)
        If failed: (None, error_message)
    """
    # Build detailed transpilation prompt with explicit normalization instructions
    prompt = f"""TASK: Translate the SQLite query below to {target_dialect.upper()} SQL using the exact schema provided.

TARGET DIALECT: {target_dialect.upper()}

{schema}

CRITICAL INSTRUCTIONS:

1. **Table and Column Name Mapping - MOST IMPORTANT**:
   - The source query uses table/column names that may NOT match the target schema exactly
   - Source might use: `Player`, `Player_Attributes`, `Customer_Name` (CamelCase/PascalCase)
   - Target schema uses: `player`, `player_attributes`, `customer_name` (lowercase)
   - **ACTION REQUIRED**: Convert ALL table and column identifiers to lowercase with underscores
   - **Apply this to**:
     * Table names in FROM/JOIN clauses: `Player` → `player`
     * Column names in SELECT/WHERE/ORDER BY: keep exact case from schema
     * Table aliases: keep as-is (t1, t2, etc.)

   **Example transformation**:
   ```sql
   -- Source (SQLite):
   SELECT t1.player_name FROM Player AS t1

   -- Target (MySQL/Postgres):
   SELECT t1.player_name FROM player AS t1
   -- Note: Changed "Player" to "player", kept "player_name" as-is
   ```

2. **Syntax Translation**:
   - For MySQL: Use backticks for identifiers, CONCAT() for concatenation, LIMIT syntax
   - For PostgreSQL: Use double quotes for identifiers if needed, || for concatenation, LIMIT/OFFSET syntax
   - Convert SQLite-specific functions to {target_dialect.upper()} equivalents

3. **Preserve Query Logic**:
   - Keep the exact same logic, joins, filters, aggregations
   - Do NOT change subqueries, CTEs, or query structure
   - Only change syntax and identifiers

4. **Output Format**:
   - Output ONLY the SQL query inside a ```sql code block
   - Do NOT include explanations or comments

SOURCE QUERY (SQLite):
{gold_sql}

CONVERTED {target_dialect.upper()} QUERY:"""

    try:
        # Use inference engine (returns list of predictions)
        from unitxt.loaders import LoadFromDictionary
        dataset = LoadFromDictionary(
            data={"test": [{"source": [{"role": "user", "content": prompt}]}]},
            data_classification_policy=["public"],
        ).process().to_dataset()

        responses = inference_engine(dataset["test"])

        if not responses or len(responses) == 0:
            return None, "No response from model"

        transpiled_sql = extract_sql_from_response(responses[0])
        return transpiled_sql, None

    except Exception as e:
        return None, str(e)


def batch_transpile_with_llm(
    requests: list[dict],
    inference_engine: CrossProviderInferenceEngineWithMoreRISTModels,
    include_mapping_instructions: bool = False
) -> list[Tuple[Optional[str], Optional[str]]]:
    """Batch transpile multiple SQL queries using LLM.

    Args:
        requests: List of dicts with keys:
            - gold_sql: Source SQL query
            - schema: Target database schema
            - source_dialect: Source SQL dialect (e.g., "sqlite")
            - target_dialect: Target SQL dialect (e.g., "mysql", "postgres")
            - _key: Optional tracking key
        include_mapping_instructions: If True, include detailed identifier mapping instructions.
                                      If False (default), use minimal prompt for pure LLM evaluation.

    Returns:
        List of (transpiled_sql, error_message) tuples in same order as requests.
        If successful: (sql_string, None)
        If failed: (None, error_message)
    """
    if not requests:
        return []

    # Build all prompts
    prompts = []
    for req in requests:
        gold_sql = req["gold_sql"]
        schema = req["schema"]
        source_dialect = req.get("source_dialect", "sqlite")
        target_dialect = req["target_dialect"]

        if include_mapping_instructions:
            # Detailed prompt with explicit identifier mapping instructions
            prompt = f"""TASK: Translate the {source_dialect.upper()} query below to {target_dialect.upper()} SQL using the exact schema provided.

TARGET DIALECT: {target_dialect.upper()}

{schema}

CRITICAL INSTRUCTIONS:

1. **Table and Column Name Mapping - MOST IMPORTANT**:
   - The source query uses table/column names that may NOT match the target schema exactly
   - Source might use: `Player`, `Player_Attributes`, `Customer_Name` (CamelCase/PascalCase)
   - Target schema uses: `player`, `player_attributes`, `customer_name` (lowercase)
   - **ACTION REQUIRED**: Convert ALL table and column identifiers to lowercase with underscores
   - **Apply this to**:
     * Table names in FROM/JOIN clauses: `Player` → `player`
     * Column names in SELECT/WHERE/ORDER BY: keep exact case from schema
     * Table aliases: keep as-is (t1, t2, etc.)

   **Example transformation**:
   ```sql
   -- Source ({source_dialect.upper()}):
   SELECT t1.player_name FROM Player AS t1

   -- Target ({target_dialect.upper()}):
   SELECT t1.player_name FROM player AS t1
   -- Note: Changed "Player" to "player", kept "player_name" as-is
   ```

2. **Syntax Translation**:
   - For MySQL: Use backticks for identifiers, CONCAT() for concatenation, LIMIT syntax
   - For PostgreSQL: Use double quotes for identifiers if needed, || for concatenation, LIMIT/OFFSET syntax
   - Convert {source_dialect.upper()}-specific functions to {target_dialect.upper()} equivalents

3. **Preserve Query Logic**:
   - Keep the exact same logic, joins, filters, aggregations
   - Do NOT change subqueries, CTEs, or query structure
   - Only change syntax and identifiers

4. **Output Format**:
   - Output ONLY the SQL query inside a ```sql code block
   - Do NOT include explanations or comments

SOURCE QUERY ({source_dialect.upper()}):
{gold_sql}

CONVERTED {target_dialect.upper()} QUERY:"""
        else:
            # Minimal prompt for pure LLM evaluation (no mapping instructions)
            prompt = f"""TASK: Translate the {source_dialect.upper()} query below to {target_dialect.upper()} SQL using the schema provided.

TARGET DIALECT: {target_dialect.upper()}

{schema}

INSTRUCTIONS:
- Translate SQL syntax from {source_dialect.upper()} to {target_dialect.upper()}
- Use table and column names as defined in the schema above
- Preserve the query logic exactly
- Output ONLY the SQL query inside a ```sql code block

SOURCE QUERY ({source_dialect.upper()}):
{gold_sql}

CONVERTED {target_dialect.upper()} QUERY:"""

        prompts.append(prompt)

    try:
        # Batch inference
        from unitxt.loaders import LoadFromDictionary
        dataset = LoadFromDictionary(
            data={"test": [{"source": [{"role": "user", "content": p}]} for p in prompts]},
            data_classification_policy=["public"],
        ).process().to_dataset()

        responses = inference_engine(dataset["test"])

        if not responses:
            return [(None, "No response from model") for _ in requests]

        # Extract SQL from each response
        results = []
        for response in responses:
            try:
                transpiled_sql = extract_sql_from_response(response)
                results.append((transpiled_sql, None))
            except Exception as e:
                results.append((None, str(e)))

        return results

    except Exception as e:
        error_msg = str(e)
        return [(None, error_msg) for _ in requests]


# ============================================================================
# Evaluation Methods
# ============================================================================


def evaluate_sqlglot_pred_to_source(
    predicted_sql: str,
    gold_sql: str,
    read_dialect: str,
    sqlite_db_path: Path
):
    """Method 1: Transpile prediction back to SQLite and compare with gold.

    Returns:
        True if transpiled prediction matches gold on SQLite, False if mismatch, pd.NA if failed
    """
    if not sqlite_db_path.exists():
        return pd.NA  # Database not found

    try:
        # Transpile prediction (MySQL/Postgres → SQLite)
        expression = sqlglot.parse_one(predicted_sql, read=read_dialect)
        expression = strip_qualifiers(expression)
        transpiled_pred = expression.sql(dialect="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, sqlglot.errors.UnsupportedError, RecursionError):
        return pd.NA  # Parse/tokenization/recursion error = method failed

    try:
        # Execute both on SQLite
        q1 = QueryInput(query=gold_sql, backend="sqlite")
        q2 = QueryInput(query=transpiled_pred, backend="sqlite")

        metric_result = generic_dialect_metric(
            q1, q2,
            db_path=sqlite_db_path,
            load_data=False,
            source_db_type="sqlite"
        )

        # Check execution status
        if not metric_result.both_executed:
            if not metric_result.query1_executed:
                # Original gold failed to execute
                return False  # Bad gold query (not transpilation's fault)
            if not metric_result.query2_executed:
                # Transpiled prediction failed to execute
                return pd.NA  # Bad transpilation

        # Both executed successfully
        if metric_result.results_equal is None:
            return pd.NA  # Comparison error
        return metric_result.results_equal  # True or False

    except Exception:
        # DB connection errors, timeouts, etc.
        return pd.NA


def evaluate_sqlglot_gold_to_target(
    gold_sql: str,
    predicted_sql: str,
    write_dialect: str,
    sqlite_db_path: Path,
    apply_schema_mapping: bool = False
):
    """Method 2: Transpile gold to target, optionally with schema mapping.

    Args:
        apply_schema_mapping: If True, apply fuzzy schema mapping heuristics on top of sqlglot.
                              If False (default), use pure sqlglot transpilation output.

    Returns:
        True if transpiled gold matches prediction on target DB, False if mismatch, pd.NA if failed
    """
    try:
        # Transpile gold (SQLite → MySQL/Postgres) using sqlglot
        expression = sqlglot.parse_one(gold_sql, read="sqlite")
        expression = strip_qualifiers(expression)
        transpiled_gold = expression.sql(dialect=write_dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, sqlglot.errors.UnsupportedError, RecursionError):
        return pd.NA  # Parse/tokenization/recursion error = method failed

    # Optionally apply schema mapping heuristics (fuzzy matching for identifiers)
    if apply_schema_mapping:
        # Extract identifiers from gold SQL to know what schema to fetch
        identifiers = extract_identifiers(gold_sql, "sqlite")

        # Compute target database name (same logic as connector)
        db_hash = hashlib.md5(str(sqlite_db_path).encode()).hexdigest()[:8]
        target_db_name = f"sqlite_import_{db_hash}"

        # Get target database schema for referenced tables
        target_schema, table_map = get_target_schema(
            target_db_name, write_dialect, identifiers['tables']
        )

        # Apply fuzzy schema mapping to use target column names
        if target_schema:
            transpiled_gold_mapped = apply_schema_mapping(
                transpiled_gold, write_dialect, target_schema, table_map
            )
        else:
            transpiled_gold_mapped = transpiled_gold
    else:
        # Pure sqlglot output (no heuristics)
        transpiled_gold_mapped = transpiled_gold

    try:
        # Execute both on target DB
        q1 = QueryInput(query=transpiled_gold_mapped, backend=write_dialect)
        q2 = QueryInput(query=predicted_sql, backend=write_dialect)

        metric_result = generic_dialect_metric(
            q1, q2,
            db_path=sqlite_db_path,
            load_data=False,
            source_db_type="sqlite"
        )

        # Check execution status
        if not metric_result.both_executed:
            if not metric_result.query1_executed:
                # Transpiled gold failed to execute
                return pd.NA  # Bad transpilation
            if not metric_result.query2_executed:
                # Original prediction failed to execute
                return False  # Bad prediction (not transpilation's fault)

        # Both executed successfully
        if metric_result.results_equal is None:
            return pd.NA  # Comparison error
        return metric_result.results_equal  # True or False

    except Exception:
        # DB connection errors, timeouts, etc.
        return pd.NA


def evaluate_llm_gold_to_target(
    gold_sql: str,
    predicted_sql: str,
    schema: str,
    write_dialect: str,
    sqlite_db_path: Path,
    inference_engine: CrossProviderInferenceEngineWithMoreRISTModels,
    verbose: bool = False
) -> Tuple[bool, bool, Optional[str]]:
    """Method 3: LLM-transpile gold to target, compare with prediction and verify correctness.

    Returns:
        (matches_prediction, transpilation_correct, transpiled_sql)
        - matches_prediction: True if LLM-transpiled gold matches model prediction
        - transpilation_correct: True if LLM-transpiled gold matches original gold results
        - transpiled_sql: The LLM-transpiled SQL query (or None if failed)
    """
    try:
        # Transpile using LLM
        transpiled_gold_llm, llm_error = transpile_with_llm(
            gold_sql=gold_sql,
            schema=schema,
            source_dialect="sqlite",
            target_dialect=write_dialect,
            inference_engine=inference_engine
        )

        if llm_error or not transpiled_gold_llm:
            return False, False, None

        if verbose:
            print(f"\nLLM-Transpiled SQL ({write_dialect}):")
            print(transpiled_gold_llm[:300])

        # Test 1: Does LLM-transpiled match prediction?
        q1 = QueryInput(query=transpiled_gold_llm, backend=write_dialect)
        q2 = QueryInput(query=predicted_sql, backend=write_dialect)

        metric_result = generic_dialect_metric(
            q1, q2,
            db_path=sqlite_db_path,
            load_data=False,
            source_db_type="sqlite"
        )

        matches_prediction = metric_result.results_equal if metric_result.results_equal is not None else False

        # Test 2: Does LLM-transpiled produce same results as original gold?
        q_gold_sqlite = QueryInput(query=gold_sql, backend="sqlite")
        q_llm_target = QueryInput(query=transpiled_gold_llm, backend=write_dialect)

        transpilation_check = generic_dialect_metric(
            q_gold_sqlite, q_llm_target,
            db_path=sqlite_db_path,
            load_data=False,
            source_db_type="sqlite"
        )

        transpilation_correct = transpilation_check.results_equal if transpilation_check.results_equal is not None else False

        return matches_prediction, transpilation_correct, transpiled_gold_llm

    except Exception:
        return False, False, None
