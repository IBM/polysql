from __future__ import annotations

import re
from typing import Dict, Set, Tuple

from polysql.evaluation.prompts.base import remove_examples_from_schema

SQL_GEN_TYPES: Set[str] = {
    "sqlite",
    "sqlite-ss",
    "duckdb",
    "postgres",
    "mysql",
    "snowflake",
    "bigquery",
    "datafusion",
    "pyspark",
    "clickhouse",
    "databricks",
}


def parse_instruction_level(level: int) -> Tuple[int, int]:
    """Parse two-digit instruction level into (cot_level, dialect_level).

    Valid combinations: 11, 12, 13, 21, 22, 23
    Format: XY where X=COT level (1-2), Y=dialect level (1-3)

    Args:
        level: Two-digit instruction level

    Returns:
        Tuple of (cot_level, dialect_level)

    Raises:
        ValueError: If level is not a valid combination
    """
    valid_levels = [11, 12, 13, 21, 22, 23]
    if level not in valid_levels:
        raise ValueError(
            f"Invalid instruction level: {level}. "
            f"Valid levels: {valid_levels} "
            f"(format: XY where X=COT level 1-2, Y=dialect level 1-3)"
        )

    cot_level = level // 10
    dialect_level = level % 10
    return cot_level, dialect_level


# Chain-of-Thought instruction templates
_COT_INSTRUCTIONS = """
INSTRUCTIONS:
1. Think step-by-step: Analyze the schema, the dialect constraints (quoting, types), and the user intent.
2. Outline your logic briefly."""

COT_TEMPLATES: Dict[int, str] = {
    1: "",  # No COT
    2: _COT_INSTRUCTIONS,  # With COT
}

# ------------------------------------------------------------------------------
# SNOWFLAKE: Focus on Quoting + Date Casts (Common Inertia Point)
# ------------------------------------------------------------------------------
_SNOWFLAKE_INSTRUCTIONS_L2: str = """
CRITICAL SNOWFLAKE RULES:
- QUOTING: ALL table/column names MUST be double-quoted ("Name") to preserve case. Unquoted identifiers are uppercased.
- DATES: Use `TO_DATE`, `TO_TIMESTAMP`, `TO_VARCHAR`. DO NOT use `strftime`.
- CASTING: Use `::` syntax (e.g., `x::STRING`).
- MATCHING: Use `ILIKE` for case-insensitive matching (Snowflake is case-sensitive by default).
"""

_SNOWFLAKE_INSTRUCTIONS_L3: str = (
    _SNOWFLAKE_INSTRUCTIONS_L2
    + """
Example (Strict Quoting + Date Handling):
Question: Find users created in 2023.
SQL:
SELECT "u"."user_name", "u"."created_at"
FROM "Users" AS "u"
WHERE "u"."created_at" >= TO_DATE('2023-01-01')
  AND "u"."status" ILIKE 'active'
"""
)

# ------------------------------------------------------------------------------
# POSTGRES: Focus on Arrays + Intervals + JSON
# ------------------------------------------------------------------------------
_POSTGRES_INSTRUCTIONS_L2: str = """
POSTGRESQL-SPECIFIC FEATURES:
- ARRAYS: Use `ARRAY[]` literals and `ANY()` for comparisons.
- CASTING: Use `::TYPE` (e.g., `col::TEXT`).
- DATES: Use `INTERVAL '1 day'` logic.
- JSON: Use `->` (field) and `->>` (text) operators.
- MATCHING: Use `ILIKE` for case-insensitive search.
"""

_POSTGRES_INSTRUCTIONS_L3: str = (
    _POSTGRES_INSTRUCTIONS_L2
    + """
Example (Arrays + Intervals):
Question: Find orders from the last 30 days containing 'book'.
SQL:
SELECT "order_id", "items"::TEXT
FROM "Orders"
WHERE "order_date" > CURRENT_DATE - INTERVAL '30 days'
  AND 'book' = ANY("items_array")
"""
)

# ------------------------------------------------------------------------------
# MYSQL: Focus on Backticks + DateFormat (Inertia: Models use || for concat)
# ------------------------------------------------------------------------------
_MYSQL_INSTRUCTIONS_L2: str = """
MYSQL-SPECIFIC FEATURES:
- QUOTING: Use backticks (`col`) not double quotes.
- CONCAT: Use `CONCAT(a, b)` function. DO NOT use `||`.
- DATES: Use `DATE_FORMAT(col, '%Y-%m-%d')`.
- LIMIT: Use `LIMIT offset, count` or `LIMIT count OFFSET offset`.
"""

_MYSQL_INSTRUCTIONS_L3: str = (
    _MYSQL_INSTRUCTIONS_L2
    + """
Example (Backticks + Concat + Date):
Question: List full names and formatted dates.
SQL:
SELECT CONCAT(`first_name`, ' ', `last_name`), DATE_FORMAT(`reg_date`, '%Y-%m')
FROM `users`
WHERE `status` = 'active'
LIMIT 10
"""
)

# ------------------------------------------------------------------------------
# DUCKDB: Focus on QUALIFY + List Lambdas (High Hallucination Area)
# ------------------------------------------------------------------------------
_DUCKDB_INSTRUCTIONS_L2: str = """
DUCKDB-SPECIFIC FEATURES:
- CASE SENSITIVITY: Identifiers are case-sensitive. "Table" != "table". Match schema EXACTLY.
- WINDOWS: Use `QUALIFY` to filter window functions (replaces subqueries).
- LISTS: Use `list_transform`, `list_filter`.
- DATES: Use `strftime` or `date_trunc`.
- QUOTING: Double quotes for identifiers.
"""

_DUCKDB_INSTRUCTIONS_L3: str = (
    _DUCKDB_INSTRUCTIONS_L2
    + """
Example (QUALIFY + Exact Case):
Question: Get the most recent order for each customer.
SQL:
SELECT "OrderID", "OrderDate", "Amount"
FROM "Orders"
WHERE "OrderDate" > '2022-01-01'
QUALIFY ROW_NUMBER() OVER (PARTITION BY "CustomerID" ORDER BY "OrderDate" DESC) = 1
"""
)

# ------------------------------------------------------------------------------
# BIGQUERY: Focus on UNNEST + Table Prefixes (Strictness Mismatch)
# ------------------------------------------------------------------------------
_BIGQUERY_INSTRUCTIONS_L2: str = """
BIGQUERY-SPECIFIC FEATURES:
- NAMING: Table names MUST be `dataset.table`. Match schema exactly.
- ARRAYS: Use `UNNEST(array_col)` to flatten.
- DATES: Use `DATE('2023-01-01')` literals. Strict types.
- QUOTING: Backticks `dataset.table` are optional but recommended.
"""

_BIGQUERY_INSTRUCTIONS_L3: str = (
    _BIGQUERY_INSTRUCTIONS_L2
    + """
Example (UNNEST + Dataset Prefix):
Question: Count distinct items in user orders.
SQL:
SELECT u.id, COUNT(DISTINCT item)
FROM `ecommerce_db.users` AS u,
UNNEST(u.order_items) AS item
WHERE u.signup_date >= DATE('2022-01-01')
GROUP BY u.id
"""
)

# ------------------------------------------------------------------------------
# SQLITE: Focus on Date Math (The #1 Failure Mode for SQLite)
# ------------------------------------------------------------------------------
_SQLITE_INSTRUCTIONS_L2: str = """
SQLITE-SPECIFIC FEATURES:
- DATES: No native date type. Use `strftime('%Y', col)` or `julianday()`.
- MATH: Date math requires `date(col, '+1 day')`.
- JOIN: No RIGHT/FULL JOIN.
- QUOTING: Double quotes "col" for identifiers with spaces.
"""

_SQLITE_INSTRUCTIONS_L3: str = (
    _SQLITE_INSTRUCTIONS_L2
    + """
Example (Date Modifiers + Glob):
Question: Find events in the last 7 days starting with 'A'.
SQL:
SELECT "Event Name", "Event Date"
FROM "Events"
WHERE "Event Date" BETWEEN date('now', '-7 days') AND date('now')
  AND "Event Name" GLOB 'A*'
"""
)

# ------------------------------------------------------------------------------
# PYSPARK: Focus on Explode + Backticks
# ------------------------------------------------------------------------------
_PYSPARK_INSTRUCTIONS_L2: str = """
SPARK SQL-SPECIFIC FEATURES:
- QUOTING: Use backticks `col`.
- ARRAYS: Use `explode(col)` or `posexplode(col)`.
- DATES: `to_date()`, `date_add()`.
- WINDOWS: Standard OVER clauses supported.
"""

_PYSPARK_INSTRUCTIONS_L3: str = (
    _PYSPARK_INSTRUCTIONS_L2
    + """
Example (Explode + Backticks):
Question: Flatten transaction items.
SQL:
SELECT `t`.`id`, `item`
FROM `transactions` AS `t`
LATERAL VIEW explode(`t`.`items`) AS `item`
WHERE `t`.`amount` > 100
"""
)

# ------------------------------------------------------------------------------
# CLICKHOUSE: Focus on String Conversion + Final
# ------------------------------------------------------------------------------
_CLICKHOUSE_INSTRUCTIONS_L2: str = """
CLICKHOUSE-SPECIFIC FEATURES:
- STRICT TYPES: Explicit casts required `toString()`, `toInt32()`.
- ARRAYS: `arrayJoin` (like UNNEST) or `arrayMap`.
- JOINS: Use `GLOBAL JOIN` or specific join syntax if sharded.
- DEDUPLICATION: Use `FINAL` on ReplacingMergeTree if required.
"""

_CLICKHOUSE_INSTRUCTIONS_L3: str = (
    _CLICKHOUSE_INSTRUCTIONS_L2
    + """
Example (Strict Types + ArrayJoin):
Question: List tags for specific posts.
SQL:
SELECT `p`.`id`, toString(`p`.`title`), `tag`
FROM `posts` AS `p`
ARRAY JOIN `p`.`tags` AS `tag`
WHERE `p`.`views` > toInt32(1000)
"""
)

# ------------------------------------------------------------------------------
# DATABRICKS: Focus on 3-Level Namespace
# ------------------------------------------------------------------------------
_DATABRICKS_INSTRUCTIONS_L2: str = """
DATABRICKS SPECIFIC FEATURES:
- NAMESPACE: MUST use `catalog`.`schema`.`table`. Match schema EXACTLY.
- FUNCTIONS: `try_cast()`, `array_contains()`.
- DATES: `date_format()`, `to_date()`.
"""

_DATABRICKS_INSTRUCTIONS_L3: str = (
    _DATABRICKS_INSTRUCTIONS_L2
    + """
Example (3-Level Namespace):
Question: Select users from the silver layer.
SQL:
SELECT `id`, `email`
FROM `main`.`silver_users`.`profiles`
WHERE `updated_at` > current_date() - INTERVAL 1 DAY
"""
)

# ------------------------------------------------------------------------------
# DATAFUSION: Focus on Quoting (Arrow limitation)
# ------------------------------------------------------------------------------
_DATAFUSION_INSTRUCTIONS_L2: str = """
DATAFUSION SPECIFIC FEATURES:
- CASE SENSITIVITY: EXTREME. "Col" != "col".
- QUOTING: ALWAYS quote "Table" and "Column".
- SYNTAX: Standard SQL (Postgres-like).
"""

_DATAFUSION_INSTRUCTIONS_L3: str = (
    _DATAFUSION_INSTRUCTIONS_L2
    + """
Example (Aggressive Quoting):
Question: Average age by city.
SQL:
SELECT "City", AVG("Age")
FROM "Census_Data"
GROUP BY "City"
ORDER BY AVG("Age") DESC
"""
)

SQL_DIALECT_INSTRUCTIONS: Dict[str, Dict[int, str]] = {
    "snowflake": {1: "", 2: _SNOWFLAKE_INSTRUCTIONS_L2, 3: _SNOWFLAKE_INSTRUCTIONS_L3},
    "postgres": {1: "", 2: _POSTGRES_INSTRUCTIONS_L2, 3: _POSTGRES_INSTRUCTIONS_L3},
    "mysql": {1: "", 2: _MYSQL_INSTRUCTIONS_L2, 3: _MYSQL_INSTRUCTIONS_L3},
    "duckdb": {1: "", 2: _DUCKDB_INSTRUCTIONS_L2, 3: _DUCKDB_INSTRUCTIONS_L3},
    "bigquery": {1: "", 2: _BIGQUERY_INSTRUCTIONS_L2, 3: _BIGQUERY_INSTRUCTIONS_L3},
    "sqlite": {1: "", 2: _SQLITE_INSTRUCTIONS_L2, 3: _SQLITE_INSTRUCTIONS_L3},
    "pyspark": {1: "", 2: _PYSPARK_INSTRUCTIONS_L2, 3: _PYSPARK_INSTRUCTIONS_L3},
    "datafusion": {
        1: "",
        2: _DATAFUSION_INSTRUCTIONS_L2,
        3: _DATAFUSION_INSTRUCTIONS_L3,
    },
    "clickhouse": {
        1: "",
        2: _CLICKHOUSE_INSTRUCTIONS_L2,
        3: _CLICKHOUSE_INSTRUCTIONS_L3,
    },
    "databricks": {
        1: "",
        2: _DATABRICKS_INSTRUCTIONS_L2,
        3: _DATABRICKS_INSTRUCTIONS_L3,
    },
}

_BASE_SQL_PROMPT_TEMPLATE_NO_COT: str = """TASK: Convert natural language to {dialect_name} SQL query.
DIALECT: {dialect_name}
{dialect_specific_instructions}
INSTRUCTIONS:
Generate ONLY the SQL query inside a markdown code block, like ```sql SELECT ... ```.
Do not include any reasoning or explanation.

Database Schema:
{schema}

Question:
{question}

Response:"""

_BASE_SQL_PROMPT_TEMPLATE_WITH_COT: str = """TASK: Convert natural language to {dialect_name} SQL query.
DIALECT: {dialect_name}
{dialect_specific_instructions}{cot_instructions}

INSTRUCTIONS:
Generate the FINAL SQL query inside a markdown code block, like ```sql SELECT ... ```.

Database Schema:
{schema}

Question:
{question}

Response:"""


def get_sql_prompt(
    schema: str,
    question: str,
    gen_type: str,
    instruction_level: int,
) -> str:
    """Generate SQL prompt with COT and dialect-specific instructions.

    Args:
        schema: Database schema
        question: Natural language question
        gen_type: SQL dialect type (e.g., 'mysql', 'sqlite-postgres')
        instruction_level: Two-digit level XY where X=COT (1-2), Y=dialect (1-3)

    Returns:
        Formatted prompt for LLM
    """
    cleaned_schema = remove_examples_from_schema(schema)

    # Parse instruction level into COT and dialect components
    cot_level, dialect_level = parse_instruction_level(instruction_level)

    # Handle sqlite-{dialect} pattern (e.g., sqlite-postgres, sqlite-duckdb)
    # For these, we generate SQLite code that will be transpiled later
    source_dialect = gen_type
    if gen_type.startswith("sqlite-"):
        source_dialect = "sqlite"

    # Map internal backend names to display names for prompts
    dialect_name = source_dialect.replace("-ss", "")
    if dialect_name == "pyspark":
        dialect_name = "SPARK"
    else:
        dialect_name = dialect_name.upper()

    # Get COT instructions
    cot_instructions = COT_TEMPLATES.get(cot_level, "")

    # Get dialect instructions
    dialect_instructions_map = SQL_DIALECT_INSTRUCTIONS.get(source_dialect, {})
    dialect_instructions = dialect_instructions_map.get(dialect_level, "")

    # Compose final instructions: dialect first, then COT
    if cot_instructions:
        if dialect_instructions:
            combined_instructions = dialect_instructions + "\n\n" + cot_instructions + "\n"
        else:
            combined_instructions = cot_instructions + "\n"
    else:
        combined_instructions = dialect_instructions

    # Choose template based on COT level
    if cot_level == 1:
        template = _BASE_SQL_PROMPT_TEMPLATE_NO_COT
    else:
        template = _BASE_SQL_PROMPT_TEMPLATE_WITH_COT

    return template.format(
        dialect_name=dialect_name,
        dialect_specific_instructions=combined_instructions,
        cot_instructions="",  # COT is now part of combined_instructions
        schema=cleaned_schema,
        question=question,
    )


def parse_sql_from_response(response: str) -> str:
    """
    Extracts the SQL query from the model response.
    1. Priority: Markdown code block (```sql ... ```).
    2. Fallback: Looks for a SQL statement starting with SELECT or WITH.
    3. Failure: Returns "Parsing Error" string.
    """
    # 1. Try Markdown Code Blocks
    # Use a variable to construct the pattern and avoid
    # writing triple backticks literally.
    bt = "`"
    pattern_block = f"{bt * 3}(?:sql)?\\s*(.*?){bt * 3}"
    matches = re.findall(pattern_block, response, re.DOTALL | re.IGNORECASE)

    if matches:
        return matches[-1].strip()

    # 2. Fallback: Try to find raw SQL (SELECT/WITH)
    # We look for a pattern starting with SELECT/WITH and grab everything to the end.
    # This salvages the query if the model forgets the formatting but gets the logic right.
    fallback_pattern = r"\b(SELECT|WITH)\s+[\s\S]+"
    match = re.search(fallback_pattern, response, re.IGNORECASE)

    if match:
        return match.group(0).strip()

    # 3. Return explicit error string
    # Ensure your pipeline catches this string before execution.
    return "Parsing Error"
