import sqlite3
from typing import Any

import ibis
from pydantic import BaseModel
from sqlglot import optimizer, parse_one, transpile
from transpilers.utils.query_execution import (
    execute_substrait,
)
from transpilers.utils.result_comparison import are_dfs_equal, compare_sql


class IBISInput(BaseModel):
    ibis_code: str
    # tables: dict[str, Any]


class SQLInput(BaseModel):
    query: str


class SSInput(BaseModel):
    plan: Any


def sql_and_ss_are_the_same(sql: SQLInput, substrait: SSInput, db_file: str) -> bool:
    """Verifies transpilation using Substrait execution comparison.

    Args:
        input_sql: The original SQL query.
        substrait_plan: The compiled Substrait plan.
        train_or_dev: The dataset split to use ('train' or 'dev').

    Returns:
        True if verification passes, False otherwise.
    """
    predicted_from_substrait = execute_substrait(db_file, substrait.plan)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    conn.execute("BEGIN TRANSACTION;")
    cursor.execute(sql.query)
    predicted_from_input_sql = cursor.fetchall()
    conn.close()

    return are_dfs_equal(predicted_from_input_sql, predicted_from_substrait)


def sql_and_ibis_are_same(sql: SQLInput, ibis_input: IBISInput, db_file: str) -> bool:
    """Verifies transpilation using SQL execution comparison.

    Args:
        input_sql: The original SQL query.
        ibis_expr: The Ibis expression to verify.
        train_or_dev: The dataset split to use ('train' or 'dev').

    Returns:
        True if verification passes, False otherwise.
    """

    # locals().update(ibis_input.tables)
    execution_scope = {}
    exec(ibis_input.ibis_code, execution_scope)
    ibis_expr = execution_scope["result"]

    # Convert Ibis expression back to SQL
    sql_from_ibis = ibis.to_sql(ibis_expr, dialect="sqlite")
    backtranslated_sql = optimizer.optimize(
        parse_one(sql_from_ibis, read="sqlite"), dialect="sqlite"
    ).sql()
    backtranslated_sql = transpile(backtranslated_sql, normalize=True, identify=True)[0]

    # Compare execution results
    compare_results = compare_sql(
        question_id=0,
        db_file=db_file,
        question="verification",
        ground_truth=sql.query,
        pred_sql=backtranslated_sql,
    )

    return compare_results[-2] == "Correct"
