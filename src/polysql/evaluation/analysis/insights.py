"""Generate LLM-authored insights for evaluation result files using judge+aggregator pattern."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from unitxt.loaders import LoadFromDictionary

from polysql.evaluation.core.model import CrossProviderInferenceEngineWithMoreRISTModels

CLASSIFICATION_KEY_FIELD = "_classification_key"

load_dotenv()

DEFAULT_JUDGE_TEMPLATE = """
You are a meticulous SQL evaluation analyst. Your task is to analyze a single failed NL-to-SQL prediction and classify the PRIMARY root cause of the failure.

================================================================================
CONTEXT: MULTI-DIALECT SQL EVALUATION (THE "GAP" ANALYSIS)
================================================================================

This evaluation tests whether language models can generalize from SQLite (Academic Standard) to Enterprise Dialects (PostgreSQL, BigQuery, Snowflake, etc.).

CRITICAL CONTEXT: 
1. The model's query SUCCEEDED in SQLite (returned correct results).
2. The exact same query (conceptually) FAILED in the Target Dialect (e.g., PostgreSQL).
3. Your job is to explain WHY the transition caused a failure.

Key fields you will receive:
- `gen_type`: The TARGET SQL dialect (e.g., postgres, bigquery, snowflake).
- `schema`: The database schema in the TARGET dialect.
- `predicted_sql`: The model's generated SQL.
- `gold_sql`: A reference query in SQLite (provided ONLY for intent understanding).
- `question`: The natural language question.
- `pred_error`: Error message from executing predicted_sql (if any).
- `results_equal`: Whether both queries returned the same results (False = failure).

================================================================================
YOUR TASK
================================================================================

1. If `results_equal` is True → output `null` (no error to classify).
2. Otherwise, determine the PRIMARY root cause.
3. Choose the MOST SPECIFIC category that applies.
4. Follow the DECISION PROCEDURE strictly (it prioritizes Logic errors over Syntax errors).

================================================================================
ERROR CATEGORIES (use these exact names)
================================================================================

-------------------------------------------------------------------------------
CATEGORY 1: schema_linking_error (HALLUCINATION)
-------------------------------------------------------------------------------
The model referenced columns or tables that DO NOT EXIST in the provided target schema.

A) COLUMN/TABLE HALLUCINATION:
   - Referencing a column/table name that doesn't exist.
   - Using a column from the wrong table.
   - NOTE: If the table exists in SQLite but NOT in the Target Schema, this is a schema_linking_error (model failed to read the new schema).

-------------------------------------------------------------------------------
CATEGORY 2: filtering_error (LOGIC FAILURE)
-------------------------------------------------------------------------------
The model wrote syntactically valid SQL (or SQL that failed due to type mismatch in a WHERE clause), but the underlying LOGIC selects the wrong rows.

A) LOGICAL MISMATCH:
   - Wrong comparison operators (`>` vs `>=`).
   - Wrong boolean logic (`AND` vs `OR`).
   - Missing JOINs or wrong JOIN conditions.
   - Constraint/Integrity Violations (e.g., query fails due to NOT NULL constraint).

B) TYPE/VALUE LOGIC ERRORS:
   - Comparing incompatible types (e.g., string '2023' vs integer 2023) IF it represents a failure to understand the data model.
   - SQLite-isms that imply wrong typing (e.g., `date_col + 7` treating date as int).
   - *Note:* If this throws a syntax error in Postgres, it is still a FILTERING error (logic) because the model fundamentally misunderstood the data type.

-------------------------------------------------------------------------------
CATEGORY 3: aggregation_error (GROUPING FAILURE)
-------------------------------------------------------------------------------
The model selected the correct rows, but failed in aggregation, grouping, or ordering.

A) GROUPING/OUTPUT LOGIC:
   - Wrong `GROUP BY` columns (often violates strict modes in Postgres/BigQuery).
   - Missing `GROUP BY` when aggregating.
   - Wrong aggregate function (`COUNT` vs `SUM`).
   - Wrong `ORDER BY` direction or column.

-------------------------------------------------------------------------------
CATEGORY 4: dialect_error (SYNTAX ONLY)
-------------------------------------------------------------------------------
The model's LOGIC (Schema, Filtering, Aggregation) is CORRECT, but it used syntax or functions forbidden in the target dialect.

A) SYNTAX IGNORANCE:
   - Using SQLite functions in Target (e.g., `strftime` in BigQuery).
   - Hallucinated functions that exist nowhere.
   - Wrong quoting (backticks vs double quotes).
   - Wrong casting SYNTAX (e.g., `::int` in MySQL).

B) STRICTNESS VIOLATION (SLOPPINESS):
   - Query would work in SQLite but violates strict typing rules in Target (where the logic is arguably correct, but syntax is too loose).
   - EXCLUDES: GROUP BY violations (Must be classified as Category 3).

================================================================================
DECISION PROCEDURE (STRICT ORDER)
================================================================================

1. CHECK FOR SCHEMA REFERENCE ERRORS
   - Did it invent a table or column?
   -> If yes: `schema_linking_error`

2. CHECK FOR ROW SELECTION/LOGIC ERRORS
   - Did it filter on the wrong logic?
   - Did it try to compare a String to an Int (logic flaw)?
   - Did it mess up the JOINs?
   -> If yes: `filtering_error`

3. CHECK FOR AGGREGATION/OUTPUT ERRORS
   - Did it group by the wrong thing?
   - Did it fail a "Strict Group By" check?
   -> If yes: `aggregation_error`

4. CHECK FOR DIALECT SYNTAX ERRORS
   - Everything else makes sense, but it used `strftime` instead of `EXTRACT`?
   - Everything else makes sense, but it used `"` instead of `` ` ``?
   -> If yes: `dialect_error`

5. CHECK FOR EVALUATION/PROCESS ISSUES (Last Resort)
   - The SQL is flawless, but execution failed? (Timeout/Crash)
   - The SQL is flawless, but `results_equal` is False? (Float/Sort mismatch)
   - The Gold Query is wrong?
   -> If yes: `invalid_evaluation`

================================================================================
OUTPUT FORMAT
================================================================================

If `results_equal` is True:
Output only: null

Otherwise output valid JSON exactly like this:
{{
    "question_id": <int>,
    "category": "<schema_linking_error|filtering_error|aggregation_error|dialect_error|invalid_evaluation>",
    "explanation": "<1-2 sentences explaining the root cause>",
    "evidence": "<Short quote from SQL or error message proving the classification>"
}}
================================================================================
PREDICTION DATA TO ANALYZE
================================================================================

{prediction_json}
"""

DEFAULT_AGGREGATOR_TEMPLATE = (
    "You are a staff evaluation analyst. You have received individual error classifications from multiple judges. "
    "Synthesize these judgments into a comprehensive evaluation report.\n\n"
    "**Input data:**\n"
    "- Experiment metadata: {metadata_json}\n"
    "- Judge classifications: {judgments_json}\n\n"
    "**Output requirements:**\n"
    "Produce a Markdown report with these sections:\n\n"
    "1. **Summary**\n"
    "   - Report: experiment_id, total, executed, correct, accuracy\n"
    "   - List all available metadata from exp_config\n\n"
    "2. **Failure Patterns**\n"
    "   - Group judgments by pattern\n"
    "   - For each pattern with 2+ occurrences, create a numbered subsection:\n"
    "     * State the pattern catalog name\n"
    "     * List question_id members\n"
    "     * Provide representative examples of key_difference (≤120 chars each)\n"
    "     * Explain the shared mistake\n"
    "     * Label as **Model error** or **Evaluation framework error**\n"
    '   - Merge singleton patterns into closest related pattern or create "Other Issues" section\n\n'
    "3. **Pattern Totals**\n"
    "   - Table with columns: `Pattern` | `Count` | `Question IDs`\n"
    "   - Include every catalog pattern listed above (show 0 if none)\n"
    "   - Sort by count descending\n\n"
    "4. **TL;DR**\n"
    "   - 3-4 bullet points summarizing key takeaways\n"
    "   - Explicitly state whether any **Evaluation framework error** patterns were detected\n\n"
    "**Rules:**\n"
    "- Use only facts from the provided data\n"
    "- Minimum 250 words\n"
    '- If a field is missing, write "not provided"\n'
)

DEFAULT_META_SUMMARY_TEMPLATE = (
    "You are a staff evaluation analyst reviewing multiple NL2DSL insight reports. "
    "Each report already follows the house format (Summary, Failure Patterns, "
    "Pattern Totals, TL;DR) and uses the shared failure pattern catalog. Combine "
    "these reports into a cross-experiment meta-summary while preserving the "
    "catalog semantics.\n\n"
    "Output must be Markdown with the following structure:\n"
    "1. Overview - list the total number of reports, each `experiment_id`, and the "
    "key metrics (`total`, `executed`, `correct`, `accuracy`) in a table.\n"
    "2. Cross-Experiment Patterns - highlight the most significant recurring "
    "failure patterns. Reference experiments and pattern names from the catalog. "
    "Include supporting evidence such as representative `question_id`s.\n"
    "3. Pattern Totals - aggregate counts across all reports. Provide a table with "
    "columns `Pattern`, `Total Question IDs`, and `Affected Experiments`. Sum the "
    "counts using each report's Pattern Totals section.\n"
    "4. TL;DR - up to four bullet points summarizing the key takeaways, explicitly "
    "stating whether any **Evaluation framework error** clusters appeared in the "
    "underlying reports.\n\n"
    "Additional rules:\n"
    "- Use only information contained in the supplied reports; do not invent data.\n"
    "- When referencing a report, cite it as `Report <n>` or by its "
    "`experiment_id`.\n"
    "- If a pattern name is absent from all reports, list it in the Pattern Totals "
    "table with zero counts.\n"
    "- Ensure the final response contains at least 200 words.\n\n"
    "Insight reports to analyze:\n{reports}\n"
)

SCORE_WITHOUT_EXE_TEMPLATE = """
Predict if predicted_sql would return the same results as gold_sql when executed on the target database.

{prediction_json}

Check for:
- Syntax errors (wrong functions/operators for dialect)
- Schema errors (wrong table/column names)
- Logic errors (different WHERE/JOIN/aggregation than gold_sql)

Examples:
{{"match": true}}   ← Correct syntax, schema, and logic
{{"match": false}}  ← Has syntax/schema/logic errors

Return ONLY valid JSON with no extra text:
"""

SCORE_WITH_EXE_TEMPLATE = """
You are a SQL evaluation expert. Your task is to determine whether two query results are semantically equivalent, even if they differ in formatting, column names, or minor details.

================================================================================
SCENARIO
================================================================================

Both the gold query and predicted query were executed on their respective databases.
You are comparing their actual results to determine if they represent the same data.

================================================================================
DATA PROVIDED
================================================================================

Question: {question}

Gold Result:
{gold_result}

Predicted Result:
{predicted_result}

================================================================================
YOUR TASK
================================================================================

Determine if the two results are semantically equivalent.

Consider these as EQUIVALENT:
- Same data with different column names (e.g., "Name" vs "NAME" vs "name")
- Same data in different row order (unless order was explicitly requested)
- Floating point precision differences (e.g., 3.14159 vs 3.14)
- Equivalent type representations (e.g., 1.0 vs 1, "true" vs true)
- Minor date/time format differences (e.g., "2023-01-01" vs "2023-01-01 00:00:00")
- Whitespace differences in strings

Consider these as DIFFERENT:
- Different number of rows
- Different values in corresponding cells
- Missing or extra columns with data
- Different data types that change meaning (e.g., "123" vs 123 in contexts where it matters)

================================================================================
OUTPUT FORMAT
================================================================================

Return ONLY valid JSON in this exact format:
{{
    "match": true
}}

OR

{{
    "match": false
}}

- "match": true means the results are semantically equivalent
- "match": false means the results represent different data

DO NOT include any explanation, reasoning, or additional fields. ONLY return the JSON with the "match" boolean.
"""


class EvaluationInsightsGenerator:
    """Judge+Aggregator pipeline for generating evaluation insights."""

    def __init__(
        self,
        model_name: str = "gpt-oss-120b",
        provider: str = "rits",
        judge_max_tokens: int = 2048,
        aggregator_max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.inference_engine = CrossProviderInferenceEngineWithMoreRISTModels(
            model=model_name,
            provider=provider,
            data_classification_policy=["public"],
            max_tokens=judge_max_tokens,
            temperature=temperature,
            seed=42,
        )
        self.aggregator_max_tokens = aggregator_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.temperature = temperature

    def _extract_schema(self, full_prompt: str) -> str:
        """Extract CREATE TABLE statements from full_prompt."""
        create_table_pattern = re.compile(
            r"(CREATE\s+TABLE\s+.*?;)", re.IGNORECASE | re.DOTALL
        )
        matches = create_table_pattern.findall(full_prompt)
        if matches:
            return "\n\n".join(matches)
        return ""

    def _extract_table_names_from_sql(self, sql: str) -> set[str]:
        """Extract table names from SQL query."""
        if not sql:
            return set()

        # Simple regex to find table names after FROM, JOIN, INTO, UPDATE
        # Handles: FROM table, JOIN table, INTO table, UPDATE table
        # Also handles: table1, table2 or table AS alias
        table_pattern = re.compile(
            r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE
        )
        matches = table_pattern.findall(sql)
        return set(name.lower() for name in matches)

    def _filter_schema_to_relevant_tables(
        self, full_schema: str, gold_sql: str, predicted_sql: str
    ) -> str:
        """Filter schema to only include tables referenced in the SQL queries."""
        if not full_schema:
            return ""

        # Extract table names from both queries
        gold_tables = self._extract_table_names_from_sql(gold_sql)
        pred_tables = self._extract_table_names_from_sql(predicted_sql)
        relevant_tables = gold_tables | pred_tables

        if not relevant_tables:
            # If we couldn't extract any tables, return full schema as fallback
            return full_schema

        # Extract all CREATE TABLE statements
        create_table_pattern = re.compile(
            r"(CREATE\s+TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+.*?;)",
            re.IGNORECASE | re.DOTALL,
        )

        filtered_statements = []
        for match in create_table_pattern.finditer(full_schema):
            full_statement = match.group(1)
            table_name = match.group(2).lower()

            if table_name in relevant_tables:
                filtered_statements.append(full_statement)

        if filtered_statements:
            return "\n\n".join(filtered_statements)
        else:
            # If filtering resulted in empty schema, return full schema as fallback
            return full_schema

    def _build_judge_prompts(self, predictions: list[dict]) -> list[str]:
        """Build judge prompts for all predictions."""
        prompts = []
        for pred in predictions:
            prediction_data = {
                "question_id": pred.get("question_id"),
                "db_id": pred.get("db_id"),
                "gen_type": pred.get("gen_type"),
                "question": pred.get("question"),
                "gold_sql": pred.get("gold_sql"),
                "predicted_sql": pred.get("predicted_sql"),
                "results_equal": pred.get("results_equal"),
                "gold_error": pred.get("gold_error"),
                "pred_error": pred.get("pred_error"),
                "schema": self._extract_schema(pred.get("full_prompt", "")),
            }
            prediction_json = json.dumps(prediction_data, indent=2)
            prompt = DEFAULT_JUDGE_TEMPLATE.format(prediction_json=prediction_json)
            prompts.append(prompt)
        return prompts

    def _validate_judge_output(self, output: str, question_id: int) -> Optional[dict]:
        """Parse and validate judge output JSON."""
        output = output.strip()

        if not output or output == "None":
            print(
                f"WARNING: Skipping question_id {question_id} - Judge output is empty or None. This usually means the model returned no output."
            )
            return None

        if output == "null":
            return None

        try:
            judgment = json.loads(output)
        except json.JSONDecodeError as e:
            print(
                f"WARNING: Skipping question_id {question_id} - Judge output is not valid JSON: {e}\nOutput: {output[:500]}"
            )
            return None

        required_fields = [
            "question_id",
            "category",
            "explanation",
        ]
        for field in required_fields:
            if field not in judgment:
                print(
                    f"WARNING: Skipping question_id {question_id} - Judge output missing required field '{field}': {judgment}"
                )
                return None

        return judgment

    def _judge_predictions(self, predictions: list[dict]) -> list[dict]:
        """Send all predictions to judge model in batch, return valid judgments only."""
        prompts = self._build_judge_prompts(predictions)

        dataset = (
            LoadFromDictionary(
                data={"test": [{"source": prompt} for prompt in prompts]},
                data_classification_policy=["public"],
            )
            .process()
            .to_dataset()
        )

        raw_outputs = self.inference_engine(dataset["test"])

        judgments = []
        for idx, (pred, output) in enumerate(zip(predictions, raw_outputs)):
            question_id = pred.get("question_id", idx)
            judgment = self._validate_judge_output(str(output), question_id)
            if judgment is not None:
                classification_key = pred.get(CLASSIFICATION_KEY_FIELD)
                if classification_key is not None:
                    judgment[CLASSIFICATION_KEY_FIELD] = classification_key
                judgments.append(judgment)

        return judgments

    def judge_predictions(self, predictions: list[dict]) -> list[dict]:
        """Public wrapper for obtaining validated judge outputs."""
        return self._judge_predictions(predictions)

    def _validate_score_output(
        self, output: str, classification_key: str
    ) -> Optional[dict]:
        """Parse and validate score output JSON. Returns None if invalid."""
        output = output.strip()

        if not output or output == "None":
            print(
                f"WARNING: Score output is empty or None for key {classification_key}. Skipping instance."
            )
            return None

        try:
            score_result = json.loads(output)
        except json.JSONDecodeError as e:
            print(
                f"WARNING: Score output is not valid JSON for key {classification_key}: {e}. Skipping instance."
            )
            return None

        if "match" not in score_result:
            print(
                f"WARNING: Score output missing 'match' field for key {classification_key}. Skipping instance."
            )
            return None

        if not isinstance(score_result["match"], bool):
            print(
                f"WARNING: Score 'match' field must be boolean for key {classification_key}. Skipping instance."
            )
            return None

        return score_result

    def score_predictions_without_exe(self, predictions: list[dict]) -> list[dict]:
        """
        Score predictions without target execution (LLM predicts correctness).

        Simulates scenario: Source DB available, target DB not available.
        LLM predicts if predicted_sql would return correct results.

        Args:
            predictions: List with keys:
                - question: natural language question
                - schema: target database schema
                - gold_sql: source SQL that was executed
                - gold_result: results from source database
                - gold_error: error from source database (if any)
                - predicted_sql: target SQL generated by model
                - gen_type: target dialect
                - _classification_key: merge key for tracking

        Returns:
            List of {_classification_key: str, score: bool}
            Fails immediately if LLM returns invalid format.
        """
        prompts = []
        for pred in predictions:
            prediction_data = {
                "question": pred.get("question"),
                "schema": pred.get("schema"),
                "gold_sql": pred.get("gold_sql"),
                "gold_result": pred.get("gold_result"),
                "gold_error": pred.get("gold_error", ""),
                "predicted_sql": pred.get("predicted_sql"),
                "gen_type": pred.get("gen_type"),
            }
            prediction_json = json.dumps(prediction_data, indent=2)
            prompt = SCORE_WITHOUT_EXE_TEMPLATE.format(prediction_json=prediction_json)
            prompts.append(prompt)

        dataset = (
            LoadFromDictionary(
                data={"test": [{"source": prompt} for prompt in prompts]},
                data_classification_policy=["public"],
            )
            .process()
            .to_dataset()
        )

        raw_outputs = self.inference_engine(dataset["test"])

        scores = []
        for pred, output in zip(predictions, raw_outputs):
            classification_key = pred.get(CLASSIFICATION_KEY_FIELD)
            if classification_key is None:
                raise ValueError("Prediction missing _classification_key field")

            score_result = self._validate_score_output(str(output), classification_key)
            if score_result is None:
                # Skip this instance - warning already printed
                continue

            scores.append(
                {
                    CLASSIFICATION_KEY_FIELD: classification_key,
                    "score": score_result["match"],
                }
            )

        return scores

    def score_predictions_with_exe(self, predictions: list[dict]) -> list[dict]:
        """
        Score predictions with execution results (LLM judges equivalence).

        Note: Heuristics should be applied BEFORE calling this method.

        Args:
            predictions: List with keys:
                - question: natural language question (context)
                - gold_result: results from source database
                - predicted_result: results from target database
                - _classification_key: merge key for tracking

        Returns:
            List of {_classification_key: str, score: bool}
            Fails immediately if LLM returns invalid format.
        """
        # prompts = []
        # for pred in predictions:
        #     prompt = SCORE_WITH_EXE_TEMPLATE.format(
        #         question=pred.get("question", ""),
        #         gold_result=pred.get("gold_result", ""),
        #         predicted_result=pred.get("predicted_result", ""),
        #     )
        #     prompts.append(prompt)

        # dataset = (
        #     LoadFromDictionary(
        #         data={"test": [{"source": prompt} for prompt in prompts]},
        #         data_classification_policy=["public"],
        #     )
        #     .process()
        #     .to_dataset()
        # )

        # raw_outputs = self.inference_engine(dataset["test"])

        scores = []
        # for pred, output in zip(predictions, raw_outputs):
        #     classification_key = pred.get(CLASSIFICATION_KEY_FIELD)
        #     if classification_key is None:
        #         raise ValueError("Prediction missing _classification_key field")

        #     score_result = self._validate_score_output(str(output), classification_key)
        #     if score_result is None:
        #         # Skip this instance - warning already printed
        #         continue

        #     scores.append(
        #         {
        #             CLASSIFICATION_KEY_FIELD: classification_key,
        #             "score": score_result["match"],
        #         }
        #     )

        return scores

    def _build_aggregator_prompt(
        self, judgments: list[dict], metadata: dict, instructions: Optional[str]
    ) -> str:
        """Build aggregator prompt from judge outputs and metadata."""
        metadata_json = json.dumps(metadata, indent=2)
        judgments_json = json.dumps(judgments, indent=2)
        template = instructions or DEFAULT_AGGREGATOR_TEMPLATE
        return template.format(
            metadata_json=metadata_json, judgments_json=judgments_json
        )

    def _aggregate_judgments(
        self, judgments: list[dict], metadata: dict, instructions: Optional[str]
    ) -> str:
        """Send judgments to aggregator model, return final markdown report."""
        self.inference_engine.max_tokens = self.aggregator_max_tokens
        prompt = self._build_aggregator_prompt(judgments, metadata, instructions)

        dataset = (
            LoadFromDictionary(
                data={"test": [{"source": prompt}]},
                data_classification_policy=["public"],
            )
            .process()
            .to_dataset()
        )

        raw_output = self.inference_engine(dataset["test"])
        if not raw_output:
            raise ValueError("Aggregator returned no output.")

        self.inference_engine.max_tokens = self.judge_max_tokens
        return str(raw_output[0])

    def generate_from_dict(
        self, result: dict, instructions: Optional[str] = None
    ) -> str:
        """Generate insights from evaluation result dictionary."""
        predictions = result.get("predictions", [])
        if not predictions:
            raise ValueError("Result has no predictions to analyze.")

        metadata = {
            "exp_config": result.get("exp_config", {}),
            "total": result.get("total", 0),
            "executed": result.get("executed", 0),
            "correct": result.get("correct", 0),
            "accuracy": result.get("accuracy", 0.0),
        }

        judgments = self._judge_predictions(predictions)
        insights = self._aggregate_judgments(judgments, metadata, instructions)

        return insights

    def generate_from_json(
        self, json_content: str, instructions: Optional[str] = None
    ) -> str:
        """Generate insights from JSON string."""
        result = json.loads(json_content)
        return self.generate_from_dict(result, instructions)

    def generate_from_path(
        self, result_path: Path, instructions: Optional[str] = None
    ) -> str:
        """Generate insights from result file path."""
        json_content = result_path.read_text()
        return self.generate_from_json(json_content, instructions)

    def generate_meta_summary(
        self, reports: list[str], instructions: Optional[str] = None
    ) -> str:
        """
        Produce a cross-experiment meta-summary from individual insight reports.

        Args:
            reports: List of already-generated insight reports (Markdown strings).
            instructions: Optional override template. Must contain a `{reports}` placeholder.

        Returns:
            Markdown meta-summary string.
        """
        if not reports:
            raise ValueError(
                "At least one report is required for meta-summary generation."
            )

        formatted_reports = []
        for idx, report in enumerate(reports, start=1):
            formatted_reports.append(f"---\nReport {idx}:\n{report.strip()}\n")
        reports_block = "\n".join(formatted_reports)
        template = instructions or DEFAULT_META_SUMMARY_TEMPLATE
        prompt = template.format(reports=reports_block)

        self.inference_engine.max_tokens = self.aggregator_max_tokens

        dataset = (
            LoadFromDictionary(
                data={"test": [{"source": prompt}]},
                data_classification_policy=["public"],
            )
            .process()
            .to_dataset()
        )

        raw_output = self.inference_engine(dataset["test"])
        if not raw_output:
            raise ValueError("Meta-summary generation returned no output.")

        self.inference_engine.max_tokens = self.judge_max_tokens
        return str(raw_output[0])


def write_insights_for_file(
    result_path: Path,
    output_path: Optional[Path] = None,
    *,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
    judge_max_tokens: int = 2048,
    aggregator_max_tokens: int = 4096,
    temperature: float = 0.0,
    instructions: Optional[str] = None,
) -> Path:
    """
    Generate insights using judge+aggregator pattern and persist them alongside the result file.

    Returns:
        Path to the written insights file.
    """
    generator = EvaluationInsightsGenerator(
        model_name=model_name,
        provider=provider,
        judge_max_tokens=judge_max_tokens,
        aggregator_max_tokens=aggregator_max_tokens,
        temperature=temperature,
    )
    insights_text = generator.generate_from_path(
        result_path=result_path, instructions=instructions
    )

    destination = (
        output_path
        if output_path is not None
        else result_path.with_suffix(".insights.md")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(insights_text)
    return destination


def write_meta_summary(
    report_paths: list[Path],
    output_path: Path,
    *,
    model_name: str = "gpt-oss-120b",
    provider: str = "rits",
    aggregator_max_tokens: int = 4096,
    temperature: float = 0.0,
    instructions: Optional[str] = None,
) -> Path:
    """
    Generate a meta-summary from multiple insight report files and write it to disk.
    """
    if not report_paths:
        raise ValueError("report_paths must contain at least one file.")

    reports = [path.read_text() for path in report_paths]
    generator = EvaluationInsightsGenerator(
        model_name=model_name,
        provider=provider,
        aggregator_max_tokens=aggregator_max_tokens,
        temperature=temperature,
    )
    meta_summary = generator.generate_meta_summary(
        reports=reports, instructions=instructions
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(meta_summary)
    return output_path


__all__ = [
    "EvaluationInsightsGenerator",
    "write_insights_for_file",
    "write_meta_summary",
]
