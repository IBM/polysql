from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_archer_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from Archer dataset format.

    Archer format is a JSON array (not JSONL) with fields:
    - db_id: database identifier
    - query: gold SQL query
    - question: natural language question
    - reasoning_type, commonsense_knowledge: metadata

    This loader converts to NL2DSL format by:
    - Constructing db_path from db_id
    - Renaming: question → nl_query, query → sql_query

    Args:
        dataset_path: Path to the Archer dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)  # Load as JSON array, not JSONL

    # Convert Archer format to NL2DSL format
    for i, sample in enumerate(samples):
        db_id = sample["db_id"]
        # Construct db_path: data/archer/database/{db_id}/{db_id}.sqlite
        sample["db_path"] = f"data/archer/database/{db_id}/{db_id}.sqlite"

        # Prepend commonsense knowledge to question if present
        question = sample.pop("question")
        commonsense = sample.get("commonsense_knowledge", "")
        if commonsense:
            sample["nl_query"] = f"{commonsense};\n{question}"
        else:
            sample["nl_query"] = question

        sample["sql_query"] = sample.pop("query")
        # Add index if not present
        if "idx" not in sample:
            sample["idx"] = i

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def load_bird_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from BIRD dataset format.

    BIRD format is a JSON array with fields:
    - question_id
    - db_id: database identifier
    - question: natural language question
    - evidence: hint/evidence for the question
    - SQL: gold SQL query
    - difficulty: simple/moderate/challenging

    This loader converts to NL2DSL format by:
    - Constructing db_path from dataset location
    - Prepending evidence to question if present
    - Renaming: question → nl_query, SQL → sql_query

    Args:
        dataset_path: Path to the BIRD dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)  # Load as JSON array

    # Determine database base path from dataset path
    # e.g., /path/to/BIRD/dev_20240627/dev.json → /path/to/BIRD/dev_20240627/dev_databases
    import os

    dataset_dir = os.path.dirname(dataset_path)
    db_base_path = os.path.join(dataset_dir, "dev_databases")

    # Convert BIRD format to NL2DSL format
    for sample in samples:
        db_id = sample["db_id"]
        # Construct db_path
        sample["db_path"] = f"{db_base_path}/{db_id}/{db_id}.sqlite"

        # Prepend evidence to question if present
        question = sample.pop("question")
        evidence = sample.pop("evidence", "")
        if evidence:
            sample["nl_query"] = f"{evidence};\n{question}"
        else:
            sample["nl_query"] = question

        # Rename SQL to sql_query
        sample["sql_query"] = sample.pop("SQL")

        # Use question_id as idx
        sample["idx"] = sample.get("question_id", 0)

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def load_bird_minidev_native(
    dataset_path: str,
    sample_size: Optional[int] = None,
    random_state: int = 0,
    db_type: str = "mysql",
) -> List[dict]:
    """Load samples from BIRD MINIDEV MySQL/PostgreSQL format.

    MINIDEV MySQL/PostgreSQL uses native database backends where each db_id
    has its own database named minidev_{db_type}_{db_id}.

    The JSON format is identical to BIRD, but db_path points to the
    native database name rather than SQLite files.

    Args:
        dataset_path: Path to the MINIDEV dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.
        db_type: Database type prefix ("mysql" or "postgres").

    Returns:
        List of dictionaries in NL2DSL format.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)

    # Convert MINIDEV format to NL2DSL format
    for sample in samples:
        db_id = sample["db_id"]
        # Map db_id to native database name
        sample["db_path"] = f"minidev_{db_type}_{db_id}"

        # Prepend evidence to question if present
        question = sample.pop("question")
        evidence = sample.pop("evidence", "")
        if evidence:
            sample["nl_query"] = f"{evidence};\n{question}"
        else:
            sample["nl_query"] = question

        # Rename SQL to sql_query
        sample["sql_query"] = sample.pop("SQL")

        # Use question_id as idx
        sample["idx"] = sample.get("question_id", 0)

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def load_beaver_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from Beaver dataset format.

    Beaver format is a JSON array with fields:
    - db_id: database identifier
    - sql: gold MySQL query
    - question: natural language question
    - oracle_sql: Oracle SQL version (not used)
    - gold_tables, mapping, join_keys: metadata

    This loader converts to NL2DSL format by:
    - Constructing db_path from db_id to MySQL dump files
    - Renaming: question → nl_query, sql → sql_query

    Args:
        dataset_path: Path to the Beaver dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)  # Load as JSON array, not JSONL

    # Convert Beaver format to NL2DSL format
    for i, sample in enumerate(samples):
        db_id = sample["db_id"]

        # Map db_id to MySQL dump file path
        if db_id == "dw":
            sample["db_path"] = "data/beaver/databases/DW/new_dw_indexed.sql"
        else:
            # NW databases: csail_stata_nova, csail_stata_neutron, etc.
            sample["db_path"] = f"data/beaver/databases/NW/{db_id}.sql"

        # Rename fields
        sample["nl_query"] = sample.pop("question")
        sample["sql_query"] = sample.pop("sql")

        # Add gold query dialect (Beaver uses MySQL)
        sample["gold_query_dialect"] = "mysql"

        # Add index if not present
        if "idx" not in sample:
            sample["idx"] = i

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def load_spider_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from Spider dataset format.

    Spider format is a JSON array with fields:
    - db_id: database identifier
    - query: gold SQL query
    - question: natural language question
    - sql: parsed SQL structure (not used)
    - query_toks, query_toks_no_value, question_toks: tokenized versions

    This loader converts to NL2DSL format by:
    - Constructing db_path from db_id
    - Renaming: question → nl_query, query → sql_query

    Args:
        dataset_path: Path to the Spider dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)

    # Determine database base path from dataset path
    # e.g., /path/to/spider/dev.json → /path/to/spider/database
    import os

    dataset_dir = os.path.dirname(dataset_path)
    db_base_path = os.path.join(dataset_dir, "database")

    # Convert Spider format to NL2DSL format
    for i, sample in enumerate(samples):
        db_id = sample["db_id"]
        # Construct db_path: data/spider/database/{db_id}/{db_id}.sqlite
        sample["db_path"] = f"{db_base_path}/{db_id}/{db_id}.sqlite"

        # Rename fields
        sample["nl_query"] = sample.pop("question")
        sample["sql_query"] = sample.pop("query")

        # Add index if not present
        if "idx" not in sample:
            sample["idx"] = i

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def load_spider2_lite_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from Spider2-Lite dataset format.

    Spider2-Lite format is JSONL (one JSON object per line) with fields:
    - instance_id: unique identifier
    - db: database name
    - question: natural language question
    - external_knowledge: reference to schema documentation (optional)
    - db_path: path to SQLite database (added by enrichment script)
    - gold_result_csv: path to pre-computed gold result CSV (added by enrichment script)
    - gold_result_variants: list of available result variants (added by enrichment script)

    This loader converts to NL2DSL format by:
    - Reading JSONL format (not JSON array)
    - Mapping: instance_id → question_id (used as idx)
    - Keeping db_path, gold_result_csv as-is (already added by setup script)
    - Validating all required files exist

    Args:
        dataset_path: Path to the Spider2-Lite dataset JSONL file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format.
    """
    import os

    samples = []

    # Load JSONL format (one JSON object per line)
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)

    # Convert Spider2-Lite format to NL2DSL format
    for sample in samples:
        # Map instance_id to both question_id and idx
        instance_id = sample.get("instance_id")
        sample["question_id"] = instance_id
        sample["idx"] = instance_id

        # Rename question to nl_query
        if "question" in sample and "nl_query" not in sample:
            sample["nl_query"] = sample["question"]

        # Set sql_query to empty string (we use pre-computed results, not gold SQL)
        # This prevents errors in code expecting sql_query field
        if "sql_query" not in sample:
            sample["sql_query"] = ""

        # Validate required fields exist
        assert "db_path" in sample, f"Missing db_path for {instance_id}"
        assert "gold_result_csv" in sample, f"Missing gold_result_csv for {instance_id}"

        # Validate files exist
        assert os.path.exists(sample["db_path"]), f"Database not found: {sample['db_path']}"
        assert os.path.exists(
            sample["gold_result_csv"]
        ), f"Gold result CSV not found: {sample['gold_result_csv']}"

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples


def _extract_llmsql_table_to_sqlite(
    source_db_path: Path, table_name: str, output_path: Path
) -> None:
    """Extract a single table from LLMSQL monolithic DB to individual SQLite file.

    Args:
        source_db_path: Path to the source SQLite database with all tables
        table_name: Name of the table to extract (from db_id)
        output_path: Path where the new SQLite file will be created
    """
    # 1. Check if already exists (caching)
    if output_path.exists():
        # Validate it has the correct table
        try:
            conn = sqlite3.connect(output_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            conn.close()
            if tables and tables[0][0] == table_name:
                return  # Already extracted, use cache
        except sqlite3.Error:
            # File exists but is corrupted, will be recreated
            pass

    # 2. Connect to source database
    source_conn = sqlite3.connect(source_db_path)

    try:
        # 3. Get CREATE TABLE statement
        create_result = source_conn.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        ).fetchone()

        if not create_result:
            source_conn.close()
            raise ValueError(f"Table '{table_name}' not found in {source_db_path}")

        create_stmt = create_result[0]

        # 4. Get all data
        rows = source_conn.execute(f"SELECT * FROM '{table_name}'").fetchall()

        source_conn.close()

        # 5. Create new SQLite file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_conn = sqlite3.connect(output_path)

        # 6. Create table and insert data
        target_conn.execute(create_stmt)
        if rows:
            placeholders = ",".join(["?"] * len(rows[0]))
            target_conn.executemany(
                f"INSERT INTO '{table_name}' VALUES ({placeholders})", rows
            )

        target_conn.commit()
        target_conn.close()
    except Exception as e:
        source_conn.close()
        raise


def load_llmsql_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from LLMSQL dataset format.

    LLMSQL format is a JSON array with normalized fields:
    - question_id: unique question identifier
    - db_id: table identifier
    - nl_query: natural language question
    - sql_query: gold SQL query
    - db_path: path to SQLite database (same for all questions)

    This loader dynamically extracts individual table SQLite files for only
    the questions being evaluated, avoiding loading all 25,609 tables.

    Args:
        dataset_path: Path to the LLMSQL dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries in NL2DSL format with updated db_paths
        pointing to per-table SQLite files.
    """
    with open(dataset_path, "r") as f:
        samples = json.load(f)

    # Convert LLMSQL format to NL2DSL format (mostly passthrough since already normalized)
    for i, sample in enumerate(samples):
        # Add index if not present
        if "idx" not in sample:
            sample["idx"] = i

    # Sample if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    # Dynamically generate per-table SQLite files for sampled questions
    if samples:
        source_db_path = Path(samples[0]["db_path"])  # Original monolithic DB
        cache_dir = source_db_path.parent / "tables"

        for sample in samples:
            db_id = sample["db_id"]
            # Use flat "tables" directory so Snowflake schema becomes "LLMSQL"."TABLES"
            # All LLMSQL tables share one schema: tables/{db_id}.sqlite
            output_path = cache_dir / f"{db_id}.sqlite"

            # Extract table to individual file (cached if exists)
            _extract_llmsql_table_to_sqlite(source_db_path, db_id, output_path)

            # Update sample to point to new file
            sample["db_path"] = str(output_path)

    return samples


def load_nl2dsl_dataset(
    dataset_path: str, sample_size: Optional[int] = None, random_state: int = 0
) -> List[dict]:
    """Load samples from NL2DSL dataset format.

    Args:
        dataset_path: Path to the NL2DSL dataset JSON file.
        sample_size: Number of samples to load. If None, loads all.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of dictionaries, each containing a complete sample with
        fields like 'idx', 'schema', 'sql_query', 'ibis_code', 'substrait_plan', etc.

    Example:
        >>> samples = load_nl2dsl_dataset('/path/to/dataset.json', sample_size=5)
        >>> sample = samples[0]
        >>> print(f"Sample {sample['idx']}: {sample['nl_query'][:50]}...")
    """
    # Auto-detect dataset format
    if "llmsql" in dataset_path.lower():
        return load_llmsql_dataset(dataset_path, sample_size, random_state)
    elif "archer" in dataset_path.lower():
        return load_archer_dataset(dataset_path, sample_size, random_state)
    elif "minidev" in dataset_path.lower():
        # MINIDEV has three variants: sqlite, mysql, postgresql
        if "mysql" in dataset_path.lower():
            return load_bird_minidev_native(dataset_path, sample_size, random_state, db_type="mysql")
        elif "postgresql" in dataset_path.lower() or "postgres" in dataset_path.lower():
            return load_bird_minidev_native(dataset_path, sample_size, random_state, db_type="postgres")
        else:
            # SQLite variant uses standard BIRD loader
            return load_bird_dataset(dataset_path, sample_size, random_state)
    elif "BIRD" in dataset_path or "bird" in dataset_path.lower():
        return load_bird_dataset(dataset_path, sample_size, random_state)
    elif "beaver" in dataset_path.lower():
        return load_beaver_dataset(dataset_path, sample_size, random_state)
    elif "spider2" in dataset_path.lower():
        return load_spider2_lite_dataset(dataset_path, sample_size, random_state)
    elif "spider" in dataset_path.lower():
        return load_spider_dataset(dataset_path, sample_size, random_state)

    samples = []

    with open(dataset_path, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            samples.append(sample)

    # Convert to DataFrame for sampling if needed
    if sample_size and sample_size < len(samples):
        df = pd.DataFrame(samples)
        df_sampled = df.sample(sample_size, random_state=random_state)
        samples = df_sampled.to_dict("records")

    return samples
