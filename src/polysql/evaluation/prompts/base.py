from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

ChatMessageType = Dict[str, str]
"""A single message in a chat, e.g., {"role": "user", "content": "..."}."""

PromptType = List[ChatMessageType]
"""A full chat prompt, which is a list of messages."""

_EXAMPLE_REMOVAL_REGEX = re.compile(r"\s*--\s*example:.*$", re.IGNORECASE)


def remove_examples_from_schema(sql_schema: str) -> str:
    """
    Cleans a SQL schema string by removing inline example comments while
    preserving structural information such as table definitions or constraints.
    """
    cleaned_lines: List[str] = []
    for line in sql_schema.strip().split("\n"):
        cleaned_line = _EXAMPLE_REMOVAL_REGEX.sub("", line)
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)


def get_prompt_for_code_generation(
    schema: str,
    question: str,
    gen_type: str,
    instruction_level: int = 1,
    prompt_override: Optional[str] = None,
    prompt_template_override: Optional[str] = None,
) -> str:
    """
    Return the appropriate SQL or Ibis prompt for the requested generation type.

    Set ``prompt_override`` to return a fully rendered prompt string, or
    ``prompt_template_override`` to provide a custom template (using the
    ``{question}`` and ``{schema}`` placeholders) for Ibis generation.
    """
    from polysql.evaluation.prompts.sql import SQL_GEN_TYPES, get_sql_prompt

    if prompt_override is not None:
        return prompt_override

    # Handle sqlite-{dialect} pattern (e.g., sqlite-postgres, sqlite-duckdb)
    if gen_type in SQL_GEN_TYPES or gen_type.startswith("sqlite-"):
        return get_sql_prompt(schema, question, gen_type, instruction_level)

    all_known_types = sorted(list(SQL_GEN_TYPES))
    raise ValueError(
        f"Unsupported gen_type: '{gen_type}'. Supported types are: {all_known_types}. "
        f"Also supported: 'sqlite-{{dialect}}' patterns (e.g., sqlite-postgres, sqlite-duckdb) "
        f"for transpilation experiments."
    )


def get_prompts(
    gen_type: str,
    dataset: List[Dict[str, Any]],
    instruction_level: int = 22,
    prompt_override: Optional[str] = None,
    prompt_template_override: Optional[str] = None,
    schema_getter: Optional[Callable[[Dict[str, Any], str], str]] = None,
    ibis_schema_getter: Optional[Callable[[Dict[str, Any], str], str]] = None,
) -> List[PromptType]:
    """Produce chat prompts for a batch of instances.

    Per-instance overrides can be supplied via the dataset keys
    ``prompt_override`` or ``prompt_template_override``; global overrides are
    available through the corresponding function parameters.

    Args:
        gen_type: Type of code to generate (SQL, Ibis, or Substrait).
        dataset: List of dataset instances.
        instruction_level: Two-digit level XY where X=COT (1-2), Y=dialect (1-3).
            Valid levels: 11, 12, 13, 21, 22, 23. Default: 22.
        prompt_override: Global prompt override.
        prompt_template_override: Global template override.
        schema_getter: Function to get SQL/Substrait schemas dynamically.
        ibis_schema_getter: Function to get Ibis schemas dynamically.
    """
    # Determine which schema getter to use based on gen_type
    if "ibis" in gen_type:
        # Use ibis_schema_getter for Ibis generation
        active_schema_getter = ibis_schema_getter
    else:
        # Use schema_getter for SQL dialects and Substrait
        active_schema_getter = schema_getter

    prompts: List[PromptType] = []
    for instance in dataset:
        question = instance["nl_query"]
        schema = active_schema_getter(instance, gen_type)

        instance_prompt_override = instance.get("prompt_override") or prompt_override
        instance_template_override = (
            instance.get("prompt_template_override") or prompt_template_override
        )
        prompt = get_prompt_for_code_generation(
            schema=schema,
            question=question,
            gen_type=gen_type,
            instruction_level=instruction_level,
            prompt_override=instance_prompt_override,
            prompt_template_override=instance_template_override,
        )
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


__all__ = [
    "ChatMessageType",
    "PromptType",
    "get_prompt_for_code_generation",
    "get_prompts",
    "remove_examples_from_schema",
]
