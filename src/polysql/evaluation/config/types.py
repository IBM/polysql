from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment."""

    model_name: str
    model_engine: str
    gen_type: str
    dataset_path: str
    dataset_name: str = ""  # Dataset identifier (bird, beaver, archer, etc.)
    n_examples: int
    instructions_level: int = 22
    experiment_id: str = ""  # Set default to empty string

    @field_validator("instructions_level")
    @classmethod
    def validate_instructions_level(cls, v: int) -> int:
        """Validate instruction level format.

        Valid two-digit levels: 11, 12, 13, 21, 22, 23
        Format: XY where X=COT level (1-2), Y=dialect level (1-3)
        """
        valid_levels = [11, 12, 13, 21, 22, 23]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid instructions_level: {v}. "
                f"Valid levels: {valid_levels}. "
                f"Format: XY where X=COT level (1-2), Y=dialect level (1-3)"
            )
        return v

    @model_validator(mode="after")
    def set_experiment_id(self) -> "ExperimentConfig":
        """Generate experiment ID after initialization if not provided."""
        # This validator runs after all fields are populated
        # and mimics the logic from __post_init__
        if not self.experiment_id:
            # Include dataset_name if provided
            if self.dataset_name:
                self.experiment_id = f"{self.model_name}_{self.gen_type}_{self.dataset_name}_{self.instructions_level}"
            else:
                self.experiment_id = (
                    f"{self.model_name}_{self.gen_type}_{self.instructions_level}"
                )
        return self


class ExperimentSummary(BaseModel):
    """Summary of a single experiment result."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(..., description="Unique experiment identifier")
    model_name: str = Field(..., description="Model name")
    model_engine: str = Field(..., description="Model engine/provider")
    gen_type: str = Field(..., description="SQL dialect or generation type")
    instructions_level: int = Field(
        ..., description="Level of instructions used (two-digit format XY)"
    )
    total: int = Field(..., description="Total number of examples")
    generated: int = Field(..., description="Number of successful generations")
    generation_failed: int = Field(..., description="Number of failed generations")
    executed: int = Field(..., description="Number of successful executions")
    execution_failed: int = Field(..., description="Number of failed executions")
    correct: int = Field(..., description="Number of correct results")
    accuracy: float = Field(..., description="Accuracy (correct / executed)")
    status: str = Field(..., description="Experiment status (success/failed)")
    error: Optional[str] = Field(None, description="Error message if failed")


class PredictionResult(BaseModel):
    """Result for a single prediction."""

    question_id: int = Field(..., description="Question identifier")
    db_id: str = Field(..., description="Database identifier")
    question: str = Field(..., description="Natural language question")
    gold_sql: str = Field(..., description="Gold SQL query")
    predicted_sql: str = Field(..., description="Predicted SQL query")
    predicted_code: Optional[str] = Field(
        None,
        description="Raw predicted code (Ibis/Substrait) before conversion",
    )
    both_executed: bool = Field(..., description="Whether both queries executed")
    results_equal: Optional[bool] = Field(None, description="Whether results are equal")
    gold_error: Optional[str] = Field(None, description="Error from gold query")
    pred_error: Optional[str] = Field(None, description="Error from predicted query")
    gold_result: Optional[List[Dict[str, Any]]] = Field(
        None, description="Gold query result data as list of records"
    )
    predicted_result: Optional[List[Dict[str, Any]]] = Field(
        None, description="Predicted query result data as list of records"
    )
    full_prompt: Optional[str] = Field(
        None, description="Full prompt used for generation"
    )


class EvaluationResult(BaseModel):
    """Result model for evaluation loop."""

    comment: Optional[str] = Field(
        "This is a Cross-DB Evaluation setup: gold query is executed on a different DB than the predicted so their dialect may not be the same, their semantics is what matters.",
        description="Optional comment about the evaluation",
    )
    exp_config: Optional[ExperimentConfig] = Field(
        ..., description="Experiment configuration"
    )
    predictions: List[PredictionResult] = Field(
        ..., description="List of prediction results"
    )
    errors: List[str] = Field(default_factory=list, description="List of errors")
    total: int = Field(..., description="Total number of examples")
    generated: int = Field(..., description="Number of successful generations")
    generation_failed: int = Field(..., description="Number of failed generations")
    executed: int = Field(..., description="Number of successful executions")
    execution_failed: int = Field(..., description="Number of failed executions")
    correct: int = Field(..., description="Number of correct results")
    accuracy: float = Field(..., description="Accuracy (correct / total)")
