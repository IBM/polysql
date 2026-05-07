import copy
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from dotenv import load_dotenv
from unitxt.inference import CrossProviderInferenceEngine, ListWithMetadata
from unitxt.loaders import LoadFromDictionary

from polysql.evaluation.prompts.base import PromptType
from polysql.evaluation.utils.cache_paths import (
    ensure_inference_cache_env,
    migrate_legacy_caches,
)

load_dotenv()
migrate_legacy_caches()
ensure_inference_cache_env()

GenType = Literal[
    "sqlite",
    "duckdb",
    "postgres",
    "snowflake",
    "bigquery",
    "datafusion",
    "pyspark",
    "mysql",
    "substrait",
    "ibis",
]

SUPPORTED_DIALECTS: list = [
    "sqlite",
    "sqlite-ss",
    "duckdb",
    "postgres",
    "snowflake",
    "bigquery",
    "datafusion",
    "pyspark",
    "mysql",
    "clickhouse",
    "substrait",
]


class CrossProviderInferenceEngineWithMoreRISTModels(CrossProviderInferenceEngine):
    """
    Extends CrossProviderInferenceEngine to include additional RITS models.

    This class modification is defined at the module level to avoid
    re-defining and deep-copying the provider map on every
    NL2DSLModel instantiation.
    """

    # Create a deep copy once at the class level
    provider_model_map = copy.deepcopy(CrossProviderInferenceEngine.provider_model_map)

    # Update the copy with new models
    provider_model_map["rits"].update(
        {
            "devstral-small-2507": "mistralai/Devstral-Small-2507",
            "deepseek-coder-33b-instruct": "deepseek-ai/deepseek-coder-33b-instruct",
            "qwen3-8b": "Qwen/Qwen3-8B",
            "llama-4-scout-17b-16e-instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "deepseek-v32": "deepseek-ai/DeepSeek-V3.2",
            "deepseek-v2-5": "deepseek-ai/DeepSeek-V2.5",
            "llama-3-1-405b-instruct-fp8": "meta-llama/llama-3-1-405b-instruct-fp8",
            "mistral-large-3-675b-2512": "mistralai/Mistral-Large-3-675B-Instruct-2512",
            "mistral-small-3-2-24b-2506": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "qwen2-5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        }
    )

    # Add custom models through LiteLLM proxy (use openai/ prefix for proxy routing)
    # The proxy expects model names like "Azure/gpt-4.1-nano" but LiteLLM needs "openai/" prefix
    # to recognize it as an OpenAI-compatible endpoint
    provider_model_map["open-ai"].update(
        {
            # Azure models through proxy
            "gpt-4.1-nano": "openai/Azure/gpt-4.1-nano",
            "gpt-4.1-mini": "openai/Azure/gpt-4.1-mini",
            "gpt-4.1": "openai/Azure/gpt-4.1",
            "gpt-4o-azure": "openai/Azure/gpt-4o",
            "gpt-5": "openai/Azure/gpt-5-2025-08-07",
            "gpt-5-mini": "openai/Azure/gpt-5-mini-2025-08-07",
            "gpt-5-nano": "openai/Azure/gpt-5-nano-2025-08-07",
            "o1": "openai/Azure/o1",
            "o1-mini": "openai/Azure/o1-mini",
            "o3-mini": "openai/Azure/o3-mini",
            # AWS Bedrock models through proxy
            "claude-sonnet-4-5": "openai/aws/claude-sonnet-4-5",
            "claude-haiku-4-5": "openai/aws/claude-haiku-4-5",
            # GCP/Vertex AI models through proxy
            "claude-3-5-haiku": "openai/GCP/claude-3-5-haiku",
            "gemini-2.0-flash": "openai/GCP/gemini-2.0-flash",
            "gemini-2.0-flash-lite": "openai/GCP/gemini-2.0-flash-lite",
            "gemini-1.5-pro": "openai/GCP/gemini-1.5-pro",
        }
    )


class NL2DSLModel:
    """
    A wrapper for an LLM inference engine specialized for NL-to-DSL tasks.

    This class handles prompt generation, model inference via unitxt,
    and optional transpilation of generated code (e.g., Ibis to SQL).
    """

    def __init__(
        self,
        model_name_or_path: str,
        gen_type: GenType = "ibis",
        model_engine: str = "rits",
        stop_token_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the NL2DSLModel.

        Args:
            model_name_or_path: The name or path of the model to use
                                (e.g., "devstral-small-2507").
            gen_type: The type of code to generate ("ibis", "sql", etc.).
            model_engine: The inference engine provider (e.g., "rits").
            stop_token_ids: An optional list of token IDs to use as
                            stop sequences during generation.
        """
        self.gen_type: GenType = gen_type
        self.model_name_or_path: str = model_name_or_path
        self.model_engine: str = model_engine
        self.stop_token_ids: List[int] = (
            stop_token_ids if stop_token_ids is not None else []
        )

        # Type hint for the inference function
        self.inference_fn: Callable[[List[PromptType]], ListWithMetadata] = (
            self.api_completion
        )

        # Use the module-level extended class
        self.inference_model = CrossProviderInferenceEngineWithMoreRISTModels(
            model=self.model_name_or_path,
            provider=self.model_engine,
            data_classification_policy=["public"],
            max_tokens=1024,
            temperature=0.0,
        )

    def api_completion(self, prompts: List[PromptType]) -> ListWithMetadata:
        """
        Performs API completion for a list of chat prompts.

        Args:
            prompts: A list of chat prompts (List[List[Dict[str, str]]]).

        Returns:
            A ListWithMetadata object containing the model's predictions.
        """
        dataset = (
            LoadFromDictionary(
                data={"test": [{"source": prompt} for prompt in prompts]},
                data_classification_policy=["public"],
            )
            .process()
            .to_dataset()
        )

        return self.inference_model(dataset["test"])

    def __call__(self, prompts: List[PromptType]) -> ListWithMetadata:
        """
        Alias for self.api_completion.

        Args:
            prompts: A list of chat prompts.

        Returns:
            A ListWithMetadata object containing the model's predictions.
        """
        return self.inference_fn(prompts)

    def cleanup(self) -> None:
        """
        Clean up resources to help prevent memory leaks.

        Sets the inference model to None to release references.
        """
        self.inference_model = None
        import gc

        gc.collect()


def run_batch_inference(
    model_name: str,
    model_engine: str,
    gen_type: GenType,
    dataset: List[Dict[str, Any]],
) -> Tuple[ListWithMetadata, float, Optional[str]]:
    """
    Run batch inference for a model on a dataset.

    Args:
        model_name: Model name or path.
        model_engine: Model engine (e.g., "rits").
        gen_type: Generation type ("ibis" or "sql").
        dataset: List of dataset samples, each a dict with "nl_query"
                 and schema.

    Returns:
        A tuple of:
        (predicted_codes, inference_time, error_message)
        - predicted_codes: ListWithMetadata (a list of generated strings)
                           or an empty ListWithMetadata on error.
        - inference_time: Time taken for inference in seconds.
        - error_message: A string error message if an exception occurred,
                         otherwise None.
    """
    model = NL2DSLModel(
        model_name_or_path=model_name,
        gen_type=gen_type,
        model_engine=model_engine,
        stop_token_ids=[],  # Pass any stop tokens if needed
    )

    # Generate prompts
    prompts: List[PromptType] = model.get_prompts(dataset=dataset)

    # Run inference
    start_time = time.time()
    try:
        predicted_codes: ListWithMetadata = model(prompts=prompts)
        inference_time = time.time() - start_time
        return predicted_codes, inference_time, None
    except Exception as e:
        inference_time = time.time() - start_time
        # Return an empty ListWithMetadata for type consistency
        return ListWithMetadata(), inference_time, str(e)
    finally:
        # Ensure cleanup to release model from memory
        model.cleanup()


# if __name__ == "__main__":
#     # Example of how to run (requires a dataset)
#
#     # simple_evaluation_loop() # From original code
#
#     # Example dataset
#     my_dataset = [
#         {
#             "nl_query": "Show me all tables",
#             "ibis_schema": "import ibis\n\nt = ibis.table(name='my_table', schema={'col1': 'string', 'col2': 'int64'})\n",
#             "sql_schema": "CREATE TABLE my_table (col1 VARCHAR, col2 BIGINT);",
#         }
#     ]
#
#     preds, inf_time, err = run_batch_inference(
#         model_name="devstral-small-2507",
#         model_engine="rits",
#         gen_type="ibis",
#         dataset=my_dataset
#     )
#
#     if err:
#         print(f"Inference failed after {inf_time:.2f}s: {err}")
#     else:
#         print(f"Inference complete in {inf_time:.2f}s")
#         print("Predictions:")
#         for p in preds:
#             print(p)
