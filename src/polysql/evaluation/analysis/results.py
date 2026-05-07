import json
from argparse import ArgumentParser
from pathlib import Path  # Use pathlib for modern, object-oriented path handling

import pandas as pd


def load_experiment_data(exp_dir: Path) -> pd.DataFrame:
    """
    Loads all experiment JSON files from a directory into a DataFrame.

    Each DataFrame will contain the 'predictions' data, with the
    'exp_config' data broadcast as columns to every row. Dataset information
    (dataset_name, dataset_path) is included from exp_config.

    If experiments used BIRD dataset with question_id and difficulty metadata
    is available, it will be merged in.

    If an enriched results file exists (all_results_enriched.csv), error
    classifications will be merged in.
    """
    json_files = [f for f in exp_dir.glob("*.json") if not f.name.startswith("summary")]

    dataframes = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        config = data.get("exp_config", {})
        predictions = data.get("predictions", [])

        if not predictions:
            continue

        records = [{**pred, **config} for pred in predictions]
        dataframes.append(pd.DataFrame(records))

    dfs = pd.concat(dataframes, ignore_index=True)

    if "question_id" in dfs.columns:
        bird_metadata_path = Path("data/BIRD/dev_20240627/dev.json")
        if bird_metadata_path.exists():
            try:
                metadata = pd.read_json(bird_metadata_path)[
                    ["question_id", "difficulty"]
                ]
                dfs = dfs.merge(metadata, on="question_id", how="left")
            except (KeyError, ValueError):
                pass
    dfs["results_equal"] = dfs["results_equal"].convert_dtypes().fillna(False)

    # Merge enriched data if available
    enriched_path = exp_dir / "all_results_enriched.csv"
    if enriched_path.exists():
        enriched_df = pd.read_csv(enriched_path)
        # Merge error_classification, explanation, and sqlite_correct columns
        if "error_classification" in enriched_df.columns:
            merge_cols = ["question_id", "db_id", "model_name", "gen_type"]
            enrich_cols = ["error_classification", "explanation"]
            # Add sqlite_correct if it exists
            if "sqlite_correct" in enriched_df.columns:
                enrich_cols.append("sqlite_correct")
            # Keep only needed columns for merge
            enriched_subset = enriched_df[merge_cols + enrich_cols].copy()
            dfs = dfs.merge(enriched_subset, on=merge_cols, how="left")

    return dfs


def main():
    """Main script logic."""
    parser = ArgumentParser(description="Load experiment results from a directory.")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="2025-11-06_17-07-14_reverent_colden",
        help="Path to the experiment results directory.",
    )
    args = parser.parse_args()

    # Convert the string path to a Path object
    exp_path = Path(args.exp_dir)

    if not exp_path.is_dir():
        print(f"Error: Directory not found at {exp_path}")
        return

    dfs = load_experiment_data(exp_path)
    # dfs.drop(
    #     columns=[
    #         "model_engine",
    #         "n_examples",
    #         "db_id",
    #         "instructions_level",
    #         "question_id",
    #         "instructions_level",
    #         "experiment_id",
    #         "dataset_path"
    #     ],
    #     inplace=True,
    # )

    dfs.to_csv(exp_path / "all_results.csv")


if __name__ == "__main__":
    main()
