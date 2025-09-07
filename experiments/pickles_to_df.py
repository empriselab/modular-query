"""Script which takes in a directory containing data pickle files and
corresponding config.json files, and produces a single pandas dataframe with
all of the data."""

import argparse
import json
import os
import pickle as pkl

import pandas as pd


def main(data_dir: str) -> None:
    """Main function to convert pickle files to a dataframe."""

    # The dataframe will have the following columns:
    # run_id, variant, num_failures, correct_confidence, incorrect_confidence,
    # dependency_structure, c_query, results_dictionary.

    # Loop through all of the config files, and populate the df one row at a time.
    df = pd.DataFrame(
        columns=[
            "run_id",
            "variant",
            "num_failures",
            "correct_confidence",
            "incorrect_confidence",
            "dependency_structure",
            "c_query",
            "results_dictionary",
        ]
    )
    for config_file in os.listdir(data_dir):
        if config_file.endswith(".json"):
            print(f"Processing {config_file}")
            with open(os.path.join(data_dir, config_file), "r", encoding="utf-8") as f:
                config = json.load(f)
            # infer the name of the pickle file from the config file.
            pickle_file = config_file.replace(".json", ".pkl")
            pickle_file = pickle_file.replace("config", "results")
            with open(os.path.join(data_dir, pickle_file), "rb") as f:
                results = pkl.load(f)
            # construct the dictionary to go into the dataframe.
            dict_to_add = {
                "run_id": config["run_id"],
                "variant": config["variant"],
                "num_failures": config["num_failures"],
                "correct_confidence": config["correct_confidence"],
                "incorrect_confidence": config["incorrect_confidence"],
                "dependency_structure": config["dependency_structure"],
                "c_query": config["c_query"],
                "results_dictionary": results,
            }
            # Use pd.concat instead of deprecated df.append
            new_row = pd.DataFrame([dict_to_add])
            df = pd.concat([df, new_row], ignore_index=True)

    # Save df to pickle file.
    df.to_pickle(os.path.join(data_dir, "combined_df.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir)
