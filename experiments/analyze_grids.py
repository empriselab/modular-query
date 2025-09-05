"""Checks to see whether config.json files line up with the expected set of
configurations (to debug any missing configurations from the parallelized
experimental grid)."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from modular_query.utils import print_and_log


def main(results_dir: str):
    """Main function to look for missing grid search results."""
    # Get the directory of the current script
    config_dir = Path(results_dir)

    # Expected configuration (copied from run_experiment.py)
    expected_config: dict[str, Any] = {
        "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
        "num_trials": 100,
        "num_failures_list": [0, 1, 2, 3],
        "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
        "redundancy_list": ["AND", "OR"],
        "c_query_list": [0.08, 0.16, 0.32, 0.64],
    }

    combo_generator = list(
        itertools.product(
            ["conservative", "balanced", "balanced-2", "greedy"],
            expected_config["num_failures_list"],
            expected_config["confidences_list"],
            expected_config["redundancy_list"],
            expected_config["c_query_list"],
        )
    )

    # Get all config.json files in the directory
    # f.endswith() doesn't work because we're iterating over Path objects
    # so we need to convert to strings
    config_files = [f for f in config_dir.iterdir() if f.name.endswith(".json")]

    # Check if JSON files cover all expected configurations
    present_config_indices = []
    for config_file in config_files:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            # keys to look at: "variant", "num_failures",
            # "correct_confidence", "incorrect_confidence", "redundancy", "c_query"
            key = (
                config["variant"],
                config["num_failures"],
                (config["correct_confidence"], config["incorrect_confidence"]),
                config["redundancy"],
                config["c_query"],
            )

            # look up the index of the combo in the combo_generator
            index = combo_generator.index(key)
            present_config_indices.append(index)

    # Check if all expected configurations are present
    for i in range(len(combo_generator)):
        if i not in present_config_indices:
            print_and_log(f"Missing configuration: {combo_generator[i]}")
    print_and_log(f"Total number of configurations: {len(combo_generator)}")
    print_and_log(f"Number of configurations present: {len(present_config_indices)}")
    print_and_log(
        f"Number of missing configurations:"
        f"{len(combo_generator) - len(present_config_indices)}"
    )


if __name__ == "__main__":
    # Create argparse to take in results directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", required=True)
    args = parser.parse_args()
    main(args.results_dir)
