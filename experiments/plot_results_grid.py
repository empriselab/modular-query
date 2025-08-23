"""Plot results from a grid search experiment.

Looks up all pkl files in the results directory and generates plots for
each of them.
"""

import argparse
import json
import os
import pickle as pkl

from modular_query.plot_utils import plot_results


def plot_results_grid(results_dir: str) -> None:
    """Plot results from a grid search experiment."""
    # Look up all pkl files in the results directory.
    pkl_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
    for pkl_file in pkl_files:
        print(f"Plotting {pkl_file}")
        # Load pickle file
        with open(os.path.join(results_dir, pkl_file), "rb") as f:
            results = pkl.load(f)
        # Load the config from the json file.
        # Same name as pkl file, but (1) with .json extension,
        # and (2) with 'config' instead of 'results'.
        with open(
            os.path.join(
                results_dir,
                pkl_file.replace(".pkl", ".json").replace("results", "config"),
            ),
            "r",
            encoding="utf-8",
        ) as f2:
            config = json.load(f2)
        metrics = [
            "query_cost_total",
            "total_failed_attempts",
            "execution_time_total",
            "total_correct",
            "total_timesteps",
        ]

        # infer graph sizes from results.
        graph_sizes = list(results["Brute Force"]["total_timesteps"].keys())
        # Create a title based on the following config keys:
        # variant, num_failures, correct_confidence, incorrect_confidence,
        # redundancy, c_query
        title = (
            f"Grid Search: Variant={config['variant']},"
            + f"Num Failures={config['num_failures']},"
            + f"Correct Confidence={config['correct_confidence']},"
            + f"Incorrect Confidence={config['incorrect_confidence']},"
            + f"Redundancy={config['redundancy']},"
            + f"Query Cost={config['c_query']}"
        )
        plot_results(
            results,
            graph_sizes,
            metrics,
            f"plot_{config['run_id']}.png",
            save_dir=results_dir,
            title=title,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, default="experiments/results", required=True
    )
    args = parser.parse_args()
    plot_results_grid(args.results_dir)
