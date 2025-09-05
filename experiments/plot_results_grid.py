"""Plot results from a grid search experiment.

Looks up all pkl files in the results directory and generates plots for
each of them.
"""

import argparse
import json
import os
import pickle as pkl

from modular_query.plot_utils import plot_results, plot_results_across_graph_sizes


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
            f"plot_{config['variant']}_{config['run_id']}.png",
            save_dir=results_dir,
            title=title,
        )


# Make the plot for varying variants..
def plot_results_grid_query_algorithms(results_dir: str) -> None:
    """Plot results from a grid search experiment for varying query
    algorithms."""
    # Get all the run IDs that exist in the results directory.
    # (all pkl files have the form results_variant_[variant_name]_run_[run_id].pkl)
    # Print files that end with .pkl.
    run_ids = [f.split("_")[4] for f in os.listdir(results_dir) if f.endswith(".pkl")]
    # strip out the .pkl extension.
    run_ids = [run_id.split(".")[0] for run_id in run_ids]
    module_selector = "Brute Force"
    for run_id in run_ids:
        # Load all pkl and json files associated with this run_id.
        # print(run_id)
        pkl_files = [
            f
            for f in os.listdir(results_dir)
            if f.endswith(".pkl") and f.split("_")[4] == f"{run_id}.pkl"
        ]
        # print(pkl_files)
        config_files = [
            f
            for f in os.listdir(results_dir)
            if f.endswith(".json") and f.split("_")[4] == f"{run_id}.json"
        ]
        # print(config_files)
        # open one of the config files to get common parameters.
        with open(
            os.path.join(results_dir, config_files[0]), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
        title = (
            f"Grid Search: Module selector={module_selector},"
            + f" Num Failures={config['num_failures']},"
            + f" Correct Confidence={config['correct_confidence']},"
            + f" Incorrect Confidence={config['incorrect_confidence']},"
            + f" Redundancy={config['redundancy']},"
            + f" Query Cost={config['c_query']}"
        )
        # Pass the pickle files to plot_results_across_graph_sizes.
        plot_results_across_graph_sizes(
            algorithm=module_selector,
            pkl_files=pkl_files,
            data_dir=results_dir,
            filename=f"plot_{run_id}.png",
            title=title,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, default="experiments/results", required=True
    )
    args = parser.parse_args()
    # plot_results_grid(args.results_dir)
    plot_results_grid_query_algorithms(args.results_dir)
