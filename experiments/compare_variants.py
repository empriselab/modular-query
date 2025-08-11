"""For the three variants (balanced, greedy, conservative), plot results of
experiment for a fixed number of failures, fixed graph size, and a fixed
algorithm (let's say, brute force).

Will create a parallel bar chart with 4 metrics:
1. Timesteps
2. Executions
3. Total Query Cost
4. Total Task Cost
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    num_failures: int, graph_size: int, algorithm: str, data_dir: str
) -> None:
    """Plot results for a fixed number of failures, graph size, and
    algorithm."""
    # Load results for each variant.
    # (pickle files created by run_experiment.py)
    variants = ["greedy", "balanced", "conservative", "balanced-2"]

    balanced_results = pkl.load(
        open(
            f"{data_dir}/"
            f"strategy_comparison_num_failures_balanced_{num_failures}.pkl",
            "rb",
        )
    )
    greedy_results = pkl.load(
        open(
            f"{data_dir}/"
            f"strategy_comparison_num_failures_greedy_{num_failures}.pkl",
            "rb",
        )
    )
    conservative_results = pkl.load(
        open(
            f"{data_dir}/"
            f"strategy_comparison_num_failures_conservative_{num_failures}.pkl",
            "rb",
        )
    )
    balanced_2_results = pkl.load(
        open(
            f"{data_dir}/"
            f"strategy_comparison_num_failures_balanced-2_{num_failures}.pkl",
            "rb",
        )
    )
    # Plot results.
    # Create a parallel bar chart with 4 metrics:
    # 1. Timesteps
    # 2. Executions
    # 3. Total Query Cost
    # 4. Total Task Cost
    # Where we have 4 group of num_variants bars each (one for each variant).
    metrics = [
        "total_timesteps",
        "total_executions",
        "total_queries",
        "total_task_cost",
    ]

    # Create a figure and axis.
    _, ax = plt.subplots()

    # For each variant, we'll create a separate bar plot for each metric
    # This allows us to use variant names in the legend
    variant_results = [
        greedy_results,
        balanced_results,
        conservative_results,
        balanced_2_results,
    ]

    x_positions = np.arange(len(metrics)) * 2
    width = 0.25  # Width of each bar

    for i, (variant, results) in enumerate(zip(variants, variant_results)):
        means = [np.mean(results[algorithm][metric][graph_size]) for metric in metrics]
        stds = [np.std(results[algorithm][metric][graph_size]) for metric in metrics]
        ax.bar(
            x_positions + i * width, means, width, yerr=stds, capsize=5, label=variant
        )

    # Add title.
    ax.set_title(
        f"Variant Comparison for {algorithm} "
        f"with {num_failures} Failures and Graph Size {graph_size}"
    )
    # X-axis labels are the different metrics.
    # Set x-ticks to be centered on each group of bars
    ax.set_xticks(x_positions + width)  # Center of the group (after 1.5 bars)
    ax.set_xticklabels(metrics)

    # Add legend.
    ax.legend()

    # Save the figure.
    plt.savefig(
        f"{data_dir}/"
        f"strategy_comparison_num_failures_{num_failures}_"
        f"graph_size_{graph_size}_algorithm_{algorithm}.png"
    )


def main(failures_list: list, graph_size: int, algorithm: str, data_dir: str) -> None:
    """Main function to plot results for a fixed number of failures, graph
    size, and algorithm."""
    for num_failures in failures_list:
        plot_results(num_failures, graph_size, algorithm, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures_list", nargs="+", type=int, required=True)
    parser.add_argument("--graph_size", type=int, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.failures_list, args.graph_size, args.algorithm, args.data_dir)
