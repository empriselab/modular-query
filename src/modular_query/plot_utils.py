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
import json
import pickle as pkl
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TITLES = {
    "query_cost": "Query Cost",
    "query_cost_total": "Total Query Cost",
    "task_cost": "Final Task Cost",
    "total_cost": "Total Cost (Final Task + Query)",
    "proxy_obj_1": "Morphological Total Cost",
    "proxy_obj_2": "Morphological Total Cost",
    "execution_time": "Query Algorithm Runtime (s)",
    "execution_time_total": "Computation Time (s)",
    "mean_queries": "Mean Queries per Time Step",
    "total_queries": "Total Queries",
    "total_correct": "Total Successful Trials",
    "total_timesteps": "Total Timesteps",
    "total_executions": "Total Executions",
    "total_failed_attempts": "Total Failed Attempts",
}

YLABELS = {
    "query_cost": "Cost",
    "query_cost_total": "Query Cost",
    "task_cost": "Cost",
    "total_cost": "Cost",
    "proxy_obj_1": "Cost",
    "proxy_obj_2": "Cost",
    "execution_time": "Runtime (s)",
    "execution_time_total": "Computation Time (s)",
    "mean_queries": "Mean Queries",
    "total_queries": "Total Queries",
    "total_correct": "Total Correct",
    "total_timesteps": "Total Timesteps",
    "total_executions": "Total Executions",
    "total_failed_attempts": "Total Failed Attempts",
}

VARIANT_NAMES = {
    "greedy": "Execute-First",
    "balanced": "Query-Then-Execute",
    "conservative": "Query-Until-Confident",
    "balanced-2": "Query-Until-Confident-Workload-Aware",
    "query-all": "Query-For-All",
}

# Define distinct line styles, markers, and colors for each strategy
# Strategies (module selectors) are different shades of green.
STRATEGY_COLORS = {
    "Never Query": {
        "color": "#b2e2e2",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2,
    },
    "Brute Force": {
        "color": "#006d2c",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 2,
    },
    "Graph Query": {
        "color": "#2ca25f",
        "linestyle": "-",
        "marker": "x",
        "linewidth": 2,
    },
    "Binary Tree Query": {
        "color": "#66c2a4",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2,
    },
    "Confidence Query": {
        "color": "#984ea3",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2,
    },
    "MIP": {
        "color": "orange",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 2,
    },
    "Always Query": {
        "color": "blue",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2,
    },
}

# Variant-specific styles.
VARIANT_STYLES = {
    "greedy": {
        "color": "#fdcc8a",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2,
        "name": "Execute-First",
    },
    "balanced": {
        "color": "#fc8d59",
        "linestyle": "-",
        "marker": "x",
        "linewidth": 2,
        "name": "Query-Then-Execute",
    },
    "conservative": {
        "color": "#e34a33",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2,
        "name": "Query-Until-Confident",
    },
    "balanced-2": {
        "color": "#b30000",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 2,
        "name": "Query-Until-Confident-Workload-Aware",
    },
}


def common_add_arrows(
    ax: plt.Axes, xaxis_position: float = 0.0, y_lim: tuple[float, float] | None = None
) -> None:
    """Remove top and right spines, and add arrows at the end of the axes.

    Args:
        ax: The axes to modify
        xaxis_position: Position for the x-axis arrow
        (bottom y-limit if y_lim not provided)
        y_lim: Optional tuple (bottom, top) to set explicit y-axis limits
    """
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Draw arrows at the end of the axes. Sadly, you have to manually set this x-lim,
    # as automatically extracting it doesn't seem to work.
    yaxis_position = -0.6
    ax.set_xlim(left=yaxis_position)

    if y_lim is not None:
        # Set both bottom and top limits
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        arrow_y_position = y_lim[0]  # Arrow at bottom limit
    else:
        # Use the default behavior
        ax.set_ylim(bottom=xaxis_position)
        arrow_y_position = xaxis_position

    # Place arrows at the end of the axes
    ax.plot(
        1, arrow_y_position, ">k", transform=ax.get_yaxis_transform(), clip_on=False
    )
    ax.plot(yaxis_position, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)


def plot_results(
    results: dict[str, dict[str, dict[int, list[float]]]],
    graph_sizes: list[int],
    metrics_to_plot: list[str] | None = None,
    plot_name: str = "strategy_comparison.png",
    save_dir: str = "experiments/results",
    title: str = "",
    use_mean_for_total_correct: bool = False,
    tick_fontsize: int = 12,
    figsize: tuple[int, int] = (24, 12),
) -> None:
    """Create plots showing the performance of different querying strategies.

    Includes two proxy objectives:
    - Proxy objective 1 (uses 1 - product of confidences)
    - Proxy objective 2 (uses sum of uncertainties)

    Args:
        results: dictionary with results from run_experiment
        graph_sizes: list of graph sizes that were tested
    """
    # Set up metrics to plot.
    if metrics_to_plot is None:
        metrics = [
            "query_cost",
            "task_cost",
            "total_cost",
            "proxy_obj_1",
            "proxy_obj_2",
            "execution_time",
            "mean_queries",
            "total_queries",
            "total_correct",
            "total_timesteps",
            "total_executions",
        ]
    else:
        metrics = metrics_to_plot

    # Set up figure
    num_cols = 3
    num_rows = (len(metrics) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)

    # Plot each metric
    lines = []
    labels = []
    for i, metric in enumerate(metrics):
        ax = axes[i // num_cols][i % num_cols]

        for strategy_name in results:
            # Calculate medians, upper quartiles, and lower quartiles for each graph size
            medians: list[np.floating | float] = []
            upper_quartiles: list[np.floating | float] = []
            lower_quartiles: list[np.floating | float] = []
            for size in graph_sizes:
                try:
                    result_array = np.array(
                        results[strategy_name][metric][size], dtype=np.float64
                    )
                    if len(result_array) == 0:
                        continue
                    # Use mean for total_correct metric if requested,
                    # otherwise use median
                    if use_mean_for_total_correct and metric == "total_correct":
                        median = np.mean(result_array)
                        std = np.std(result_array)
                        upper_quartile = median + std
                        lower_quartile = median - std
                    else:
                        median = np.median(result_array)
                        upper_quartile, lower_quartile = np.percentile(
                            result_array, [75, 25]
                        )
                except KeyError:
                    continue
                except TypeError:
                    print(f"Dumping values: {result_array}")
                    raise
                medians.append(median)
                upper_quartiles.append(upper_quartile)
                lower_quartiles.append(lower_quartile)

            # Plot the data with strategy-specific styling
            style = STRATEGY_COLORS[strategy_name]
            line = ax.plot(
                graph_sizes[: len(medians)],
                medians,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=8,
                label=strategy_name,
            )

            # Plot the interquartile range band
            ax.fill_between(
                graph_sizes[: len(medians)],
                np.array(lower_quartiles),
                np.array(upper_quartiles),
                alpha=0.3,
                label="quartiles",
                color=style["color"],
            )

            # Only store lines and labels from the first subplot
            if i == 0:
                lines.append(line[0])
                labels.append(strategy_name)

        # Explicitly enable x-axis tick labels for all subplots (not just bottom)
        ax.tick_params(labelbottom=True)
        # Set x-tick label size (not y-tick label size)
        ax.tick_params(labelsize=tick_fontsize, axis="x")
        ax.set_ylabel(YLABELS[metric], fontsize=18, fontfamily="serif")

        common_add_arrows(ax, xaxis_position=-0.005)

    # Turn off unused subplots.
    for j in range(i + 1, num_rows * num_cols):
        axes[j // num_cols][j % num_cols].axis("off")

    # Add shared legend to the figure
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(results),
        frameon=True,
    )

    # Add overall title.
    if title:
        fig.suptitle(title)

    plt.tight_layout()

    # Add padding at the bottom for the legend
    plt.subplots_adjust(bottom=0.15)

    # Create directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)

    # Save the figure
    plt.savefig(Path(save_dir) / plot_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_results_variants_bar_chart(
    num_failures: int, graph_size: int, algorithm: str, data_dir: str
) -> None:
    """Create bar plot of results for a fixed number of failures, graph size,
    and algorithm."""
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
            x_positions + i * width,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=VARIANT_NAMES[variant],
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


def plot_results_across_graph_sizes(
    algorithm: str,
    pkl_files: list[str],
    data_dir: str,
    output_dir: str,
    title: str,
    filename: str,
    use_mean_for_total_correct: bool = False,
    tick_fontsize: int = 12,
    figsize: tuple[int, int] = (24, 12),
) -> None:
    """Plots where x-axis is the number of modules in the graph, and subplots
    correspond to different metrics. Each variant is a separate line.

    Similar to plot_results(), but for multiple variants, rather than a
    single variant.
    """
    variants = ["greedy", "balanced", "conservative", "balanced-2"]

    results: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = {
        variant: {} for variant in variants
    }

    for pkl_file in pkl_files:
        # if there are any forward slashes in the pkl file name, extract the last part.
        if "/" in pkl_file:
            tail = pkl_file.split("/")[-1]
            variant = tail.split("_")[2]
        else:
            # extract the variant from the pkl file name.
            variant = pkl_file.split("_")[2]
        results[variant] = pkl.load(open(Path(data_dir) / pkl_file, "rb"))

    # Extract the graph sizes from the results.
    graph_sizes = list(results["greedy"][algorithm]["total_timesteps"].keys())

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    # Create a figure and axis.
    num_cols = 3
    num_rows = (len(metrics) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)

    # Define distinct line styles, markers, and colors for each strategy

    # For each variant, we'll create a separate line plot for each metric
    # This allows us to use variant names in the legend
    lines = []
    labels = []
    for i, metric in enumerate(metrics):
        ax = axes[i // num_cols][i % num_cols]
        for variant in variants:
            medians: list[np.floating | float] = []
            upper_quartiles: list[np.floating | float] = []
            lower_quartiles: list[np.floating | float] = []
            for size in graph_sizes:
                try:
                    result_array = np.array(
                        results[variant][algorithm][metric][size], dtype=np.float64
                    )
                    # Skip if result_array is empty
                    if len(result_array) == 0:
                        continue

                    # Use mean for total_correct metric if requested,
                    # otherwise use median
                    if use_mean_for_total_correct and metric == "total_correct":
                        median = np.mean(result_array)
                        std = np.std(result_array)
                        upper_quartile = median + std
                        lower_quartile = median - std
                    else:
                        median = np.median(result_array)
                        upper_quartile, lower_quartile = np.percentile(
                            result_array, [75, 25]
                        )
                except KeyError:
                    continue
                except TypeError:
                    print(f"Dumping values: {result_array}")
                    raise
                medians.append(median)
                upper_quartiles.append(upper_quartile)
                lower_quartiles.append(lower_quartile)
            line = ax.plot(
                medians,
                label=VARIANT_NAMES[variant],
                color=VARIANT_STYLES[variant]["color"],
                linestyle=VARIANT_STYLES[variant]["linestyle"],
                marker=VARIANT_STYLES[variant]["marker"],
                linewidth=VARIANT_STYLES[variant]["linewidth"],
            )
            if i == 0:
                lines.append(line[0])
                labels.append(VARIANT_NAMES[variant])
            # Only fill between if we have data for all sizes
            if len(lower_quartiles) == len(graph_sizes) and len(upper_quartiles) == len(
                graph_sizes
            ):
                ax.fill_between(
                    np.arange(len(graph_sizes)),
                    lower_quartiles,
                    upper_quartiles,
                    alpha=0.3,
                    color=VARIANT_STYLES[variant]["color"],
                )
            ax.set_ylabel(YLABELS[metric], fontsize=18, fontfamily="serif")
            # Show x-axis values as integers.
            # Only shows the first 5 graph sizes.
            max_graph_sizes = 5
            ax.set_xticks(np.arange(max_graph_sizes))
            ax.set_xticklabels(graph_sizes[:max_graph_sizes], size=tick_fontsize)
            ax.set_ylim(0, 0.06)
            # Explicitly enable x-axis tick labels for all subplots (not just bottom)
            ax.tick_params(labelbottom=True)

            common_add_arrows(ax)
    # Turn off unused subplots.
    for j in range(i + 1, num_rows * num_cols):
        axes[j // num_cols][j % num_cols].axis("off")

    # Add title.
    # If num_failures is 1, we should say "1 Failure" instead of "1 Failures".
    fig.suptitle(title)

    # Save the figure.

    # Add shared legend to the figure
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(results),
        frameon=True,
    )

    plt.tight_layout()

    # Add padding at the bottom for the legend
    plt.subplots_adjust(bottom=0.15)

    # Create directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Save the figure
    plt.savefig(
        Path(output_dir) / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(
    failures_list: list,
    graph_size: int,
    algorithm: str,
    data_dir: str,
    variant: str,
    plot_type: str,
) -> None:
    """Main function to plot results for a fixed number of failures, graph
    size, and algorithm."""
    if plot_type == "bar":
        for num_failures in failures_list:
            plot_results_variants_bar_chart(
                num_failures, graph_size, algorithm, data_dir
            )
    elif plot_type == "compare_algos_for_fixed_variant":
        for num_failures in failures_list:
            results = pkl.load(
                open(
                    f"{data_dir}/"
                    f"strategy_comparison_num_failures_{variant}_{num_failures}.pkl",
                    "rb",
                )
            )
            metrics = [
                "query_cost_total",
                "total_failed_attempts",
                "execution_time_total",
                "total_correct",
                "total_timesteps",
            ]

            # infer graph sizes from results.
            graph_sizes = list(results["MIP"]["total_timesteps"].keys())
            plot_results(
                results,
                graph_sizes,
                metrics,
                f"strategy_comparison_num_failures_{variant}_{num_failures}.png",
            )
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_type",
        type=str,
        required=True,
        choices=[
            "bar",
            "compare_variants_by_graph_size",
            "compare_algos_for_fixed_variant",
        ],
    )
    parser.add_argument("--failures_list", nargs="+", type=int, required=True)
    parser.add_argument(
        "--graph_size",
        type=int,
        required=True,
        help="Graph size. Ignored if plot_type"
        "is compare_variants_by_graph_size or compare_algos_for_fixed_variant.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["Brute Force", "Graph Query", "Binary Tree Query", "MIP"],
        help="Algorithm to plot results for."
        "Ignored if plot_type is compare_algos_for_fixed_variant.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant to plot results for."
        "Only used if plot_type is compare_algos_for_fixed_variant.",
    )
    args = parser.parse_args()
    # Save args to a json file.
    out_dir = Path(args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"args_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f)
    main(
        args.failures_list,
        args.graph_size,
        args.algorithm,
        args.data_dir,
        args.variant,
        args.plot_type,
    )
