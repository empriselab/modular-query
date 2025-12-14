"""
Created: 12/13/2025 for the conditional acceptance.

Unified plot that puts multiple plots (for different independent variables) into a single matplotlib figure.

General layout is as follows:
- First row (module selectors): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct) 
Workloads (metric: Total Timesteps) 
- Second row (querying algorithms): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct) 
Workloads (metric: Total Timesteps) 

Structured as a 2 x 4 matplotlib grid (this way we can enforce equal sizes for plots.)


Usage
- python experiments/plot_conditionalacceptance.py  --output_dir experiments/results
"""

import argparse
import json
import os
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from modular_query.plot_utils import (
    STRATEGY_COLORS,
    VARIANT_STYLES,
    YLABELS,
    plot_results,
    plot_results_across_graph_sizes,
    common_add_arrows,
)

# Constants:
UNIFIED_PLOT_FIGSIZE = (24, 12)
XTICK_OFFSET = 0.5
TICK_FONTSIZE = {"num_modules":20, "dependency_structure":12, "confidence":16, "c_query":20}
LEGEND_FONT_SIZE = 20

def individual_plot(df: pd.DataFrame, fixed_variables: dict, metric: str, \
    column: str, order: list, ax: plt.Axes, graph_size: int) -> None:
    """
    Makes an individual plot for a given column (i.e. independent variable)
    """

    # First, df_filtered should have all fixed variables set (except for the one that is varying).

    # Create a boolean mask for rows that match the fixed variables.
    mask = True
    for col, value in fixed_variables.items():
        if col != column:
            mask = mask & (df[col] == value)
    df_filtered = df[mask]

    legend_added = set()
    for i, value in enumerate(order):
        # Filter the df for only those run IDs.
        df_filtered_graph_structure = df_filtered[
            df_filtered[column] == value
        ]
        # Extract the results (from the results_dictionary column)
        results = df_filtered_graph_structure["results_dictionary"].values[0]
        for k, algorithm in enumerate(results.keys()):
            # Only add label to legend if we haven't seen this algorithm before
            label = algorithm if algorithm not in legend_added else ""
            # Use mean for total_correct metric, median for others
            value = (
                np.mean(results[algorithm][metric][graph_size])
                if metric == "total_correct"
                else np.median(results[algorithm][metric][graph_size])
            )
            ax.bar(
                i * len(results.keys()) + k,
                value,
                label=label,
                color=STRATEGY_COLORS[algorithm]["color"],
            )
            legend_added.add(algorithm)

    ax.set_xticks(np.arange(len(order)) * len(results.keys())+XTICK_OFFSET)
    ax.set_xticklabels(order, fontsize=TICK_FONTSIZE[column])
    ax.set_ylabel(YLABELS[metric], fontsize=18, fontfamily='serif')

    common_add_arrows(ax)

def individual_plot_num_modules(df: pd.DataFrame, fixed_variables: dict, metric: str, \
    column: str, order: list, ax: plt.Axes, graph_size: int) -> None:
    """
    Makes an individual plot for the number of modules.
    """
    use_mean_for_total_correct = True

    # Extract the appropriate result from the dataframe.
    mask = True
    for col, value in fixed_variables.items():
        if col != column:
            mask = mask & (df[col] == value)
    df_filtered = df[mask]
    results = df_filtered["results_dictionary"].values[0]

    # Infer graph_sizes from the results.
    graph_sizes = list(results["Brute Force"][metric].keys())

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

    # Explicitly enable x-axis tick labels for all subplots (not just bottom)
    ax.tick_params(labelbottom=True)
    ax.tick_params(labelsize=TICK_FONTSIZE[column], axis='x')
    ax.set_ylabel(YLABELS[metric], fontsize=18, fontfamily='serif')

    common_add_arrows(ax, xaxis_position=-0.005)


def unified_plot_conditionalacceptance(output_dir: str) -> None:
    """Unified plot that puts multiple plots (for different independent variables) into a single matplotlib figure.
    General layout is as follows:
    - First row (module selectors): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct) 
    Workloads (metric: Total Timesteps) 
    - Second row (querying algorithms): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct) 
    Workloads (metric: Total Timesteps) 
    Structured as a 2 x 4 matplotlib grid (this way we can enforce equal sizes for plots.)
    """
    # Load the dataframe.
    results_dir = Path("experiments/results/20251208_hricondaccept/")
    fixed_variant = "balanced-2"
    fixed_module_selector = "Graph Query"
    graph_size = 10
    df_original = pd.read_pickle(results_dir / "combined_df.pkl")

    # Add a new column confidence which has the correct and incorrect confidence paired into a tuple.
    # Drop the original correct and incorrect confidence columns.
    df_original["confidence"] = df_original.apply(lambda row: (row["correct_confidence"], row["incorrect_confidence"]), axis=1)
    df_original = df_original.drop(columns=["correct_confidence", "incorrect_confidence"])


    # Get a variant-specific df (variant = querying algorithm)
    df_fixed_variant = df_original[df_original["variant"] == fixed_variant]


    # Fixed variables.
    fixed_variables = {
        "variant": fixed_variant,
        "num_failures": 3,
        "dependency_structure": "all_AND",
        "confidence": (1.0, 0.1),
        "c_query": 0.32
    }

    metrics = [
        "execution_time_total",
        "total_timesteps",
        "total_correct",
        "total_timesteps"
    ]

    rows = ["module_selectors", "querying_algorithms"]
    columns = ["num_modules","dependency_structure", "confidence", "c_query"]
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=UNIFIED_PLOT_FIGSIZE)

    # orders for IVs
    graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
    confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
    query_cost_order = [0.08, 0.16, 0.32, 0.64]

    order_dict = {}
    order_dict["dependency_structure"] = graph_structure_order
    order_dict["confidence"] = confidence_order
    order_dict["c_query"] = query_cost_order
    order_dict["num_modules"] = None

    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            df = df_fixed_variant
            
            ax = axes[i, j]
            metric = metrics[j]

            if column == "num_modules":
                individual_plot_num_modules(df, fixed_variables, metric, column, order_dict[column], ax, graph_size)
            else:   
                individual_plot(df, fixed_variables, metric, column, order_dict[column], ax, graph_size)

    # Step 6. Add legend.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5, -0.1),ncol=len(labels),fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Step 3. Save the figure.
    plt.savefig(
        f"{output_dir}/plot_conditional_acceptance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="experiments/results", required=True)
    args = parser.parse_args()
    unified_plot_conditionalacceptance(args.output_dir)
