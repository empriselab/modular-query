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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from modular_query.plot_utils import (
    VARIANT_NAMES,
    YLABELS,
    common_add_arrows,
)

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
mpl.rcParams["ps.fonttype"] = 42  # embed TrueType, not Type 3
# Constants:
UNIFIED_PLOT_FIGSIZE = (27.5, 8)
XTICK_OFFSET = 0.5
TICK_FONTSIZE = {
    "num_modules": 20,
    "dependency_structure": 12,
    "confidence": 18,
    "c_query": 20,
    "confidence_2": 12,
    "confidence_finer": 12,
    "expert_query_confidence": 20,
}
LEGEND_FONT_SIZE = 16
XLABELS = {
    "num_modules": "Number of Modules",
    "dependency_structure": "Redundancy",
    "confidence": "Confidences",
    "c_query": "Query Costs",
    "confidence_2": "Confidences",
    "expert_query_confidence": "Expert Confidence",
    "confidence_finer": "Confidences",
}
XLABEL_FONTSIZE = 20
MARKER_SIZE = 8
DEFAULT_ALPHA = 1.0

module_selector_order = [
    "Never Query",
    "Graph Query",
    "Brute Force",
    "Binary Tree Query",
    "Confidence Query",
    "Topo Query",
    "MIP",
    "Always Query",
]

STRATEGY_COLORS = {
    "Never Query": {
        "color": "gray",
        "linestyle": "--",
        "marker": "x",
        "linewidth": 2,
        "markersize": 10,
        "alpha": 0.5,
    },
    "Brute Force": {
        "color": "#006d2c",
        # "color": "#1f77b4",
        # "color": "#1b3a57",
        "linestyle": ":",
        "marker": "o",
        "linewidth": 2,
        "markersize": 12,
        "alpha": 0.25,
    },
    "Graph Query": {
        "color": "#ff7f0e",
        # "color": "#c0392b",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2,
        "markersize": 16,
        "alpha": 0.55,
    },
    "Binary Tree Query": {
        "color": "gray",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2,
        "markersize": MARKER_SIZE,
        "alpha": 0.5,
    },
    "Confidence Query": {
        "color": "#000000",
        # "color": "#2f2f2f",
        "linestyle": "-",
        "marker": "+",
        "linewidth": 2,
        "markersize": MARKER_SIZE,
        "alpha": DEFAULT_ALPHA,
    },
    "Topo Query": {
        "color": "gray",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2,
        "markersize": MARKER_SIZE,
        "alpha": 0.75,
    },
    "MIP": {
        "color": "gray",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 2,
        "markersize": MARKER_SIZE,
        "alpha": DEFAULT_ALPHA,
    },
    "Always Query": {
        "color": "gray",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2,
        "markersize": MARKER_SIZE,
        "alpha": DEFAULT_ALPHA,
    },
}

VARIANT_STYLES = {
    "greedy": {
        "color": "gray",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2,
        "name": "Execute-First",
        "markersize": MARKER_SIZE,
        "alpha": 0.5,
    },
    "balanced": {
        "color": "gray",
        "linestyle": "-",
        "marker": "x",
        "linewidth": 2,
        "name": "Query-Then-Execute",
        "markersize": MARKER_SIZE,
        "alpha": 0.5,
    },
    "conservative": {
        "color": "#e34a33",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2,
        "name": "Query-Until-Confident",
        "markersize": 12,
        "alpha": 0.75,
    },
    "balanced-2": {
        "color": "#b30000",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 2,
        "name": "Query-Until-Confident-Workload-Aware",
        "markersize": MARKER_SIZE,
        "alpha": DEFAULT_ALPHA,
    },
    "query-all": {
        "color": "gray",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2,
        "name": "Query-For-All",
        "markersize": MARKER_SIZE,
        "alpha": DEFAULT_ALPHA,
    },
}


def individual_plot(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    order: list,
    ax: plt.Axes,
    graph_size: int,
    add_x_label: bool = False,
    title: str = "",
) -> None:
    """Makes an individual plot for a given column (i.e. independent
    variable)"""

    if column == "confidence_2" or column == "confidence_finer":
        df_column = "confidence"
    else:
        df_column = column

    # First, df_filtered should have all fixed variables set (except for the one that is varying).

    # Create a boolean mask for rows that match the fixed variables.
    mask = True
    for col, value in fixed_variables.items():
        if col != df_column:
            mask = mask & (df[col] == value)
    df_filtered = df[mask]
    results_sample = df_filtered["results_dictionary"].values[0]
    x_base = np.arange(len(order)) * len(results_sample.keys()) + XTICK_OFFSET

    legend_added = set()

    # TOP3 = {"Brute Force", "Graph Query", "Confidence Query"}
    # others = [algorithm for algorithm in results_sample.keys() if algorithm not in TOP3]
    # max_jitter = 0.1
    # if len(others) > 1:
    #     other_offsets = np.linspace(-max_jitter, max_jitter, len(others))
    # else:
    #     other_offsets = [0]
    # OFFSETS = {algorithm: offset for algorithm, offset in zip(others, other_offsets)}
    # OFFSETS.update({algorithm: 0 for algorithm in TOP3})
    OFFSETS = {algorithm: 0 for algorithm in results_sample.keys()}

    ## trial 1: line plot code.
    # Collect data for each algorithm across all order values
    algorithm_data = {}
    for algorithm in results_sample.keys():
        algorithm_data[algorithm] = []

    for i, value in enumerate(order):
        # Filter the df for only those run IDs.
        df_filtered_value = df_filtered[df_filtered[df_column] == value]
        # Extract the results (from the results_dictionary column)
        results = df_filtered_value["results_dictionary"].values[0]
        for algorithm in results.keys():
            # Use mean for total_correct metric, median for others
            y_value = (
                1 - np.mean(results[algorithm][metric][graph_size])
                if metric == "total_correct"
                else np.median(results[algorithm][metric][graph_size])
            )
            algorithm_data[algorithm].append(y_value)

    # Plot a line for each algorithm
    for algorithm in module_selector_order:
        if algorithm in algorithm_data.keys():
            # Only add label to legend if we haven't seen this algorithm before
            label = algorithm if algorithm not in legend_added else ""
            style = STRATEGY_COLORS[algorithm]
            x_positions = x_base + OFFSETS[algorithm]
            ax.plot(
                x_positions,
                algorithm_data[algorithm],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
            )
            legend_added.add(algorithm)

    # # OLD BAR PLOT CODE.
    # for i, value in enumerate(order):
    #     # Filter the df for only those run IDs.
    #     df_filtered_value = df_filtered[
    #         df_filtered[column] == value
    #     ]
    #     # Extract the results (from the results_dictionary column)
    #     results = df_filtered_value["results_dictionary"].values[0]
    #     for k, algorithm in enumerate(results.keys()):
    #         # Only add label to legend if we haven't seen this algorithm before
    #         label = algorithm if algorithm not in legend_added else ""
    #         # Use mean for total_correct metric, median for others
    #         value = (
    #             1-np.mean(results[algorithm][metric][graph_size])
    #             if metric == "total_correct"
    #             else np.median(results[algorithm][metric][graph_size])
    #         )
    #         ax.bar(
    #             i * len(results.keys()) + k,
    #             value,
    #             label=label,
    #             color=STRATEGY_COLORS[algorithm]["color"],
    #         )
    #         legend_added.add(algorithm)

    # ax.set_xticks(np.arange(len(order)) * len(results.keys())+XTICK_OFFSET)
    ax.set_xticks(x_base)
    # if column is "dependency_structure", replace any underscores with hyphens
    if column == "dependency_structure":
        order = [item.replace("_", "-") for item in order]
    ax.set_xticklabels(order, fontsize=TICK_FONTSIZE[column])
    if add_x_label:
        ax.set_xlabel(
            XLABELS[column], fontsize=XLABEL_FONTSIZE, fontfamily="serif", labelpad=10
        )
        ax.xaxis.set_label_coords(
            0.5, -0.25
        )  # x=0.5 means center, y=-0.15 means 15% down from the bottom
    if title:
        ax.set_title(title, fontsize=20, fontfamily="serif", fontweight="bold")
    ax.set_ylabel(
        YLABELS[metric] if metric != "total_correct" else "Task Cost",
        fontsize=18,
        fontfamily="serif",
    )
    if metric == "total_timesteps" or metric == "total_failed_attempts":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # just for confidence_2, set the y-tick to start from 0.4 with top at 0.9.
    if column == "confidence_2":
        common_add_arrows(ax, y_lim=(0.3, 0.9))
    else:
        common_add_arrows(ax)


def individual_plot_fixed_moduleselector(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    order_dict: dict,
    ax: plt.Axes,
    graph_size: int,
    module_selector: str,
) -> None:
    """Makes an individual plot for a fixed module selector."""
    # Create a boolean mask for rows that match this combination
    mask = True
    for col, value in fixed_variables.items():
        if col != column and col != "variant":
            mask = mask & (df[col] == value)
    df_filtered = df[mask]

    legend_added = set()

    ## TRIAL 1: LINE PLOT CODE.
    # Collect data for each variant across all order values
    variant_data = {}
    for variant in order_dict["variant"]:
        variant_data[variant] = []

    for i, value in enumerate(order_dict[column]):
        # Filter the df for only those run IDs.
        df_filtered_value = df_filtered[df_filtered[column] == value]
        for variant in order_dict["variant"]:
            # Extract the row for this variant.
            row = df_filtered_value[df_filtered_value["variant"] == variant]
            # Extract the results (from the results_dictionary column)
            results = row["results_dictionary"].values[0]
            # Use mean for total_correct metric, median for others
            y_value = (
                1 - np.mean(results[module_selector][metric][graph_size])
                if metric == "total_correct"
                else np.median(results[module_selector][metric][graph_size])
            )
            variant_data[variant].append(y_value)

    # Plot a line for each variant
    x_positions = (
        np.arange(len(order_dict[column])) * len(order_dict["variant"]) + XTICK_OFFSET
    )
    for variant in variant_data.keys():
        # Only add label to legend if we haven't seen this variant before
        label = VARIANT_STYLES[variant]["name"] if variant not in legend_added else ""
        style = VARIANT_STYLES[variant]
        ax.plot(
            x_positions,
            variant_data[variant],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
        )
        legend_added.add(variant)

    ## ORIGINAL BAR PLOT CODE.
    # for i, value in enumerate(order_dict[column]):
    #     # Filter the df for only those run IDs.
    #     df_filtered_value = df_filtered[
    #         df_filtered[column] == value
    #     ]
    #     for j, variant in enumerate(order_dict["variant"]):
    #         # Only add label to legend if we haven't seen this algorithm before
    #         label = VARIANT_STYLES[variant]["name"] if variant not in legend_added else ""
    #         # Extract the row for this variant.
    #         row = df_filtered_value[
    #             df_filtered_value["variant"] == variant
    #         ]
    #         # Extract the results (from the results_dictionary column)
    #         results = row["results_dictionary"].values[0]
    #         # Use mean for total_correct metric, median for others
    #         value = (
    #             1-np.mean(results[module_selector][metric][graph_size])
    #             if metric == "total_correct"
    #             else np.median(results[module_selector][metric][graph_size])
    #         )
    #         ax.bar(
    #             i * len(order_dict["variant"]) + j,
    #             value,
    #             label=label,
    #             color=VARIANT_STYLES[variant]["color"],
    #         )
    #         legend_added.add(variant)

    ax.set_xticks(
        np.arange(len(order_dict[column])) * len(order_dict["variant"]) + XTICK_OFFSET
    )
    if column == "dependency_structure":
        order_to_use = [item.replace("_", "-") for item in order_dict[column]]
    else:
        order_to_use = order_dict[column]
    ax.set_xticklabels(order_to_use, fontsize=TICK_FONTSIZE[column])
    ax.set_ylabel(
        YLABELS[metric] if metric != "total_correct" else "Task Cost",
        fontsize=18,
        fontfamily="serif",
    )
    common_add_arrows(ax)
    ax.set_xlabel(
        XLABELS[column], fontsize=XLABEL_FONTSIZE, fontfamily="serif", labelpad=10
    )
    ax.xaxis.set_label_coords(
        0.5, -0.25
    )  # x=0.5 means center, y=-0.15 means 15% down from the bottom
    if metric == "total_timesteps" or metric == "total_failed_attempts":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def individual_plot_num_modules(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    order: list,
    ax: plt.Axes,
    graph_size: int,
) -> None:
    """Makes an individual plot for the number of modules."""
    use_mean_for_total_correct = True

    # Extract the appropriate result from the dataframe.
    mask = True
    for col, value in fixed_variables.items():
        if col != column:
            mask = mask & (df[col] == value)
    df_filtered = df[mask]
    results = df_filtered["results_dictionary"].values[0]

    # Infer graph_sizes from the results.
    if "Brute Force" in results:
        graph_sizes = list(results["Brute Force"][metric].keys())
    else:
        graph_sizes = list(results["Confidence Query"][metric].keys())

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
                    median = np.mean(1 - result_array)
                    std = np.std(1 - result_array)
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
            markersize=style["markersize"],
            label=strategy_name,
            alpha=style["alpha"],
        )

        # Plot the interquartile range band
        ax.fill_between(
            graph_sizes[: len(medians)],
            np.array(lower_quartiles),
            np.array(upper_quartiles),
            alpha=0.3,
            color=style["color"],
        )

    # Explicitly enable x-axis tick labels for all subplots (not just bottom)
    ax.tick_params(labelbottom=True)
    ax.tick_params(labelsize=TICK_FONTSIZE[column], axis="x")
    ax.set_ylabel(
        YLABELS[metric] if metric != "total_correct" else "Task Cost",
        fontsize=18,
        fontfamily="serif",
    )

    common_add_arrows(ax, xaxis_position=-0.005)

    if metric == "total_timesteps" or metric == "total_failed_attempts":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def individual_plot_num_modules_fixed_moduleselector(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    order_dict: dict,
    ax: plt.Axes,
    graph_size: int,
    module_selector: str,
    ymax=None,
) -> None:
    """Makes an individual plot for the number of modules, for a fixed module
    selector."""
    use_mean_for_total_correct = True
    # Extract the appropriate result from the dataframe.
    mask = True
    for col, value in fixed_variables.items():
        if col != column and col != "variant":
            mask = mask & (df[col] == value)
    df_filtered = df[mask]

    for variant in order_dict["variant"]:
        medians: list[np.floating | float] = []
        upper_quartiles: list[np.floating | float] = []
        lower_quartiles: list[np.floating | float] = []

        results = df_filtered[df_filtered["variant"] == variant][
            "results_dictionary"
        ].values[0]
        graph_sizes = list(results[module_selector][metric].keys())
        for size in graph_sizes:
            try:
                result_array = np.array(
                    results[module_selector][metric][size], dtype=np.float64
                )
                # Skip if result_array is empty
                if len(result_array) == 0:
                    continue

                # Use mean for total_correct metric if requested,
                # otherwise use median
                if use_mean_for_total_correct and metric == "total_correct":
                    median = np.mean(1 - result_array)
                    std = np.std(1 - result_array)
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
            markersize=VARIANT_STYLES[variant]["markersize"],
            alpha=VARIANT_STYLES[variant]["alpha"],
        )
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
        # ax.set_title(TITLES[metric])
        # ax.set_xlabel("Number of Graph Nodes")
        ax.set_ylabel(
            YLABELS[metric] if metric != "total_correct" else "Task Cost",
            fontsize=18,
            fontfamily="serif",
        )  # ax.grid(True, linestyle="--", alpha=0.7)
        # ax.grid(True, linestyle="--", alpha=0.7)
        # Show x-axis values as integers.
        # NOTE: totally hardcoded to only show the first 5 graph sizes (the rest don't have data)
        max_graph_sizes = 5
        ax.set_xticks(np.arange(max_graph_sizes))
        ax.set_xticklabels(graph_sizes[:max_graph_sizes], size=TICK_FONTSIZE[column])
        # Hardcoding the y-max to be 0.06, if no y-max is provided.
        if ymax is not None:
            ax.set_ylim(0, ymax)
        # Explicitly enable x-axis tick labels for all subplots (not just bottom)
        ax.tick_params(labelbottom=True)

        common_add_arrows(ax)
        ax.set_xlabel(
            XLABELS[column], fontsize=XLABEL_FONTSIZE, fontfamily="serif", labelpad=10
        )
        ax.xaxis.set_label_coords(
            0.5, -0.25
        )  # x=0.5 means center, y=-0.2 means 20% down from the bottom

        if metric == "total_timesteps" or metric == "total_failed_attempts":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def unified_plot_conditionalacceptance(output_dir: str) -> None:
    """Unified plot that puts multiple plots (for different independent
    variables) into a single matplotlib figure.

    General layout is as follows:
    - First row (module selectors): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
    Workloads (metric: Total Timesteps)
    - Second row (querying algorithms): Number of Modules (metric: Computation Time), Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
    Workloads (metric: Total Timesteps)
    Structured as a 2 x 4 matplotlib grid (this way we can enforce equal sizes for plots.)
    """
    # Set the fixed variables.
    fixed_variant = "balanced-2"
    fixed_module_selector = "Graph Query"
    graph_size = 10

    # Fixed variables.
    fixed_variables = {
        "variant": fixed_variant,
        "num_failures": 3,
        "dependency_structure": "all_AND",
        "confidence": (1.0, 0.1),
        "c_query": 0.32,
    }

    metrics = [
        "execution_time_total",
        "total_timesteps",
        "total_correct",
        "total_timesteps",
        "total_correct",
    ]

    rows = ["module_selectors", "querying_algorithms"]
    columns = [
        "num_modules",
        "dependency_structure",
        "confidence",
        "c_query",
        "confidence_2",
    ]
    fig, axes = plt.subplots(
        nrows=len(rows), ncols=len(columns), figsize=UNIFIED_PLOT_FIGSIZE
    )
    # data_locations = {"module_selectors": {col: "experiments/results/20251208_hricondaccept/" for col in columns}, \
    #     "querying_algorithms": {col: "experiments/results/20250929_fixbruteforce/" for col in columns}}
    # data_locations["querying_algorithms"]["confidence"] = "experiments/results/20250929_fixbruteforce_varyconfidences/"
    # Switch to using 20260111_newbaselines for all of the above commented-out data locations.
    data_locations = {
        "module_selectors": {
            col: "experiments/results/20260111_newbaselines/" for col in columns
        },
        "querying_algorithms": {
            col: "experiments/results/20260111_newbaselines/" for col in columns
        },
    }

    data_locations["module_selectors"][
        "confidence_2"
    ] = "experiments/results/20251110_finerconfidences_exp2/"
    # this one is tricky because it's actually not showing querying algorithms; it's showing module selectors.
    # but it'll be in the second row of the plot.
    data_locations["querying_algorithms"][
        "confidence_2"
    ] = "experiments/results/20251110_finerconfidences_exp3/"

    # orders for IVs
    graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
    confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
    confidence_2_order = [
        (0.8, 0.3),
        (0.75, 0.35),
        (0.7, 0.4),
        (0.65, 0.45),
        (0.6, 0.5),
    ]
    query_cost_order = [0.08, 0.16, 0.32, 0.64]
    variant_order = ["greedy", "balanced", "conservative", "balanced-2", "query-all"]

    order_dict = {}
    order_dict["dependency_structure"] = graph_structure_order
    order_dict["confidence"] = confidence_order
    order_dict["c_query"] = query_cost_order
    order_dict["num_modules"] = None
    order_dict["variant"] = variant_order
    order_dict["confidence_2"] = confidence_2_order

    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            metric = metrics[j]

            # Data loading.
            results_dir = Path(data_locations[row][column])
            df_original = pd.read_pickle(results_dir / "combined_df.pkl")

            # Add a new column confidence which has the correct and incorrect confidence paired into a tuple.
            # Drop the original correct and incorrect confidence columns.
            df_original["confidence"] = df_original.apply(
                lambda row: (row["correct_confidence"], row["incorrect_confidence"]),
                axis=1,
            )
            df_original = df_original.drop(
                columns=["correct_confidence", "incorrect_confidence"]
            )

            if row == "module_selectors":
                df = df_original[df_original["variant"] == fixed_variant]
                if column == "num_modules":
                    individual_plot_num_modules(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict[column],
                        ax,
                        graph_size,
                    )
                elif column == "confidence_2":
                    individual_plot(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict[column],
                        ax,
                        graph_size,
                        title=r"Low Variance ($\beta$ = 0.4)",
                    )
                else:
                    individual_plot(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict[column],
                        ax,
                        graph_size,
                    )

            elif row == "querying_algorithms":
                df = df_original
                if column == "num_modules":
                    individual_plot_num_modules_fixed_moduleselector(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict,
                        ax,
                        graph_size,
                        fixed_module_selector,
                        ymax=0.06,
                    )
                elif column == "confidence_2":
                    individual_plot(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict[column],
                        ax,
                        graph_size,
                        add_x_label=True,
                        title=r"High Variance ($\beta$ = 0.6)",
                    )
                else:
                    individual_plot_fixed_moduleselector(
                        df,
                        fixed_variables,
                        metric,
                        column,
                        order_dict,
                        ax,
                        graph_size,
                        fixed_module_selector,
                    )

    for j in range(axes.shape[1]):
        fig.align_ylabels(axes[:, j])

    # Step 6. Add legend - split into two groups
    # Collect handles and labels separately for each row
    module_selector_handles = []
    module_selector_labels = []
    querying_algorithm_handles = []
    querying_algorithm_labels = []

    seen_module_selector_labels = set()
    seen_querying_algorithm_labels = set()

    # Top row (module_selectors) - these show querying algorithms
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in seen_module_selector_labels:
                module_selector_handles.append(handle)
                module_selector_labels.append(label)
                seen_module_selector_labels.add(label)

    # Bottom row (querying_algorithms) - these show module selectors
    for ax in axes[1]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if (
                label
                and label not in seen_querying_algorithm_labels
                and label not in seen_module_selector_labels
            ):
                querying_algorithm_handles.append(handle)
                querying_algorithm_labels.append(label)
                seen_querying_algorithm_labels.add(label)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45, bottom=0.08)

    # Calculate the center of the first 4 columns for legend positioning
    bbox_1st_top = axes[0, 0].get_position()  # Top row, 1st column
    bbox_4th_top = axes[0, 3].get_position()  # Top row, 4th column
    center_first_four = (bbox_1st_top.x0 + bbox_4th_top.x1) / 2

    # Create two legends side by side, centered over first 4 columns
    # First legend: Module Selectors (from bottom row)
    legend1 = fig.legend(
        module_selector_handles,
        module_selector_labels,
        loc="upper center",
        bbox_to_anchor=(center_first_four, 0.99),
        ncol=len(module_selector_labels),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    # Second legend: Querying Algorithms (from top row)
    legend2 = fig.legend(
        querying_algorithm_handles,
        querying_algorithm_labels,
        loc="upper center",
        bbox_to_anchor=(center_first_four, 0.5),
        ncol=len(querying_algorithm_labels),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    # Add dashed vertical separator between columns 4 and 5 (after layout adjustments)
    # Calculate the midpoint between the right edge of column 4 and left edge of column 5
    bbox_4th_top = axes[0, 3].get_position()  # Top row, 4th column
    bbox_5th_top = axes[0, 4].get_position()  # Top row, 5th column
    bbox_5th_bottom = axes[1, 4].get_position()  # Bottom row, 5th column
    separator_x = (
        bbox_4th_top.x1 + bbox_5th_top.x0
    ) / 2 - 0.01  # Midpoint between columns

    # Draw a single dashed vertical line spanning both rows
    fig.add_artist(
        plt.Line2D(
            [separator_x, separator_x],
            [
                bbox_5th_bottom.y0,
                bbox_5th_top.y1,
            ],  # From bottom of bottom row to top of top row
            color="gray",
            linestyle="--",
            linewidth=1.5,
            transform=fig.transFigure,
            clip_on=False,
        )
    )

    # Add subplot labels (a) and (b)
    axes[0, 0].text(
        -0.15,
        1.25,
        "(a)",
        transform=axes[0, 0].transAxes,
        fontsize=18,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="serif",
    )
    axes[0, 4].text(
        -0.2,
        1.25,
        "(b)",
        transform=axes[0, 4].transAxes,
        fontsize=18,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="serif",
    )  # Step 3. Save the figure.

    plt.savefig(
        f"{output_dir}/plot_conditional_acceptance_fontfix.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/plot_conditional_acceptance_fontfix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="experiments/results", required=True
    )
    args = parser.parse_args()
    unified_plot_conditionalacceptance(args.output_dir)
