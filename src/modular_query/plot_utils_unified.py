import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from modular_query.plot_utils import (
    VARIANT_NAMES,
    YLABELS,
    common_add_arrows,
)

from matplotlib.ticker import MaxNLocator


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

    if column in ("confidence_2", "confidence_finer"):
        df_column = "confidence"
    else:
        df_column = column

    # First, df_filtered should have all fixed variables set
    # (except for the one that is varying).

    # Create a boolean mask for rows that match the fixed variables.
    mask = True
    for col, value in fixed_variables.items():
        if col != df_column:
            mask = mask & (df[col] == value)
    df_filtered = df[mask]
    results_sample = df_filtered["results_dictionary"].values[0]
    x_base = np.arange(len(order)) * len(results_sample.keys()) + XTICK_OFFSET

    legend_added = set()

    OFFSETS = {algorithm: 0 for algorithm in results_sample.keys()}

    ## trial 1: line plot code.
    # Collect data for each algorithm across all order values
    algorithm_data: dict[str, list[float]] = {}
    for algorithm in results_sample.keys():
        algorithm_data[algorithm] = []

    for value in order:
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
        if algorithm in algorithm_data:
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

    ax.set_xticks(x_base)
    # if column is "dependency_structure", replace any underscores with hyphens
    if column == "dependency_structure":
        order = [item.replace("_", "-") for item in order]
    ax.set_xticklabels(order, fontsize=TICK_FONTSIZE[column])
    if add_x_label:
        ax.set_xlabel(
            XLABELS[column], fontsize=XLABEL_FONTSIZE, fontfamily="serif", labelpad=10
        )
        ax.xaxis.set_label_coords(0.5, -0.25)
    if title:
        ax.set_title(title, fontsize=20, fontfamily="serif", fontweight="bold")
    ax.set_ylabel(
        YLABELS[metric] if metric != "total_correct" else "Task Cost",
        fontsize=18,
        fontfamily="serif",
    )
    if metric in ("total_timesteps", "total_failed_attempts"):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
        if col not in (column, "variant"):
            mask = mask & (df[col] == value)
    df_filtered = df[mask]

    legend_added = set()

    ## TRIAL 1: LINE PLOT CODE.
    # Collect data for each variant across all order values
    variant_data: dict[str, list[float]] = {}
    for variant in order_dict["variant"]:
        variant_data[variant] = []

    for value in order_dict[column]:
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
    for variant, data in variant_data.items():
        # Only add label to legend if we haven't seen this variant before
        label = VARIANT_STYLES[variant]["name"] if variant not in legend_added else ""
        style = VARIANT_STYLES[variant]
        ax.plot(
            x_positions,
            data,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
        )
        legend_added.add(variant)

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
    ax.xaxis.set_label_coords(0.5, -0.25)
    if metric in ("total_timesteps", "total_failed_attempts"):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def individual_plot_num_modules(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    ax: plt.Axes,
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
        ax.plot(
            graph_sizes[: len(medians)],
            np.array(medians),
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

    if metric in ("total_timesteps", "total_failed_attempts"):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def individual_plot_num_modules_fixed_moduleselector(
    df: pd.DataFrame,
    fixed_variables: dict,
    metric: str,
    column: str,
    order_dict: dict,
    ax: plt.Axes,
    module_selector: str,
    ymax=None,
) -> None:
    """Makes an individual plot for the number of modules, for a fixed module
    selector."""
    use_mean_for_total_correct = True
    # Extract the appropriate result from the dataframe.
    mask = True
    for col, value in fixed_variables.items():
        if col not in (column, "variant"):
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
        ax.plot(
            np.array(medians),
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
                np.array(lower_quartiles),
                np.array(upper_quartiles),
                alpha=0.3,
                color=VARIANT_STYLES[variant]["color"],
            )
        ax.set_ylabel(
            YLABELS[metric] if metric != "total_correct" else "Task Cost",
            fontsize=18,
            fontfamily="serif",
        )
        # Show x-axis values as integers.
        # Only shows the first 5 graph sizes.
        max_graph_sizes = 5
        ax.set_xticks(np.arange(max_graph_sizes))
        ax.set_xticklabels(graph_sizes[:max_graph_sizes], size=TICK_FONTSIZE[column])
        if ymax is not None:
            ax.set_ylim(0, ymax)
        # Explicitly enable x-axis tick labels for all subplots (not just bottom)
        ax.tick_params(labelbottom=True)

        common_add_arrows(ax)
        ax.set_xlabel(
            XLABELS[column], fontsize=XLABEL_FONTSIZE, fontfamily="serif", labelpad=10
        )
        ax.xaxis.set_label_coords(0.5, -0.25)

        if metric in ("total_timesteps", "total_failed_attempts"):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
