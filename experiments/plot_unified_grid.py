"""
Created: 12/13/2025.

Unified plot that puts multiple plots (for different independent variables)
into a single matplotlib figure.

General layout is as follows:
- First row (module selectors): Number of Modules (metric: Computation Time),
Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
Workloads (metric: Total Timesteps)
- Second row (querying algorithms): Number of Modules (metric: Computation Time),
Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
Workloads (metric: Total Timesteps)

Structured as a 2 x 4 matplotlib grid (this way we can enforce equal sizes for plots.)


Usage
- python experiments/plot_unified_grid.py  --output_dir experiments/results
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from modular_query.plot_utils_unified import (
    LEGEND_FONT_SIZE,
    UNIFIED_PLOT_FIGSIZE,
    individual_plot,
    individual_plot_fixed_moduleselector,
    individual_plot_num_modules,
    individual_plot_num_modules_fixed_moduleselector,
)

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
mpl.rcParams["ps.fonttype"] = 42  # embed TrueType, not Type 3


def unified_plot(output_dir: str) -> None:
    """Unified plot that puts multiple plots (for different independent
    variables) into a single matplotlib figure.

    General layout is as follows:
    - First row (module selectors): Number of Modules (metric: Computation Time),
    Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
    Workloads (metric: Total Timesteps)
    - Second row (querying algorithms): Number of Modules (metric: Computation Time),
    Redundancy (metric: Total Timesteps), Confidences (metric: Total Correct)
    Workloads (metric: Total Timesteps)
    Structured as a 2 x 4 matplotlib grid
    (this way we can enforce equal sizes for plots.)
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
    data_locations = {
        "module_selectors": {col: "experiments/results/core_ivs/" for col in columns},
        "querying_algorithms": {
            col: "experiments/results/core_ivs/" for col in columns
        },
    }

    data_locations["module_selectors"][
        "confidence_2"
    ] = "experiments/results/finer_conf_1/"
    # this one is tricky because it's actually not showing querying algorithms;
    # it's showing module selectors.
    # but it'll be in the second row of the plot.
    data_locations["querying_algorithms"][
        "confidence_2"
    ] = "experiments/results/finer_conf_2/"

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

    order_dict: dict[str, list[Any]] = {}
    order_dict["dependency_structure"] = graph_structure_order
    order_dict["confidence"] = confidence_order
    order_dict["c_query"] = query_cost_order
    order_dict["num_modules"] = []
    order_dict["variant"] = variant_order
    order_dict["confidence_2"] = confidence_2_order

    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            metric = metrics[j]

            # Data loading.
            results_dir = Path(data_locations[row][column])
            df_original = pd.read_pickle(results_dir / "combined_df.pkl")

            # Add a new column confidence which has the correct
            # and incorrect confidence paired into a tuple.
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
                        ax,
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
    fig.legend(
        module_selector_handles,
        module_selector_labels,
        loc="upper center",
        bbox_to_anchor=(center_first_four, 0.99),
        ncol=len(module_selector_labels),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    # Second legend: Querying Algorithms (from top row)
    fig.legend(
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
        f"{output_dir}/plot_unified_grid.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/plot_unified_grid.png",
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
    unified_plot(args.output_dir)
