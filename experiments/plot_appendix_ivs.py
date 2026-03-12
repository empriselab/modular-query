"""## Camera-ready plot for appendix IVs. ## Outcome: for all 5 IVs, we
generate two rows of plots. ## First row: compares module selectors. ## Second
row: compares querying algorithms. ## Each row has 5 plots in it (one for each
metric.)

General layout is as follows:
- First row (module selectors): Query Cost, Total Failed Attempts, Computation Time, Total Correct, Total Timesteps
- Second row (querying algorithms): Query Cost, Total Failed Attempts, Computation Time, Total Correct, Total Timesteps

Structured as a 2 x 5 matplotlib grid (this way we can enforce equal sizes for plots.)

Usage:
- python experiments/plot_appendix_ivs.py  --output_dir experiments/results

will produce 5 PDFs, one for each IV (num modules, graph structures, confidences, query costs, expert confidence).
"""

import argparse
import json
import os
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_conditionalacceptance import (
    individual_plot,
    individual_plot_fixed_moduleselector,
    individual_plot_num_modules,
    individual_plot_num_modules_fixed_moduleselector,
)


# Adapted from plot_conditionalacceptance.py.
def plot_appendix_ivs(output_dir: str, iv: str, plot_index: int) -> None:
    """Plot appendix IVs for all 5 IVs."""
    # Constants:
    UNIFIED_PLOT_FIGSIZE = (27.5, 8)
    LEGEND_FONT_SIZE = 16

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
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    rows = ["module_selectors", "querying_algorithms"]
    columns = metrics
    fig, axes = plt.subplots(
        nrows=len(rows), ncols=len(columns), figsize=UNIFIED_PLOT_FIGSIZE
    )
    if iv in ["num_modules", "dependency_structure", "c_query"]:
        # data_locations = {"module_selectors": "experiments/results/20251208_hricondaccept/", \
        # "querying_algorithms": "experiments/results/20250929_fixbruteforce/"}
        # Switch to using 20260111_newbaselines.
        data_locations = {
            "module_selectors": "experiments/results/20260111_newbaselines/",
            "querying_algorithms": "experiments/results/20260111_newbaselines/",
        }
    elif iv == "confidence":
        # data_locations = {"module_selectors": "experiments/results/20251208_hricondaccept/", \
        # "querying_algorithms": "experiments/results/20250929_fixbruteforce_varyconfidences/"}
        # Switch to using 20260111_newbaselines.
        data_locations = {
            "module_selectors": "experiments/results/20260111_newbaselines/",
            "querying_algorithms": "experiments/results/20260111_newbaselines/",
        }
    elif iv == "expert_query_confidence":
        # data_locations = {"module_selectors": "experiments/results/20260105_noisyexpertfinal/", \
        # "querying_algorithms": "experiments/results/20260105_noisyexpertfinal/"}
        data_locations = {
            "module_selectors": "experiments/results/20260112_noisyexpert_newbaselines/",
            "querying_algorithms": "experiments/results/20260112_noisyexpert_newbaselines/",
        }

    # orders for IVs
    graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
    confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
    query_cost_order = [0.08, 0.16, 0.32, 0.64]
    variant_order = ["greedy", "balanced", "conservative", "balanced-2", "query-all"]
    expert_query_confidence_order = [1.0, 0.8, 0.6, 0.4]

    order_dict = {}
    order_dict["dependency_structure"] = graph_structure_order
    order_dict["confidence"] = confidence_order
    order_dict["c_query"] = query_cost_order
    order_dict["num_modules"] = None
    order_dict["variant"] = variant_order
    order_dict["expert_query_confidence"] = expert_query_confidence_order

    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            metric = column

            # Data loading.
            results_dir = Path(data_locations[row])
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
                if iv == "num_modules":
                    individual_plot_num_modules(
                        df, fixed_variables, metric, iv, order_dict[iv], ax, graph_size
                    )
                else:
                    individual_plot(
                        df, fixed_variables, metric, iv, order_dict[iv], ax, graph_size
                    )

            elif row == "querying_algorithms":
                df = df_original
                if iv == "num_modules":
                    if metric == "execution_time_total":
                        ymax = 0.06
                    else:
                        ymax = None
                    individual_plot_num_modules_fixed_moduleselector(
                        df,
                        fixed_variables,
                        metric,
                        iv,
                        order_dict,
                        ax,
                        graph_size,
                        fixed_module_selector,
                        ymax=ymax,
                    )
                else:
                    individual_plot_fixed_moduleselector(
                        df,
                        fixed_variables,
                        metric,
                        iv,
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

    plt.savefig(
        f"{output_dir}/plot_appendix_ivs_{plot_index+1:02d}_{iv}.pdf",
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

    ivs = [
        "num_modules",
        "dependency_structure",
        "confidence",
        "c_query",
        "expert_query_confidence",
    ]
    for i, iv in enumerate(ivs):
        print(f"Plotting appendix plot for IV: {iv}...")
        plot_appendix_ivs(args.output_dir, iv, i)
