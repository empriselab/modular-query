"""
## Camera-ready plot for appendix IVs.
## We generate a plot that is 3 rows x 4 columns.
## Each row corresponds to a different heterogeneity level (beta).
## The first 2 columns correspond to one set of confidence levels (broader spacing)
## The last 2 columsn correspond to another set of confidence levels (narrower spacing)
## So we'll have a dashed line separating the first and second sets of columns.

There are two columns because we are showing 2 metrics: Total Failed Attempts and Total Incorrect.

Structured as a 3 x 4 matplotlib grid (this way we can enforce equal sizes for plots.)

Usage:
- python experiments/plot_appendix_module_heterogeneity.py  --output_dir experiments/results

will produce 1 PDF.
"""

import argparse
import json
import os
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_conditionalacceptance import individual_plot, individual_plot_num_modules, individual_plot_fixed_moduleselector, individual_plot_num_modules_fixed_moduleselector


def plot_appendix_module_heterogeneity(output_dir: str) -> None:
    """Plot appendix module heterogeneity plot.
    """
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
        "c_query": 0.32
    }

    metrics = [
        "total_failed_attempts",
        "total_correct",
        "total_failed_attempts",
        "total_correct",
    ]

    rows = ["beta_0.2", "beta_0.4", "beta_0.6"]
    columns = ["confidence","confidence","confidence_finer","confidence_finer"]
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=UNIFIED_PLOT_FIGSIZE)
    data_locations = {"beta_0.2": {"confidence": "experiments/results/20251110_exp1",
    "confidence_finer": "experiments/results/20251110_finerconfidences_exp1"},
                    "beta_0.4": {"confidence": "experiments/results/20251110_exp2",
                    "confidence_finer": "experiments/results/20251110_finerconfidences_exp2"},
                    "beta_0.6": {"confidence": "experiments/results/20251110_exp3",
                    "confidence_finer": "experiments/results/20251110_finerconfidences_exp3"}}

    # orders for IVs
    graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
    confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
    confidence_finer_order = [(0.8, 0.3), (0.75, 0.35), (0.7, 0.4), (0.65, 0.45), (0.6, 0.5)]
    query_cost_order = [0.08, 0.16, 0.32, 0.64]
    variant_order = ["greedy", "balanced", "conservative", "balanced-2"]
    expert_query_confidence_order = [1.0, 0.8, 0.6, 0.4]

    order_dict = {}
    order_dict["dependency_structure"] = graph_structure_order
    order_dict["confidence"] = confidence_order
    order_dict["confidence_finer"] = confidence_finer_order
    order_dict["c_query"] = query_cost_order
    order_dict["num_modules"] = None
    order_dict["variant"] = variant_order
    order_dict["expert_query_confidence"] = expert_query_confidence_order


    for i, row in enumerate(rows):
        for j, column in enumerate(columns):            
            ax = axes[i, j]
            metric = metrics[j]

            # Data loading.
            results_dir = Path(data_locations[row][column])
            df_original = pd.read_pickle(results_dir / "combined_df.pkl")

            # Add a new column confidence which has the correct and incorrect confidence paired into a tuple.
            # Drop the original correct and incorrect confidence columns.
            df_original["confidence"] = df_original.apply(lambda row: (row["correct_confidence"], row["incorrect_confidence"]), axis=1)
            df_original = df_original.drop(columns=["correct_confidence", "incorrect_confidence"])

            df = df_original[df_original["variant"] == fixed_variant]
            individual_plot(df, fixed_variables, metric, column, order_dict[column], ax, graph_size)

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
            if label and label not in seen_querying_algorithm_labels and label not in seen_module_selector_labels:
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
        loc='upper center',
        bbox_to_anchor=(center_first_four, 0.99),
        ncol=len(module_selector_labels),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE
    )
        
    plt.savefig(
        f"{output_dir}/plot_appendix_module_heterogeneity.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="experiments/results", required=True)
    args = parser.parse_args()

    plot_appendix_module_heterogeneity(args.output_dir)
