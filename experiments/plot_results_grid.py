"""Plot results from a grid search experiment.

Looks up all pkl files in the results directory and generates plots for
each of them.
"""

import argparse
import json
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from modular_query.plot_utils import (
    STRATEGY_COLORS,
    VARIANT_STYLES,
    plot_results,
    plot_results_across_graph_sizes,
)


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
            use_mean_for_total_correct=True,
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
            use_mean_for_total_correct=True,
        )


def plot_results_grid_confidences(
    results_dir: str, variant: str, graph_size: int
) -> None:
    """Plot results from a grid search experiment for varying confidences, for
    a fixed set of metrics.

    (analogous to plot_results_grid_graph_structures, but with
    confidence setting as the IV.)
    """
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))
    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: confidences,
    # secondary x-axis: iterate over (correct_confidence, incorrect_confidence) pairs.
    # y-axis: metric.
    # so each group of bars corresponds to a different confidence setting,
    # and each bar corresponds to a different
    # (correct_confidence, incorrect_confidence) pair.

    # Filter on the variant.
    df = df[df["variant"] == variant]

    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("run_id")
    ivs.remove("variant")
    ivs.remove("results_dictionary")
    ivs.remove("correct_confidence")
    ivs.remove("incorrect_confidence")
    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    # For each unique combination of IVs,
    # we need to collect the run IDs that have that combination
    # (there should be num_confidences of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])

            run_ids = df[mask]["run_id"].unique()
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the confidences, in a particular order

            confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, confidence in enumerate(confidence_order):
                # Filter the df for only those run IDs.
                # (the confidence pairs are unique,
                # so we can just filter on the first element of the tuple)
                df_filtered_confidence = df_filtered[
                    df_filtered["correct_confidence"] == confidence[0]
                ]
                # Extract the results (from the results_dictionary column)
                results = df_filtered_confidence["results_dictionary"].values[0]
                for j, algorithm in enumerate(results.keys()):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = algorithm if algorithm not in legend_added else ""
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[algorithm][metric][graph_size]) if metric == "total_correct" else np.median(results[algorithm][metric][graph_size])
                    ax.bar(
                        i * len(results.keys()) + j,
                        value,
                        label=label,
                        color=STRATEGY_COLORS[algorithm]["color"],
                    )
                    legend_added.add(algorithm)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Confidences")
            # Tick labels are the (correct_confidence, incorrect_confidence) pairs.
            ax.set_xticks(np.arange(len(confidence_order)) * len(results.keys()))
            ax.set_xticklabels(confidence_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same variant names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Confidence Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_results_grid_confidences_fixed_module_selector(
    results_dir: str, module_selector: str, graph_size: int
) -> None:
    """Plot results from a grid search experiment for varying confidences, for
    a fixed set of metrics.

    (analogous to plot_results_grid_graph_structures, but with
    confidence setting as the IV.)
    """
    # Assumes module selector is fixed.
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))

    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: confidences,
    # secondary x-axis: iterate over (correct_confidence, incorrect_confidence) pairs.
    # y-axis: metric.
    # so each group of bars corresponds to a different confidence setting,
    # and each bar corresponds to a different
    # (correct_confidence, incorrect_confidence) pair.
    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("run_id")
    ivs.remove("variant")
    ivs.remove("results_dictionary")
    ivs.remove("correct_confidence")
    ivs.remove("incorrect_confidence")
    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    # so each group of bars corresponds to a different confidence setting,
    # and each bar corresponds to a different
    # (correct_confidence, incorrect_confidence) pair.
    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    variant_order = ["greedy", "balanced", "conservative", "balanced-2"]

    # For each unique combination of IVs,
    # we need to collect the run IDs that have that combination
    # (there should be num_confidences of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])
            run_ids = df[mask]["run_id"].unique()
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the confidences, in a particular order
            confidence_order = [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, confidence in enumerate(confidence_order):
                # Filter the df for only those run IDs.
                df_filtered_confidence = df_filtered[
                    df_filtered["correct_confidence"] == confidence[0]
                ]
                # Extract the results (from the results_dictionary column)
                results = df_filtered_confidence["results_dictionary"].values[0]
                for j, variant in enumerate(variant_order):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = variant if variant not in legend_added else ""
                    # Extract the row for this variant.
                    row = df_filtered_confidence[
                        df_filtered_confidence["variant"] == variant
                    ]
                    # Extract the results (from the results_dictionary column)
                    results = row["results_dictionary"].values[0]
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[module_selector][metric][graph_size]) if metric == "total_correct" else np.median(results[module_selector][metric][graph_size])
                    ax.bar(
                        i * len(confidence_order) + j,
                        value,
                        label=label,
                        color=VARIANT_STYLES[variant]["color"],
                    )
                    legend_added.add(variant)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Confidences")
            # Tick labels are the (correct_confidence, incorrect_confidence) pairs.
            ax.set_xticks(np.arange(len(confidence_order)) * len(variant_order))
            ax.set_xticklabels(confidence_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same variant names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Confidence Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}_{module_selector}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Plot for graph structures.
def plot_results_grid_graph_structures(
    results_dir: str, variant: str, graph_size: int
) -> None:
    """Plot results from a grid search experiment for varying graph structures,
    for a fixed set of metrics."""
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))

    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: graph structures,
    # secondary x-axis: iterate over algorithms/module selectors.
    # y-axis: metric.
    # so each group of bars corresponds to a different graph structure,
    # and each bar corresponds to a different algorithm/module selector.

    # First, we only want to look at run IDs for the given variant.
    # Then, we need to collect the run IDs
    # that have fixed values of the IVs except for the graph structure.
    df = df[df["variant"] == variant]
    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("variant")
    ivs.remove("run_id")
    ivs.remove("results_dictionary")
    ivs.remove("dependency_structure")
    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    # For each unique combination of IVs, we need to collect the run IDs
    # that have that combination
    # (there should be num_graph_structures of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])
            run_ids = df[mask]["run_id"].unique()
            # print(f"Run IDs for combination {combination.to_dict()}: {run_ids}")
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the graph structures, in a particular order
            graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, graph_structure in enumerate(graph_structure_order):
                # Filter the df for only those run IDs.
                df_filtered_graph_structure = df_filtered[
                    df_filtered["dependency_structure"] == graph_structure
                ]
                # Extract the results (from the results_dictionary column)
                results = df_filtered_graph_structure["results_dictionary"].values[0]
                # Plot the results for this graph structure.
                # results is structured with keys
                # equal to the different algorithms/module selectors.
                # We want to plot the values of the metric
                # for each algorithm/module selector.
                # want bars of the same algorithm to have the same color.
                for j, algorithm in enumerate(results.keys()):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = algorithm if algorithm not in legend_added else ""
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[algorithm][metric][graph_size]) if metric == "total_correct" else np.median(results[algorithm][metric][graph_size])
                    ax.bar(
                        i * len(results.keys()) + j,
                        value,
                        label=label,
                        color=STRATEGY_COLORS[algorithm]["color"],
                    )
                    legend_added.add(algorithm)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Algorithms")
            # Tick labels are the graph structures.
            ax.set_xticks(np.arange(len(graph_structure_order)) * len(results.keys()))
            ax.set_xticklabels(graph_structure_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same algorithm names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Graph Structure Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Analogous function to above,
# but we fix the module selector/strategy, and vary the variant.
def plot_results_grid_graph_structures_fixed_module_selector(
    results_dir: str, module_selector: str, graph_size: int
) -> None:
    """Plot results from a grid search experiment for varying graph structures,
    for a fixed set of metrics.

    Assumes module selector is fixed.
    """
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))
    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: graph structures, secondary x-axis: iterate over variants.
    # y-axis: metric.
    # so each group of bars corresponds to a different graph structure,
    # and each bar corresponds to a different variant.

    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("run_id")
    ivs.remove("variant")
    ivs.remove("results_dictionary")
    ivs.remove("dependency_structure")

    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    variant_order = ["greedy", "balanced", "conservative", "balanced-2"]

    # For each unique combination of IVs,
    # we need to collect the run IDs that have that combination
    # (there should be num_graph_structures of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])
            run_ids = df[mask]["run_id"].unique()
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the graph structures, in a particular order
            graph_structure_order = ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, graph_structure in enumerate(graph_structure_order):
                # Filter the df for only those run IDs.
                df_filtered_graph_structure = df_filtered[
                    df_filtered["dependency_structure"] == graph_structure
                ]
                # Plot the results for this graph structure.
                # results is structured with keys
                # equal to the different algorithms/module selectors.
                # We want to plot the values of the metric
                # for each algorithm/module selector.
                # want bars of the same algorithm to have the same color.
                for j, variant in enumerate(variant_order):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = variant if variant not in legend_added else ""
                    # Extract the row for this variant.
                    row = df_filtered_graph_structure[
                        df_filtered_graph_structure["variant"] == variant
                    ]
                    # Extract the results (from the results_dictionary column)
                    results = row["results_dictionary"].values[0]
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[module_selector][metric][graph_size]) if metric == "total_correct" else np.median(results[module_selector][metric][graph_size])
                    ax.bar(
                        i * len(variant_order) + j,
                        value,
                        label=label,
                        color=VARIANT_STYLES[variant]["color"],
                    )
                    legend_added.add(variant)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Variants")
            # Tick labels are the graph structures.
            ax.set_xticks(np.arange(len(graph_structure_order)) * len(variant_order))
            ax.set_xticklabels(graph_structure_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same algorithm names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Variant Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}_{module_selector}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_results_grid_cquery(results_dir: str, variant: str, graph_size: int) -> None:
    """Plot results from a grid search experiment for varying query costs, for
    a fixed set of metrics.

    Analogous to plot_results_grid_confidences.
    """
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))
    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: query costs, secondary x-axis: iterate over algorithms/module selectors.
    # y-axis: metric.
    # so each group of bars corresponds to a different query cost,
    # and each bar corresponds to a different algorithm/module selector.

    # First, we only want to look at run IDs for the given variant.
    # Then, we need to collect the run IDs
    # that have fixed values of the IVs except for the query cost.
    df = df[df["variant"] == variant]
    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("variant")
    ivs.remove("run_id")
    ivs.remove("results_dictionary")
    ivs.remove("c_query")
    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    # For each unique combination of IVs, we need to collect the run IDs
    # that have that combination
    # (there should be num_graph_structures of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])
            run_ids = df[mask]["run_id"].unique()
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the query costs, in a particular order
            query_cost_order = [0.08, 0.16, 0.32, 0.64]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, query_cost in enumerate(query_cost_order):
                # Filter the df for only those run IDs.
                df_filtered_query_cost = df_filtered[
                    df_filtered["c_query"] == query_cost
                ]
                # Extract the results (from the results_dictionary column)
                results = df_filtered_query_cost["results_dictionary"].values[0]
                for j, algorithm in enumerate(results.keys()):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = algorithm if algorithm not in legend_added else ""
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[algorithm][metric][graph_size]) if metric == "total_correct" else np.median(results[algorithm][metric][graph_size])
                    ax.bar(
                        i * len(query_cost_order) + j,
                        value,
                        label=label,
                        color=STRATEGY_COLORS[algorithm]["color"],
                    )
                    legend_added.add(algorithm)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Query Costs")
            # Tick labels are the query costs.
            ax.set_xticks(np.arange(len(query_cost_order)) * len(results.keys()))
            ax.set_xticklabels(query_cost_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same algorithm names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Query Cost Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}_c_query.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_results_grid_cquery_fixed_module_selector(
    results_dir: str, module_selector: str, graph_size: int
) -> None:
    """Plot results from a grid search experiment for varying query costs, for
    a fixed set of metrics.

    Analogous to plot_results_grid_confidences_fixed_module_selector.
    """
    # Load the dataframe.
    df = pd.read_pickle(os.path.join(results_dir, "combined_df.pkl"))
    # The general structure is as follows:
    # we want to produce a grouped bar chart with the following structure:
    # x-axis: query costs, secondary x-axis: iterate over variants.
    # y-axis: metric.
    # so each group of bars corresponds to a different query cost,
    # and each bar corresponds to a different variant.
    # Get the unique values of the IVs.
    ivs = df.columns.tolist()
    ivs.remove("run_id")
    ivs.remove("variant")
    ivs.remove("results_dictionary")
    ivs.remove("c_query")
    # Get the unique combinations of values for the IVs.
    unique_combinations = df[ivs].drop_duplicates()

    metrics = [
        "query_cost_total",
        "total_failed_attempts",
        "execution_time_total",
        "total_correct",
        "total_timesteps",
    ]

    variant_order = ["greedy", "balanced", "conservative", "balanced-2"]

    # For each unique combination of IVs, we need to collect the run IDs
    # that have that combination
    # (there should be num_graph_structures of these run IDs in total.)
    for _, combination in tqdm(unique_combinations.iterrows()):
        _, axes = plt.subplots(ncols=len(metrics), figsize=(24, 8), sharex=True)
        for i, metric in enumerate(metrics):
            # Create a boolean mask for rows that match this combination
            mask = True
            for col in ivs:
                mask = mask & (df[col] == combination[col])
            run_ids = df[mask]["run_id"].unique()
            # Filter the df for only those run IDs.
            df_filtered = df[df["run_id"].isin(run_ids)]
            # Create the grouped bar chart accordingly
            # (need to write custom code for this).
            # Step 1. Create a figure and axis.
            ax = axes[i]
            # Step 2. Iterate over the query costs, in a particular order
            query_cost_order = [0.08, 0.16, 0.32, 0.64]
            # need to handle x offsets carefully here.
            # Track which algorithms we've already added to legend
            legend_added = set()
            for i, query_cost in enumerate(query_cost_order):
                # Filter the df for only those run IDs.
                df_filtered_query_cost = df_filtered[
                    df_filtered["c_query"] == query_cost
                ]
                # Extract the results (from the results_dictionary column)
                results = df_filtered_query_cost["results_dictionary"].values[0]
                for j, variant in enumerate(variant_order):
                    # Only add label to legend if we haven't seen this algorithm before
                    label = variant if variant not in legend_added else ""
                    # Extract the row for this variant.
                    row = df_filtered_query_cost[
                        df_filtered_query_cost["variant"] == variant
                    ]
                    # Extract the results (from the results_dictionary column)
                    results = row["results_dictionary"].values[0]
                    # Use mean for total_correct metric, median for others
                    value = np.mean(results[module_selector][metric][graph_size]) if metric == "total_correct" else np.median(results[module_selector][metric][graph_size])
                    ax.bar(
                        i * len(query_cost_order) + j,
                        value,
                        label=label,
                        color=VARIANT_STYLES[variant]["color"],
                    )
                    legend_added.add(variant)
            # Step 3. Add title. Put in all IV values too.
            # ax.set_title(f"{metric}")
            # Step 4. Add x-axis labels.
            ax.set_xlabel("Query Costs")
            # Tick labels are the query costs.
            ax.set_xticks(np.arange(len(query_cost_order)) * len(results.keys()))
            ax.set_xticklabels(query_cost_order)
            # Step 5. Add y-axis labels.
            ax.set_ylabel(metric)
            # Step 6. Add legend.
            # but I don't want it to repeatedly display the same algorithm names.
            ax.legend()
        # Add title before tight_layout to avoid overlap
        plt.suptitle(f"Query Cost Comparison for Graph Size {graph_size}")
        plt.tight_layout()
        # Add extra space at the top for the title
        plt.subplots_adjust(top=0.9)
        # Step 3. Save the figure.
        plt.savefig(
            f"{results_dir}/plot_{combination.to_dict()}_c_query_{module_selector}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, default="experiments/results", required=True
    )
    args = parser.parse_args()

    fixed_graph_size = 10
    # plot_results_grid(args.results_dir)
    # plot_results_grid_query_algorithms(args.results_dir)
    # plot_results_grid_graph_structures(args.results_dir, "balanced", 10)
    # plot_results_grid_graph_structures_fixed_module_selector(
    #     args.results_dir, "Brute Force", 10
    # )
    # plot_results_grid_cquery(args.results_dir, "balanced", fixed_graph_size)
    plot_results_grid_cquery_fixed_module_selector(
        args.results_dir, "Brute Force", fixed_graph_size
    )
