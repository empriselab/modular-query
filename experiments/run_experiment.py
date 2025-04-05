#!/usr/bin/env python3
"""Experiment to measure and compare performance of different querying
strategies."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from modular_query.modular_policy import ModularPolicy
from modular_query.modules import StateModule
from modular_query.query_strategies.always_query import AlwaysQueryStrategy
from modular_query.query_strategies.brute_force import BruteForceQueryStrategy
from modular_query.query_strategies.milp import MILPQueryStrategy
from modular_query.query_strategies.never_query import NeverQueryStrategy
from modular_query.utils import generate_random_logic_gate_module_graph


def run_experiment(
    graph_sizes: list[int],
    num_trials: int = 5,
    edge_probability: float = 0.3,
    correct_answer_cost: float = 0.0,
    incorrect_answer_cost: float = 1000.0,
    min_querying_cost: float = 10.0,
    max_querying_cost: float = 100.0,
    seed: int = 0,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Run experiments with different graph sizes and querying strategies."""
    # Set up RNG.
    rng = np.random.default_rng(seed)

    def query_cost_sampler(rng: np.random.Generator) -> float:
        return rng.uniform(min_querying_cost, max_querying_cost)

    # Initialize strategies.
    strategies = {
        "Always Query": AlwaysQueryStrategy(correct_answer_cost, incorrect_answer_cost),
        "Never Query": NeverQueryStrategy(correct_answer_cost, incorrect_answer_cost),
        "Brute Force": BruteForceQueryStrategy(
            correct_answer_cost, incorrect_answer_cost
        ),
        "MILP": MILPQueryStrategy(correct_answer_cost, incorrect_answer_cost),
    }

    # Initialize results structure:
    # strategy -> metric -> graph_size -> list of values
    results: dict[str, dict[str, dict[int, list[float]]]] = {}
    for strategy_name in strategies:
        results[strategy_name] = {
            "query_cost": {size: [] for size in graph_sizes},
            "task_cost": {size: [] for size in graph_sizes},
            "total_cost": {size: [] for size in graph_sizes},
            "execution_time": {size: [] for size in graph_sizes},
        }

    # Run experiments for each graph size.
    for size in graph_sizes:
        print(f"Running experiments for graph size {size}")

        for _ in range(num_trials):

            # Generate a random graph.
            module_graph = generate_random_logic_gate_module_graph(
                num_modules=size,
                edge_probability=edge_probability,
                query_cost_sampler=query_cost_sampler,
                rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                is_policy=True,
            )

            # Use random state inputs.
            state = bool(rng.integers(0, 2))

            # Get the correct expected output.
            all_queryable_module_names = {
                m.get_name() for m in module_graph.get_modules()
            }
            all_queryable_module_names.remove("state")
            assert isinstance(module_graph.root, StateModule)
            module_graph.root.set_state(state)
            computed_values, _, _ = module_graph.compute_values(
                expert_query_module_names=all_queryable_module_names
            )
            ground_truth_output = computed_values[module_graph.leaf]

            # Run each strategy on the same graph.
            for strategy_name, strategy in strategies.items():
                policy = ModularPolicy(
                    module_graph=module_graph,
                    query_strategy=strategy,
                )

                # NOTE: skip brute force for large graphs.
                if strategy_name == "Brute Force" and size > 18:
                    continue

                # Measure execution time.
                start_time = time.perf_counter()

                # Run the policy.
                action, query_cost = policy.get_action(state=state)

                # Record execution time.
                execution_time = time.perf_counter() - start_time

                correct = action == ground_truth_output
                task_cost = correct_answer_cost if correct else incorrect_answer_cost

                # Store metrics.
                results[strategy_name]["query_cost"][size].append(query_cost)
                results[strategy_name]["task_cost"][size].append(task_cost)
                results[strategy_name]["total_cost"][size].append(
                    task_cost + query_cost
                )
                results[strategy_name]["execution_time"][size].append(execution_time)

    return results


def plot_results(
    results: dict[str, dict[str, dict[int, list[float]]]], graph_sizes: list[int]
) -> None:
    """Create plots showing the performance of different querying strategies.

    Args:
        results: dictionary with results from run_experiment
        graph_sizes: list of graph sizes that were tested
    """
    # Set up figure with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)

    metrics = ["query_cost", "task_cost", "total_cost", "execution_time"]
    titles = ["Query Cost", "Task Cost", "Total Cost", "Execution Time (s)"]

    # Define distinct line styles, markers, and colors for each strategy
    styles = {
        "Always Query": {
            "color": "blue",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2,
        },
        "Never Query": {
            "color": "red",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2,
        },
        "Brute Force": {
            "color": "green",
            "linestyle": ":",
            "marker": "^",
            "linewidth": 2,
        },
        "MILP": {
            "color": "orange",
            "linestyle": "-.",
            "marker": "D",
            "linewidth": 2,
        },
    }

    # Plot each metric
    lines = []
    labels = []
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        for strategy_name in results:
            # Calculate mean for each graph size
            means: list[np.floating | float] = []
            for size in graph_sizes:
                try:
                    mean = np.mean(results[strategy_name][metric][size])
                except KeyError:
                    continue
                means.append(mean)

            # Plot the data with strategy-specific styling
            style = styles[strategy_name]
            line = ax.plot(
                graph_sizes[: len(means)],
                means,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=8,
                label=strategy_name,
            )

            # Only store lines and labels from the first subplot
            if i == 0:
                lines.append(line[0])
                labels.append(strategy_name)

        ax.set_title(title)
        ax.set_xlabel("Number of Graph Nodes")
        ax.set_ylabel("Time (s)" if metric == "execution_time" else "Cost")
        ax.grid(True, linestyle="--", alpha=0.7)

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
    os.makedirs("experiments/results", exist_ok=True)

    # Save the figure
    plt.savefig(
        "experiments/results/strategy_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main() -> None:
    """Run the experiment and generate plots."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]

    # Run the experiment
    results = run_experiment(graph_sizes=graph_sizes, num_trials=100)

    # Plot the results
    plot_results(results, graph_sizes)

    print("Experiment complete!")


if __name__ == "__main__":
    main()
