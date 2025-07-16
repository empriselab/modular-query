#!/usr/bin/env python3
"""Experiment to measure and compare performance of different querying
strategies."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from modular_query.modular_policy import ModularPolicy
from modular_query.module_utils import generate_random_and_gate_module_graph
from modular_query.modules import StateModule
from modular_query.query_strategies.brute_force import BruteForceQueryStrategy
from modular_query.query_strategies.graph_query import GraphQueryStrategy
from modular_query.query_strategies.mip import MIPQueryStrategy
from modular_query.query_strategies.never_query import NeverQueryStrategy


def run_experiment(
    graph_sizes: list[int],
    num_trials: int = 5,
    edge_probability: float = 0.3,
    correct_answer_cost: float = 0.0,
    incorrect_answer_cost: float = 1000.0,
    query_cost: float = 0.1,  # for uniform query-cost settings.
    seed: int = 0,
    workload_eps: float = 0.1,
    time_horizon: int = 5,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Run experiments with different graph sizes and querying strategies.

    Querying costs: not used for the polynomial module graph,
    but used for the logic gate module graph.
    """
    assert query_cost > 0, "Query cost for run_experiment should be positive."
    # Set up RNG.
    rng = np.random.default_rng(seed)

    # Initialize strategies.
    strategies = {
        "Never Query": NeverQueryStrategy(correct_answer_cost, incorrect_answer_cost),
        "Brute Force": BruteForceQueryStrategy(
            correct_answer_cost, incorrect_answer_cost
        ),
        "MIP": MIPQueryStrategy(correct_answer_cost, incorrect_answer_cost),
        "Graph Query": GraphQueryStrategy(
            correct_answer_cost,
            incorrect_answer_cost,
            workload_eps=workload_eps,
        ),
    }

    # Initialize results structure:
    # strategy -> metric -> graph_size -> list of values
    # These store means over time.
    results: dict[str, dict[str, dict[int, list[float]]]] = {}
    for strategy_name in strategies:
        results[strategy_name] = {
            "query_cost": {size: [] for size in graph_sizes},
            "task_cost": {size: [] for size in graph_sizes},
            "total_cost": {size: [] for size in graph_sizes},
            "execution_time": {size: [] for size in graph_sizes},
            "queries": {size: [] for size in graph_sizes},
        }

    # Run experiments for each graph size.
    for size in graph_sizes:
        print(f"Running experiments for graph size {size}")

        for _ in range(num_trials):

            # Generate a random logic-gate graph.
            # module_graph = generate_random_logic_gate_module_graph(
            #     num_modules=size,
            #     edge_probability=edge_probability,
            #     query_cost_sampler=query_cost_sampler,
            #     rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
            #     is_policy=True,
            #     num_incorrect_modules=1
            # )

            # Generate a random polynomial module graph.
            # module_graph = generate_random_polynomial_module_graph(
            #     num_modules=size,
            #     edge_probability=edge_probability,
            #     query_cost=0.1,
            #     rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
            #     num_incorrect_modules=1,
            #     incorrect_module_confidence=0.1,
            # )

            # Generate a random AND-gate graph.
            module_graph = generate_random_and_gate_module_graph(
                num_modules=size,
                edge_probability=edge_probability,
                query_cost=query_cost,
                rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                num_incorrect_modules=1,
            )

            # # Use random state inputs.
            # state = bool(rng.integers(0, 2))
            # Always set state to True for AND-gate graph.
            state = True

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
                # Reset strategy's internal state.
                strategy.reset()

                policy = ModularPolicy(
                    module_graph=module_graph,
                    query_strategy=strategy,
                )

                # Temporal loop.
                # Initialize accumulators.
                acc_query_cost = 0.0
                acc_task_cost = 0.0
                acc_execution_time = 0.0
                acc_queried = 0
                timesteps_elapsed = 0
                correct = False
                while timesteps_elapsed < time_horizon and not correct:
                    # Measure execution time.
                    start_time = time.perf_counter()

                    # Run the policy.
                    action, current_query_cost, queried = policy.get_action(state=state)
                    if queried:
                        assert (
                            current_query_cost > 0
                        ), "Query cost should be positive if we query!"

                    # Record execution time.
                    execution_time = time.perf_counter() - start_time

                    correct = action == ground_truth_output
                    task_cost = (
                        correct_answer_cost if correct else incorrect_answer_cost
                    )

                    # Add to accumulators.
                    acc_query_cost += current_query_cost
                    acc_task_cost += task_cost
                    acc_execution_time += execution_time
                    acc_queried += queried

                    # Increment timesteps elapsed.
                    timesteps_elapsed += 1

                # Compute temporal means.
                mean_query_cost = acc_query_cost / timesteps_elapsed
                mean_task_cost = acc_task_cost / timesteps_elapsed
                mean_execution_time = acc_execution_time / timesteps_elapsed
                mean_queries = acc_queried / timesteps_elapsed

                # Store metrics.
                results[strategy_name]["query_cost"][size].append(mean_query_cost)
                results[strategy_name]["task_cost"][size].append(mean_task_cost)
                results[strategy_name]["total_cost"][size].append(
                    mean_task_cost + mean_query_cost
                )
                results[strategy_name]["execution_time"][size].append(
                    mean_execution_time
                )
                results[strategy_name]["queries"][size].append(mean_queries)

    return results


def plot_results(
    results: dict[str, dict[str, dict[int, list[float]]]],
    graph_sizes: list[int],
    plot_name: str = "strategy_comparison.png",
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
        "Graph Query": {
            "color": "purple",
            "linestyle": "-",
            "marker": "x",
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
        "MIP": {
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
            # Calculate means and standard deviations for each graph size
            means: list[np.floating | float] = []
            stds: list[np.floating | float] = []
            for size in graph_sizes:
                try:
                    mean = np.mean(results[strategy_name][metric][size])
                    std = np.std(results[strategy_name][metric][size])
                except KeyError:
                    continue
                means.append(mean)
                stds.append(std)

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

            # Plot the standard deviation band
            ax.fill_between(
                graph_sizes[: len(means)],
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.3,
                label="±1 std dev",
                color=style["color"],
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
    plt.savefig(f"experiments/results/{plot_name}", dpi=300, bbox_inches="tight")
    plt.close()


def exp_vary_cquery() -> None:
    """Run the experiment with varying query cost."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]
    # graph_sizes = [3, 5, 10]
    time_horizon = 5

    # 6/12: vary c_query now, but keep workload_eps=1.0
    workload_eps = 1.0
    c_query_list = np.linspace(0.1, 1.0, 10)
    # c_query_list = [1.0]

    for c_query in c_query_list:
        print(f"Running experiments with c_query = {c_query:.2f}")
        results = run_experiment(
            graph_sizes=graph_sizes,
            num_trials=100,
            query_cost=c_query,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            workload_eps=workload_eps,
            time_horizon=time_horizon,
        )

        # Plot the results
        plot_results(
            results,
            graph_sizes,
            plot_name=f"strategy_comparison_c_query_{c_query:.2f}.png",
        )

    print("Experiment with varying query cost complete!")


def main() -> None:
    """Run the experiment and generate plots."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]

    # Run the experiment
    # Original setting.
    results = run_experiment(graph_sizes=graph_sizes, num_trials=100)

    # Running with querying cost between 0 and 1, and with varying workload epsilon.
    # Querying costs in [1e-3, 1.0] and task reward is binary (0 or 1).
    # We should see behavior interpolate between always query (workload-eps = 0)
    # and never query (workload-eps = 1).
    # workload_epsilons_small = np.linspace(0, 1.0, 11)
    # workload_epsilons_large = np.linspace(2.0, 10.0, 9)
    # workload_epsilons = np.concatenate(
    #     (workload_epsilons_small, workload_epsilons_large)
    # )
    workload_epsilons = [1.0]
    # workload_epsilons = workload_epsilons_large

    # Time horizon. 5 by default.
    time_horizon = 5
    # time_horizon = 1

    for workload_eps in workload_epsilons:
        print(f"Running experiments with workload_eps = {workload_eps:.2f}")
        results = run_experiment(
            graph_sizes=graph_sizes,
            num_trials=100,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            workload_eps=workload_eps,
            time_horizon=time_horizon,
        )

        # Plot the results
        plot_results(
            results,
            graph_sizes,
            plot_name=f"strategy_comparison_eps_{workload_eps:.2f}.png",
        )

    # Run the experiment with varying query cost.
    exp_vary_cquery()

    print("Experiment complete!")


if __name__ == "__main__":
    main()
