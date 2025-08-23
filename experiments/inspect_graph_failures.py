#!/usr/bin/env python3
"""Experiment where we iterate over different graph sizes and trials to answer the
question: How likely is it that a graph with a single module failure will lead to
global failure?"""

import matplotlib.pyplot as plt
import numpy as np

from modular_query.modular_policy import ModularPolicy
from modular_query.module_utils import generate_random_module_graph
from modular_query.modules import StateModule
from modular_query.query_strategies.never_query import NeverQueryStrategy
from modular_query.utils import print_and_log


def run_experiment(
    graph_sizes: list[int],
    num_trials: int = 5,
    edge_probability: float = 0.3,
    correct_answer_cost: float = 0.0,
    incorrect_answer_cost: float = 1000.0,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Run experiments with different graph sizes and querying strategies."""
    # Set up RNG.
    rng = np.random.default_rng(seed)

    # Initialize strategies.
    strategies = {
        "Never Query": NeverQueryStrategy(correct_answer_cost, incorrect_answer_cost),
    }

    # Initialize results structure:
    # strategy -> metric -> graph_size -> list of values
    # These store means over time.
    results: dict[str, dict[str, dict[int, list[float]]]] = {}
    for strategy_name in strategies:
        results[strategy_name] = {
            "task_cost": {size: [] for size in graph_sizes},
        }

    # Run experiments for each graph size.
    for size in graph_sizes:
        print_and_log(f"Running experiments for graph size {size}")

        for _ in range(num_trials):

            # Generate a random graph.
            # module_graph = generate_random_logic_gate_module_graph(
            #     num_modules=size,
            #     edge_probability=edge_probability,
            #     query_cost_sampler=query_cost_sampler,
            #     rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
            #     is_policy=True,
            #     num_incorrect_modules=1
            # )

            # Switch to polynomial graphs.
            # module_graph = generate_random_polynomial_module_graph(
            #     num_modules=size,
            #     edge_probability=edge_probability,
            #     rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
            #     num_incorrect_modules=1,
            #     query_cost=0.1,
            #     incorrect_module_confidence=0.1,
            # )

            # Switch to AND gate graphs.
            module_graph = generate_random_module_graph(
                num_modules=size,
                edge_probability=edge_probability,
                rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                num_incorrect_modules=1,
                query_cost=0.1,
                incorrect_module_confidence=0.1,
                redundancy="AND",
            )

            # We always want the state input to be 1 (True)
            state = True

            # Get the correct expected output.
            if verbose:
                print_and_log("Generating ground truth output.")
            all_queryable_module_names = {
                m.get_name() for m in module_graph.get_modules()
            }
            all_queryable_module_names.remove("state")
            assert isinstance(module_graph.root, StateModule)
            module_graph.root.set_state(state)
            computed_values, _, _ = module_graph.compute_values(
                expert_query_module_names=all_queryable_module_names,
                expert_values_cache={},
            )
            ground_truth_output = computed_values[module_graph.leaf]

            # Run each strategy on the same graph.
            for strategy_name, strategy in strategies.items():
                policy = ModularPolicy(
                    module_graph=module_graph,
                    query_strategy=strategy,
                )

                # Run the policy.
                action, _, _, _, _, _, _ = policy.get_action(state=state)

                correct = action == ground_truth_output
                task_cost = correct_answer_cost if correct else incorrect_answer_cost

                # Store metrics.
                results[strategy_name]["task_cost"][size].append(task_cost)

    return results


def plot_results(
    results: dict[str, dict[str, dict[int, list[float]]]],
    graph_sizes: list[int],
    num_trials: int = 5,
) -> None:
    """Plot the results of the experiment (graph size vs mean task cost)."""
    plt.figure(figsize=(10, 6))

    for strategy_name, metrics in results.items():
        task_costs = metrics["task_cost"]
        means = [np.mean(task_costs[size]) for size in graph_sizes]
        # plot with markers
        plt.plot(graph_sizes, means, marker="o", label=strategy_name)

    plt.xlabel("Graph Size")
    plt.ylabel("Mean Task Cost")
    plt.title(f"Mean Task Cost vs Graph Size (Over {num_trials} Trials)")
    plt.legend()
    plt.grid()
    plt.savefig("experiments/results/mean_task_cost_vs_graph_size.png")
    plt.show()


def main() -> None:
    """Run the experiment and generate plots."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]
    # graph_sizes = [3, 5]
    # graph_sizes = [3, 5, 10, 15]

    num_trials = 100
    # num_trials = 1

    print("Running experiments.")
    results = run_experiment(
        graph_sizes=graph_sizes,
        num_trials=num_trials,
        correct_answer_cost=0.0,
        incorrect_answer_cost=1.0,
    )

    plot_results(results, graph_sizes, num_trials=num_trials)

    print("Experiment complete!")


if __name__ == "__main__":
    main()
