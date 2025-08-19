#!/usr/bin/env python3
"""Experiment to measure and compare performance of different querying
strategies."""

import abc
import argparse
import pickle as pkl
import time
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_results import plot_results
from modular_query.modular_policy import ModularPolicy
from modular_query.module_utils import generate_random_and_gate_module_graph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.binary_tree_query import BinaryTreeQueryStrategy
from modular_query.query_strategies.brute_force import BruteForceQueryStrategy
from modular_query.query_strategies.graph_query import GraphQueryStrategy
from modular_query.query_strategies.mip import MIPQueryStrategy
from modular_query.query_strategies.never_query import NeverQueryStrategy
from modular_query.utils import print_and_log

# Set matplotlib backend to avoid tkinter conflicts with Pyomo
matplotlib.use("Agg")  # Use non-interactive backend


def product_of_confidences(confidences: dict[Module, float]) -> float:
    """Compute the product of confidences."""
    product = 1.0
    for conf in confidences.values():
        product *= conf
    return product


def sum_of_uncertainties(confidences: dict[Module, float]) -> float:
    """Compute the sum of uncertainties."""
    # Uncertainty is 1 - confidence.
    uncertainty_sum = 0.0
    for conf in confidences.values():
        uncertainty_sum += 1.0 - conf
    return uncertainty_sum


class TerminationCondition(abc.ABC):
    """Base class for termination conditions in query loops."""

    def __init__(self, name: str) -> None:
        """Initialize the termination condition with a name."""
        self.name = name

    @abc.abstractmethod
    def evaluate(
        self,
        new_confidences: dict[Module, float],
        old_confidences: dict[Module, float],
        queried_module: Module,
        current_query_cost: float,
    ) -> bool:
        """Evaluate whether to terminate the query loop."""


class ConservativeTerminationCondition(TerminationCondition):
    """Termination condition that stops when product of confidences exceeds
    threshold."""

    def __init__(self, thresh: float) -> None:
        """Initialize with confidence threshold."""
        super().__init__("conservative")
        self.thresh = thresh

    def evaluate(
        self,
        new_confidences: dict[Module, float],
        old_confidences: dict[Module, float],
        queried_module: Module,
        current_query_cost: float,
    ) -> bool:
        return product_of_confidences(new_confidences) > self.thresh


class Balanced2TerminationCondition(TerminationCondition):
    """Termination condition for balanced-2 variant (terminates when confidence
    gain from querying is less than query cost)"""

    def __init__(self) -> None:
        """Initialize the balanced-2 termination condition."""
        super().__init__("balanced-2")

    def evaluate(
        self,
        new_confidences: dict[Module, float],
        old_confidences: dict[Module, float],
        queried_module: Module,
        current_query_cost: float,
    ) -> bool:
        return 1 - old_confidences[queried_module] < current_query_cost


class DummyTerminationCondition(TerminationCondition):
    """Dummy termination condition that never terminates."""

    def __init__(self) -> None:
        """Initialize the dummy termination condition."""
        super().__init__("dummy")

    def evaluate(
        self,
        new_confidences: dict[Module, float],
        old_confidences: dict[Module, float],
        queried_module: Module,
        current_query_cost: float,
    ) -> bool:
        return False


def run_experiment(
    graph_sizes: list[int],
    num_trials: int = 5,
    edge_probability: float = 0.3,
    correct_answer_cost: float = 0.0,
    incorrect_answer_cost: float = 1000.0,
    query_cost: float = 0.1,  # for uniform query-cost settings.
    seed: int = 0,
    workload_eps: float = 0.1,
    incorrect_module_count: np.ndarray | None = None,
    variant: str = "balanced",
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Run experiments with different graph sizes and querying strategies.

    Querying costs: not used for the polynomial module graph,
    but used for the logic gate module graph.

    incorrect_module_count: array where each value
    is the number of trials with that many incorrect modules.
    sum(incorrect_module_count) must = num_trials.

    Time horizon is now set based on the graph size
    (in particular, T = 2 * num_modules)
    """
    assert query_cost > 0, "Query cost for run_experiment should be positive."

    # If incorrect_module_count is None, set to a default value.
    # (all trials will have 1 failure.)
    if incorrect_module_count is None:
        incorrect_module_count = np.array([0, num_trials])

    # Print variant.
    print_and_log(f"Running experiment with variant {variant}")

    # Set up RNG.
    rng = np.random.default_rng(seed)

    # Variant-specific parameters.
    # Set termination conditions for 'conservative' and 'balanced-2' variants.
    loop_termination_condition: TerminationCondition
    pre_make_query_termination_condition: TerminationCondition

    if variant == "conservative":
        loop_termination_condition = ConservativeTerminationCondition(thresh=0.5)
        pre_make_query_termination_condition = DummyTerminationCondition()
    elif variant == "balanced-2":
        loop_termination_condition = DummyTerminationCondition()
        pre_make_query_termination_condition = Balanced2TerminationCondition()
    else:
        # Default case - these won't be used for other variants
        loop_termination_condition = DummyTerminationCondition()
        pre_make_query_termination_condition = DummyTerminationCondition()

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
        "Binary Tree Query": BinaryTreeQueryStrategy(
            correct_answer_cost,
            incorrect_answer_cost,
        ),
    }

    # Store timing info for binary tree query.
    binary_tree_query_timing_info: dict[str, dict[int, list[float]]] = {
        "t_create_graph": {size: [] for size in graph_sizes},
        "t_run_a_star": {size: [] for size in graph_sizes},
    }
    # Store timing info for MIP query.
    mip_query_timing_info: dict[str, dict[int, list[float]]] = {
        "t_construct_problem": {size: [] for size in graph_sizes},
        "t_solve_problem": {size: [] for size in graph_sizes},
    }

    # Initialize results structure:
    # strategy -> metric -> graph_size -> list of values
    # These store means over time.
    results: dict[str, dict[str, dict[int, list[float]]]] = {}
    metric_names = [
        "query_cost",
        "query_cost_total",
        "task_cost",
        "total_cost",
        "proxy_obj_1",
        "proxy_obj_2",
        "execution_time",
        "execution_time_total",
        "mean_queries",
        "total_queries",
        "total_correct",
        "total_timesteps",
        "total_executions",
        "total_failed_attempts",
        "total_task_cost",
        "total_get_action_calls",
    ]
    for strategy_name in strategies:
        results[strategy_name] = {
            metric_name: {size: [] for size in graph_sizes}
            for metric_name in metric_names
        }

    # Create cumulative distribution of incorrect module counts.
    incorrect_module_counts_cumsum = np.cumsum(incorrect_module_count)

    # Run experiments for each graph size.
    for size in graph_sizes:
        print(f"Running experiments for graph size {size}")

        # Reset num incorrect modules for each graph size.
        num_incorrect_modules = 0

        # Set time horizon based on graph size.
        time_horizon = 2 * size

        for trial_idx in range(num_trials):
            # If we exceeded the current cumulative incorrect module count,
            # increment num_incorrect_modules.
            while trial_idx >= incorrect_module_counts_cumsum[num_incorrect_modules]:
                num_incorrect_modules += 1

            # Generate a random AND-gate graph.
            module_graph = generate_random_and_gate_module_graph(
                num_modules=size,
                edge_probability=edge_probability,
                query_cost=query_cost,
                rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                num_incorrect_modules=num_incorrect_modules,
            )

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
                expert_query_module_names=all_queryable_module_names,
                expert_values_cache={},
            )
            ground_truth_output = computed_values[module_graph.leaf]

            # Run each strategy on the same graph.
            for strategy_name, strategy in strategies.items():
                # Reset strategy's internal state.
                strategy.reset()

                policy = ModularPolicy(
                    module_graph=module_graph, query_strategy=strategy, verbose=False
                )

                # Temporal loop.
                # Initialize accumulators.
                # Timesteps: increment every time we query or execute.
                # Executions: increment every time we execute only.
                acc_query_cost = 0.0
                acc_task_cost = 0.0
                acc_proxy_obj_1 = 0.0  # just the task part of the proxy objective.
                acc_proxy_obj_2 = 0.0  # just the task part of the proxy objective.
                acc_execution_time = 0.0
                acc_get_action_calls = 0
                acc_queried = 0
                timesteps_elapsed = 0
                num_executions = 0
                correct = False

                # Strategy-specific temporal accumulators.
                if strategy_name == "Binary Tree Query":
                    acc_t_create_graph = 0.0
                    acc_t_run_a_star = 0.0
                elif strategy_name == "MIP":
                    acc_t_construct_problem = 0.0
                    acc_t_solve_problem = 0.0

                # If greedy, do an initial forward pass + execution.
                if variant == "greedy":  # Measure execution time solely for get_action.
                    start_time = time.perf_counter()

                    # Forward pass only.
                    action, _, computed_confidences = policy.forward_pass_only(state)

                    computation_time = time.perf_counter() - start_time
                    acc_execution_time += computation_time

                    # Execute action.
                    correct = action == ground_truth_output
                    timesteps_elapsed += 1
                    num_executions += 1

                    task_cost = (
                        correct_answer_cost if correct else incorrect_answer_cost
                    )
                    # Update accumulators.
                    acc_task_cost += task_cost
                    acc_proxy_obj_1 += 1 - product_of_confidences(computed_confidences)
                    acc_proxy_obj_2 += sum_of_uncertainties(computed_confidences)

                while timesteps_elapsed < time_horizon and not correct:
                    start_time = time.perf_counter()

                    # Initialize number of queries in this execution loop.
                    num_queries_in_loop = 0

                    # Run the policy.
                    (
                        action,
                        current_query_cost,
                        queried,
                        queried_module,
                        pre_query_confidences,
                        post_query_confidences,
                        timing_info,
                    ) = policy.get_action(state=state)
                    computation_time = time.perf_counter() - start_time
                    acc_get_action_calls += 1
                    acc_execution_time += computation_time
                    if queried:
                        assert (
                            current_query_cost > 0
                        ), "Query cost should be positive if we query!"
                    # Increment accumulators.
                    acc_query_cost += current_query_cost
                    acc_proxy_obj_1 += 1 - product_of_confidences(
                        post_query_confidences
                    )
                    acc_proxy_obj_2 += sum_of_uncertainties(post_query_confidences)
                    acc_queried += queried
                    timesteps_elapsed += 1 if queried else 0
                    # If we exceed time_horizon, break.
                    if timesteps_elapsed >= time_horizon:
                        break
                    num_queries_in_loop += 1 if queried else 0

                    # Conservative runs this loop multiple times
                    # (until we decide not to query.)
                    start_time = time.perf_counter()
                    if variant in ("conservative", "balanced-2"):
                        while (
                            queried
                            and queried_module is not None
                            and not loop_termination_condition.evaluate(
                                new_confidences=post_query_confidences,
                                old_confidences=pre_query_confidences,
                                queried_module=queried_module,
                                current_query_cost=current_query_cost,
                            )
                        ):
                            # Then, do forward pass/query algorithm call.
                            (
                                action,
                                current_query_cost,
                                queried,
                                queried_module,
                                pre_query_confidences,
                                post_query_confidences,
                                timing_info,
                            ) = policy.get_action(state=state)
                            acc_get_action_calls += 1
                            acc_execution_time += computation_time
                            if queried:
                                assert (
                                    current_query_cost > 0
                                ), "Query cost should be positive if we query!"
                            # Balanced-2 actually has to intervene *here*
                            # i.e. - after we decide what to query,
                            # but before we actually make the query
                            # (i.e. we need to distinguish between
                            # get-query and make-query)
                            if (
                                queried
                                and queried_module is not None
                                and pre_make_query_termination_condition.evaluate(
                                    new_confidences=post_query_confidences,
                                    old_confidences=pre_query_confidences,
                                    queried_module=queried_module,
                                    current_query_cost=current_query_cost,
                                )
                            ):
                                break
                            # Increment accumulators and timesteps elapsed.
                            acc_query_cost += current_query_cost
                            acc_proxy_obj_1 += 1 - product_of_confidences(
                                post_query_confidences
                            )
                            acc_proxy_obj_2 += sum_of_uncertainties(
                                post_query_confidences
                            )
                            acc_queried += queried
                            timesteps_elapsed += 1 if queried else 0
                            # If we exceed time_horizon, break.
                            if timesteps_elapsed >= time_horizon:
                                break
                            num_queries_in_loop += 1 if queried else 0

                    # Measure execution time for the entire loop.
                    computation_time = time.perf_counter() - start_time
                    acc_execution_time += computation_time

                    # If we exceed time_horizon, break.
                    if timesteps_elapsed >= time_horizon:
                        break

                    # Execute action and increment time only.
                    correct = action == ground_truth_output
                    timesteps_elapsed += 1
                    num_executions += 1

                    task_cost = (
                        correct_answer_cost if correct else incorrect_answer_cost
                    )

                    # Add to accumulators.
                    acc_task_cost += task_cost

                    # Strategy-specific temporal accumulation.
                    if strategy_name == "Binary Tree Query":
                        assert timing_info is not None
                        acc_t_create_graph += timing_info["t_create_graph"]
                        acc_t_run_a_star += timing_info["t_run_a_star"]
                    elif strategy_name == "MIP":
                        assert timing_info is not None
                        acc_t_construct_problem += timing_info["t_construct_problem"]
                        acc_t_solve_problem += timing_info["t_solve_problem"]

                # Compute temporal means.
                mean_query_cost = acc_query_cost / timesteps_elapsed
                mean_task_cost = acc_task_cost / timesteps_elapsed
                mean_proxy_obj_1 = acc_proxy_obj_1 / timesteps_elapsed
                mean_proxy_obj_2 = acc_proxy_obj_2 / timesteps_elapsed
                mean_execution_time = acc_execution_time / timesteps_elapsed
                mean_queries = acc_queried / timesteps_elapsed

                # Strategy-specific temporal means.
                mean_t_create_graph = 0.0
                mean_t_run_a_star = 0.0
                mean_t_construct_problem = 0.0
                mean_t_solve_problem = 0.0
                if strategy_name == "Binary Tree Query":
                    mean_t_create_graph = acc_t_create_graph / timesteps_elapsed
                    mean_t_run_a_star = acc_t_run_a_star / timesteps_elapsed
                elif strategy_name == "MIP":
                    mean_t_construct_problem = (
                        acc_t_construct_problem / timesteps_elapsed
                    )
                    mean_t_solve_problem = acc_t_solve_problem / timesteps_elapsed
                # Store metrics.
                results[strategy_name]["query_cost"][size].append(mean_query_cost)
                results[strategy_name]["query_cost_total"][size].append(acc_query_cost)
                results[strategy_name]["task_cost"][size].append(mean_task_cost)
                results[strategy_name]["total_cost"][size].append(
                    mean_task_cost + mean_query_cost
                )
                results[strategy_name]["proxy_obj_1"][size].append(
                    mean_proxy_obj_1 + mean_query_cost
                )
                results[strategy_name]["proxy_obj_2"][size].append(
                    mean_proxy_obj_2 + mean_query_cost
                )
                results[strategy_name]["execution_time"][size].append(
                    mean_execution_time
                )
                results[strategy_name]["execution_time_total"][size].append(
                    acc_execution_time
                )
                results[strategy_name]["mean_queries"][size].append(mean_queries)
                results[strategy_name]["total_queries"][size].append(acc_queried)
                results[strategy_name]["total_correct"][size].append(correct)
                results[strategy_name]["total_timesteps"][size].append(
                    timesteps_elapsed
                )
                results[strategy_name]["total_executions"][size].append(num_executions)
                results[strategy_name]["total_task_cost"][size].append(acc_task_cost)
                results[strategy_name]["total_failed_attempts"][size].append(
                    num_executions - correct
                )
                results[strategy_name]["total_get_action_calls"][size].append(
                    acc_get_action_calls
                )
                # Store timing info for binary tree query.
                if strategy_name == "Binary Tree Query":
                    binary_tree_query_timing_info["t_create_graph"][size].append(
                        mean_t_create_graph
                    )
                    binary_tree_query_timing_info["t_run_a_star"][size].append(
                        mean_t_run_a_star
                    )

                # Store timing info for MIP query.
                if strategy_name == "MIP":
                    mip_query_timing_info["t_construct_problem"][size].append(
                        mean_t_construct_problem
                    )
                    mip_query_timing_info["t_solve_problem"][size].append(
                        mean_t_solve_problem
                    )

    # Print and log the total number of get_action calls.
    print_and_log(
        f"Total number of get_action calls: Brute Force, graph size 100:"
        f"{results['Brute Force']['total_get_action_calls'][100]}"
    )

    # Print and log the timing info for binary tree query.
    print_and_log("Timing info for binary tree query:")
    for size in graph_sizes:
        print_and_log(f"Size {size}:")
        print_and_log(
            f"  t_create_graph:"
            f"{np.mean(binary_tree_query_timing_info['t_create_graph'][size]):.6f}"
            f"±{np.std(binary_tree_query_timing_info['t_create_graph'][size]):.6f}"
        )
        print_and_log(
            f"  t_run_a_star:"
            f"{np.mean(binary_tree_query_timing_info['t_run_a_star'][size]):.6f}"
            f"±{np.std(binary_tree_query_timing_info['t_run_a_star'][size]):.6f}"
        )
    # Make a quick plot of the timing info (as a function of graph size).
    plt.plot(
        graph_sizes,
        [
            np.mean(binary_tree_query_timing_info["t_create_graph"][size])
            for size in graph_sizes
        ],
        label="t_create_graph",
    )
    plt.plot(
        graph_sizes,
        [
            np.mean(binary_tree_query_timing_info["t_run_a_star"][size])
            for size in graph_sizes
        ],
        label="t_run_a_star",
    )
    plt.legend()
    plt.savefig("experiments/results/binary_tree_query_timing.png")
    plt.close()

    # Print, log, plot the timing info for MIP query.
    print_and_log("Timing info for MIP query:")
    for size in graph_sizes:
        print_and_log(f"Size {size}:")
        print_and_log(
            f"  t_construct_problem:"
            f"{np.mean(mip_query_timing_info['t_construct_problem'][size]):.6f}"
            f"± {np.std(mip_query_timing_info['t_construct_problem'][size]):.6f}"
        )
        print_and_log(
            f"  t_solve_problem: "
            f"{np.mean(mip_query_timing_info['t_solve_problem'][size]):.6f}"
            f"± {np.std(mip_query_timing_info['t_solve_problem'][size]):.6f}"
        )
    # Make a quick plot of the timing info (as a function of graph size).
    plt.plot(
        graph_sizes,
        [
            np.mean(mip_query_timing_info["t_construct_problem"][size])
            for size in graph_sizes
        ],
        label="t_construct_problem",
    )
    plt.plot(
        graph_sizes,
        [
            np.mean(mip_query_timing_info["t_solve_problem"][size])
            for size in graph_sizes
        ],
        label="t_solve_problem",
    )
    plt.legend()
    plt.savefig("experiments/results/mip_query_timing.png")
    plt.close()

    return results


def exp_vary_cquery(variant: str, config: dict[str, Any]) -> None:
    """Run the experiment with varying query cost, where by default all trials
    have 1 'incorrect' module.

    Required config keys: graph_sizes, num_trials, workload_eps, c_query_list
    """

    for c_query in config["c_query_list"]:
        print_and_log(f"Running experiments with c_query = {c_query:.2f}")
        results = run_experiment(
            graph_sizes=config["graph_sizes"],
            num_trials=config["num_trials"],
            query_cost=c_query,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            workload_eps=config["workload_eps"],
            incorrect_module_count=np.array([0, config["num_trials"]]),
            variant=variant,
        )

        # Plot the results
        metrics_to_plot = [
            "query_cost",
            "task_cost",
            "total_cost",
            "proxy_obj_1",
            "execution_time",
            "total_correct",
            "total_executions",
        ]
        plot_results(
            results,
            config["graph_sizes"],
            metrics_to_plot=metrics_to_plot,
            plot_name=f"strategy_comparison_c_query_{c_query:.2f}.png",
        )

    print("Experiment with varying query cost complete!")


def exp_vary_num_failures(variant: str, config: dict[str, Any]) -> None:
    """Run the experiment with varying number of incorrect modules; also saves
    results to a pickle file and plots the results.

    Required config keys: graph_sizes, num_trials, c_query, num_failures_list
    """
    for num_failures in config["num_failures_list"]:
        # Convert num_failures to a one-hot vector.
        incorrect_module_count = np.zeros(num_failures + 1)
        incorrect_module_count[-1] = config["num_trials"]
        # Don't include graph sizes that are too small for the number of failures.
        graph_sizes_to_use = [
            size for size in config["graph_sizes"] if size > num_failures
        ]
        print_and_log(f"Running experiments with num_failures = {num_failures}")
        results = run_experiment(
            graph_sizes=graph_sizes_to_use,
            num_trials=config["num_trials"],
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            query_cost=config["c_query"],
            incorrect_module_count=incorrect_module_count,
            workload_eps=1.0,
            variant=variant,
        )

        # Save results to a pickle file.
        with open(
            "experiments/results/strategy_comparison_num_failures_"
            f"{variant}_{num_failures}.pkl",
            "wb",
        ) as f:
            pkl.dump(results, f)

        # Plot the results
        metrics_to_plot = [
            "query_cost",
            "task_cost",
            "total_cost",
            "proxy_obj_1",
            "execution_time",
            "total_correct",
            "total_executions",
        ]
        plot_results(
            results,
            graph_sizes_to_use,
            metrics_to_plot=metrics_to_plot,
            plot_name=f"strategy_comparison_num_failures_{variant}_{num_failures}.png",
        )


def exp_mixed_failure_population(variant: str, config: dict[str, Any]) -> None:
    """Run the experiment with a mixed failure population.

    Required config keys: graph_sizes, num_trials, c_query, failure_populations
    """
    for failure_population in config["failure_populations"]:
        print_and_log(
            f"Running experiments with failure population {failure_population}"
        )
        results = run_experiment(
            graph_sizes=config["graph_sizes"],
            num_trials=config["num_trials"],
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            query_cost=config["c_query"],
            incorrect_module_count=failure_population,
            variant=variant,
        )

        # Plot the results
        metrics_to_plot = [
            "query_cost",
            "task_cost",
            "total_cost",
            "proxy_obj_1",
            "execution_time",
            "total_correct",
            "total_executions",
        ]
        plot_results(
            results,
            config["graph_sizes"],
            metrics_to_plot=metrics_to_plot,
            plot_name=f"strategy_comparison_mixed_failure_population"
            f"_{variant}_{failure_population}.png",
        )


def main(variant: str) -> None:
    """Run the experiment and generate plots."""
    # Run the experiment with varying query cost.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "time_horizon": 5,
    #     "workload_eps": 1.0,
    #     "c_query_list": [0.1],
    #     "num_trials": 100,
    # }
    # exp_vary_cquery(variant)

    # Run the experiment with 1 failure.
    config = {
        "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
        "num_trials": 100,
        "c_query": 0.08,
        "num_failures_list": [1],
    }
    if variant == "all-variants":
        for variant_to_use in ["balanced", "greedy", "conservative", "balanced-2"]:
            exp_vary_num_failures(variant_to_use, config)
    else:
        exp_vary_num_failures(variant, config)

    # Run the experiment with a mixed failure population.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "failure_populations":
    #     [np.array([100, 0]), np.array([0, 100]), np.array([50, 50])],
    # }
    # exp_mixed_failure_population(variant)

    print("Experiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        choices=["balanced", "greedy", "conservative", "balanced-2", "all-variants"],
        required=True,
    )
    args = parser.parse_args()
    main(args.variant)
