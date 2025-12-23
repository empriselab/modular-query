#!/usr/bin/env python3
"""Experiment to measure and compare performance of different querying
strategies."""

import abc
import argparse
import itertools
import json
import multiprocessing as mp
import pickle as pkl
import time
from datetime import datetime
from functools import partial
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from modular_query.modular_policy import ModularPolicy
from modular_query.module_utils import (
    generate_random_module_graph,
    generate_random_top_bottom_module_graph,
    task_cost_proxy,
)
from modular_query.modules import Module, StateModule
from modular_query.plot_utils import plot_results
from modular_query.query_strategies.binary_tree_query import BinaryTreeQueryStrategy
from modular_query.query_strategies.brute_force import BruteForceQueryStrategy
from modular_query.query_strategies.confidence_query import ConfidenceQueryStrategy
from modular_query.query_strategies.graph_query import GraphQueryStrategy
from modular_query.query_strategies.mip import MIPQueryStrategy
from modular_query.query_strategies.never_query import NeverQueryStrategy
from modular_query.utils import (
    print_and_log,
    product_of_confidences,
    sum_of_uncertainties,
)

# Set matplotlib backend to avoid tkinter conflicts with Pyomo
matplotlib.use("Agg")  # Use non-interactive backend


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
        and_modules: set[str],
        or_modules: set[str],
        module_name_to_module: dict[str, Module],
    ) -> bool:
        """Evaluate whether to terminate the query loop."""


class ConservativeTerminationCondition(TerminationCondition):
    """Termination condition that stops when the proxy task cost is smaller
    than a threshold."""

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
        and_modules: set[str],
        or_modules: set[str],
        module_name_to_module: dict[str, Module],
    ) -> bool:
        return (
            task_cost_proxy(
                new_confidences, module_name_to_module, and_modules, or_modules
            )
            < self.thresh
        )


class Balanced2TerminationCondition(TerminationCondition):
    """Termination condition for balanced-2 variant (terminates when confidence
    gain from querying is less than query cost).

    Assumes that confidence of a module after querying is 1.0.
    """

    def __init__(self) -> None:
        """Initialize the balanced-2 termination condition."""
        super().__init__("balanced-2")

    def evaluate(
        self,
        new_confidences: dict[Module, float],
        old_confidences: dict[Module, float],
        queried_module: Module,
        current_query_cost: float,
        and_modules: set[str],
        or_modules: set[str],
        module_name_to_module: dict[str, Module],
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
        and_modules: set[str],
        or_modules: set[str],
        module_name_to_module: dict[str, Module],
    ) -> bool:
        return False


def run_experiment(
    graph_sizes: list[int],
    num_trials: int = 5,
    edge_probability: float = 0.3,
    correct_answer_cost: float = 0.0,
    incorrect_answer_cost: float = 1000.0,
    correct_module_confidence: float = 1.0,
    incorrect_module_confidence: float = 0.1,
    query_cost: float = 0.1,  # for uniform query-cost settings.
    seed: int = 0,
    workload_eps: float = 0.1,
    incorrect_module_count: np.ndarray | None = None,
    variant: str = "balanced",
    dependency_structure: str = "all_AND",
    disable_mip: bool = False,
    expert_query_confidence: float = 1.0,
    query_cost_noise_width_fraction: float = 0.1,
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
        pre_make_query_termination_condition = (
            DummyTerminationCondition()
        )  # No longer needed, because checking is done in get_action()
    elif variant in ("greedy", "balanced"):
        # Default case - these won't be used for other variants
        loop_termination_condition = DummyTerminationCondition()
        pre_make_query_termination_condition = DummyTerminationCondition()
    else:
        raise ValueError(f"Invalid variant: {variant}")

    # Initialize strategies.
    # (initially with null and_modules and or_modules,
    # which will be set after the module graph is generated.)
    # NOTE: disabling other strategies for the rebuttal for now.
    strategies = {
        "Never Query": NeverQueryStrategy(correct_answer_cost, incorrect_answer_cost),
        "Brute Force": BruteForceQueryStrategy(
            correct_answer_cost, incorrect_answer_cost
        ),
        "Graph Query": GraphQueryStrategy(
            correct_answer_cost,
            incorrect_answer_cost,
            workload_eps=workload_eps,
        ),
        "Binary Tree Query": BinaryTreeQueryStrategy(
            correct_answer_cost,
            incorrect_answer_cost,
        ),
        "Confidence Query": ConfidenceQueryStrategy(
            correct_answer_cost,
            incorrect_answer_cost,
        ),
        # "Hybrid Graph Query": HybridGraphQueryStrategy(
        #     correct_answer_cost,
        #     incorrect_answer_cost,
        # ),
    }
    if not disable_mip:
        strategies["MIP"] = MIPQueryStrategy(correct_answer_cost, incorrect_answer_cost)

        # Store timing info for MIP query.
        mip_query_timing_info: dict[str, dict[int, list[float]]] = {
            "t_construct_problem": {size: [] for size in graph_sizes},
            "t_solve_problem": {size: [] for size in graph_sizes},
        }
    else:
        mip_query_timing_info = {}

    # Store timing info for binary tree query.
    binary_tree_query_timing_info: dict[str, dict[int, list[float]]] = {
        "t_create_graph": {size: [] for size in graph_sizes},
        "t_run_a_star": {size: [] for size in graph_sizes},
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

            # Generate a random module graph.
            if dependency_structure == "all_AND":
                module_graph = generate_random_module_graph(
                    num_modules=size,
                    edge_probability=edge_probability,
                    query_cost=query_cost,
                    rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                    num_incorrect_modules=num_incorrect_modules,
                    correct_module_confidence=correct_module_confidence,
                    incorrect_module_confidence=incorrect_module_confidence,
                    redundancy="AND",
                    expert_query_confidence=expert_query_confidence,
                    query_cost_noise_width_fraction=query_cost_noise_width_fraction,
                )
            elif dependency_structure == "all_OR":
                module_graph = generate_random_module_graph(
                    num_modules=size,
                    edge_probability=edge_probability,
                    query_cost=query_cost,
                    rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                    num_incorrect_modules=num_incorrect_modules,
                    correct_module_confidence=correct_module_confidence,
                    incorrect_module_confidence=incorrect_module_confidence,
                    redundancy="OR",
                    expert_query_confidence=expert_query_confidence,
                    query_cost_noise_width_fraction=query_cost_noise_width_fraction,
                )
            elif dependency_structure == "AND_then_OR":
                module_graph = generate_random_top_bottom_module_graph(
                    num_modules=size,
                    edge_probability=edge_probability,
                    query_cost=query_cost,
                    rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                    num_incorrect_modules=num_incorrect_modules,
                    correct_module_confidence=correct_module_confidence,
                    incorrect_module_confidence=incorrect_module_confidence,
                    gate_top="AND",
                    gate_bottom="OR",
                    expert_query_confidence=expert_query_confidence,
                    query_cost_noise_width_fraction=query_cost_noise_width_fraction,
                )
            elif dependency_structure == "OR_then_AND":
                module_graph = generate_random_top_bottom_module_graph(
                    num_modules=size,
                    edge_probability=edge_probability,
                    query_cost=query_cost,
                    rng=rng.spawn(1)[0],  # create a new RNG to avoid affecting main one
                    num_incorrect_modules=num_incorrect_modules,
                    correct_module_confidence=correct_module_confidence,
                    incorrect_module_confidence=incorrect_module_confidence,
                    gate_top="OR",
                    gate_bottom="AND",
                    expert_query_confidence=expert_query_confidence,
                    query_cost_noise_width_fraction=query_cost_noise_width_fraction,
                )
            else:
                raise ValueError(
                    f"Invalid dependency structure: {dependency_structure}"
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
                disable_expert_noise=True,
            )
            try:
                ground_truth_output = computed_values[module_graph.leaf]
            except KeyError:
                print(computed_values)
                print([m.get_name() for m in module_graph.get_modules()])
                print(f"topo order: {[m.get_name() for m in module_graph.topo_order]}")
                raise KeyError(
                    f"Module {module_graph.leaf.get_name()} not found in computed_values"
                )
            module_name_to_module = {
                m.get_name(): m for m in module_graph.get_modules()
            }

            # Run each strategy on the same graph.
            for strategy_name, strategy in strategies.items():
                # Reset strategy's internal state.
                strategy.reset()

                # Set the strategy's and_modules and or_modules.
                if dependency_structure == "all_AND":
                    strategy.and_modules = all_queryable_module_names
                elif dependency_structure == "all_OR":
                    strategy.or_modules = all_queryable_module_names

                policy = ModularPolicy(
                    module_graph=module_graph,
                    query_strategy=strategy,
                    verbose=False,
                    variant=variant,
                )

                # NOTE: skip graphquery for large graphs (>= 50)
                # because computation time exceeds 1 s.
                if strategy_name == "Graph Query" and size >= 50:
                    continue

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
                    # Measure execution time solely for get_action.
                    start_time = time.perf_counter()
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
                                and_modules=strategy.and_modules,
                                or_modules=strategy.or_modules,
                                module_name_to_module=module_name_to_module,
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
                                    and_modules=strategy.and_modules,
                                    or_modules=strategy.or_modules,
                                    module_name_to_module=module_name_to_module,
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
    # print_and_log(
    #     f"Total number of get_action calls: Brute Force, graph size 100:"
    #     f"{results['Brute Force']['total_get_action_calls'][100]}"
    # )

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

    if not disable_mip:
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


def run_single_grid_search_experiment(
    combo: tuple, combo_generator: list[tuple], variant: str, config: dict[str, Any]
) -> tuple[int, dict[str, dict[str, dict[int, list[float]]]], dict[str, Any]]:
    """Run a single experiment for the grid search.

    Args:
        combo: Tuple of (num_failures, confidence, redundancy, c_query)
        variant: The variant to use
        config: The configuration dictionary

    Returns:
        Tuple of (run_id, results, config_to_save)
    """
    (num_failures, confidence, redundancy, c_query, expert_query_confidence) = combo
    (correct_confidence, incorrect_confidence) = confidence

    print_and_log(
        f"Running experiment with num_failures={num_failures},"
        f"correct_confidence={correct_confidence},"
        f"incorrect_confidence={incorrect_confidence},"
        f"redundancy={redundancy}, c_query={c_query}, expert_query_confidence={expert_query_confidence}"
    )

    # Convert num_failures to a one-hot vector.
    incorrect_module_count = np.zeros(num_failures + 1)
    incorrect_module_count[-1] = config["num_trials"]

    # Use only the graph sizes that are larger than the number of failures.
    graph_sizes_to_use = [size for size in config["graph_sizes"] if size > num_failures]

    # Assume a constant query cost noise width fraction.
    query_cost_noise_width_fraction = config["query_cost_noise_width_fraction"]

    print_and_log(f"Query cost noise width fraction: {query_cost_noise_width_fraction}")

    results = run_experiment(
        graph_sizes=graph_sizes_to_use,
        num_trials=config["num_trials"],
        correct_answer_cost=0.0,
        incorrect_answer_cost=1.0,
        query_cost=c_query,
        workload_eps=1.0,
        incorrect_module_count=incorrect_module_count,
        variant=variant,
        dependency_structure=redundancy,
        correct_module_confidence=correct_confidence,
        incorrect_module_confidence=incorrect_confidence,
        expert_query_confidence=expert_query_confidence,
        disable_mip=True,
        query_cost_noise_width_fraction=query_cost_noise_width_fraction,
    )

    # Run ID is the index of the configuration in the grid.
    run_id = combo_generator.index(combo)

    # Save the variant and combo to a json file.
    config_to_save = {
        "run_id": run_id,
        "variant": variant,
        "num_failures": num_failures,
        "correct_confidence": correct_confidence,
        "incorrect_confidence": incorrect_confidence,
        "dependency_structure": redundancy,
        "c_query": c_query,
        "expert_query_confidence": expert_query_confidence,
        "query_cost_noise_width_fraction": query_cost_noise_width_fraction,
    }

    return run_id, results, config_to_save


def exp_grid_search_parallel(
    variant: str, config: dict[str, Any], num_processes: int = 0
) -> None:
    """Run the experiment over an experimental grid of parameters using
    multiprocessing.

    This is a parallelized version of exp_grid_search.

    Args:
        variant: The variant to use
        config: The configuration dictionary
        num_processes: Number of processes to use. If 0, uses CPU count.
    """
    if num_processes == 0:
        num_processes = mp.cpu_count()

    combo_generator = list(
        itertools.product(
            config["num_failures_list"],
            config["confidences_list"],
            config["redundancy_list"],
            config["c_query_list"],
            config["expert_query_confidence_list"],
        )
    )

    print_and_log(
        f"Running {len(combo_generator)} experiments using {num_processes} processes"
    )

    # Create a partial function with fixed variant and config
    run_single_experiment = partial(
        run_single_grid_search_experiment,
        variant=variant,
        config=config,
        combo_generator=combo_generator,
    )

    # Run experiments in parallel
    with mp.Pool(processes=num_processes) as pool:
        results_list = pool.map(run_single_experiment, combo_generator)

    # Save results
    # Run_ID is the index of the configuration in the grid.
    # Format with number of digits equal to the length of the combo_generator.
    run_id_length = len(str(len(combo_generator)))
    for run_id, results, config_to_save in results_list:
        # Save results to a pickle file
        with open(
            f"experiments/results/results_variant_{variant}_run_"
            f"{run_id:0{run_id_length}d}.pkl",
            "wb",
        ) as f:
            pkl.dump(results, f)

        # Save config to a json file
        with open(
            f"experiments/results/config_variant_{variant}_run_"
            f"{run_id:0{run_id_length}d}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_to_save, f, indent=4)

    print_and_log("All parallel experiments completed!")


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


def exp_vary_num_failures(
    variant: str, config: dict[str, Any], dependency_structure: str
) -> None:
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
            variant=variant,
            dependency_structure=dependency_structure,
        )

        # Save results to a pickle file.
        with open(
            "experiments/results/strategy_comparison_num_failures_"
            f"{variant}_{num_failures}_{dependency_structure}.pkl",
            "wb",
        ) as f:
            pkl.dump(results, f)

        # Proxy objective depends on the dependency structure.
        if dependency_structure == "all_AND":
            proxy_objective_name = "proxy_obj_1"
        elif dependency_structure == "all_OR":
            proxy_objective_name = "proxy_obj_2"
        else:
            raise NotImplementedError

        # Plot the results
        metrics_to_plot = [
            "query_cost",
            "task_cost",
            "total_cost",
            proxy_objective_name,
            "execution_time",
            "total_correct",
            "total_executions",
        ]
        plot_results(
            results,
            graph_sizes_to_use,
            metrics_to_plot=metrics_to_plot,
            plot_name=f"strategy_comparison_num_failures_"
            f"{variant}_{num_failures}_{dependency_structure}.png",
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


def exp_grid_search(variant: str, config: dict[str, Any]) -> None:
    """Run the experiment over an experimental grid of parameters.

    Have to disable MIP for now because it has a confidence sensitivity issue.

    Expected keys in config:
    - graph_sizes: list of graph sizes to use
    - num_failures_list: list of number of unconfident modules to use
    - confidences_list: list of confidences to use
    - redundancy_list: list of redundancies to use
    - c_query_list: list of query costs to use
    - num_trials: number of trials to run
    """
    combo_generator = itertools.product(
        config["num_failures_list"],
        config["confidences_list"],
        config["redundancy_list"],
        config["c_query_list"],
        config["expert_query_confidence_list"],
    )
    for combo in combo_generator:
        (num_failures, confidence, redundancy, c_query, expert_query_confidence) = combo
        (correct_confidence, incorrect_confidence) = confidence
        print_and_log(
            f"Running experiment with num_failures={num_failures},"
            f"correct_confidence={correct_confidence},"
            f"incorrect_confidence={incorrect_confidence},"
            f"redundancy={redundancy}, c_query={c_query}, expert_query_confidence={expert_query_confidence}"
        )
        # Convert num_failures to a one-hot vector.
        incorrect_module_count = np.zeros(num_failures + 1)
        incorrect_module_count[-1] = config["num_trials"]
        # Use only the graph sizes that are larger than the number of failures.
        graph_sizes_to_use = [
            size for size in config["graph_sizes"] if size > num_failures
        ]
        results = run_experiment(
            graph_sizes=graph_sizes_to_use,
            num_trials=config["num_trials"],
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            query_cost=c_query,
            workload_eps=1.0,
            incorrect_module_count=incorrect_module_count,
            variant=variant,
            dependency_structure=redundancy,
            correct_module_confidence=correct_confidence,
            incorrect_module_confidence=incorrect_confidence,
            disable_mip=True,
            expert_query_confidence=expert_query_confidence,
        )
        # Save results to a pickle file.
        # Generate a unique run ID based on current date and time.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(
            f"experiments/results/results_{run_id}.pkl",
            "wb",
        ) as f:
            pkl.dump(results, f)
        # Save the variant and combo to a json file.
        config_to_save = {
            "run_id": run_id,
            "variant": variant,
            "num_failures": num_failures,
            "correct_confidence": correct_confidence,
            "incorrect_confidence": incorrect_confidence,
            "redundancy": redundancy,
            "c_query": c_query,
            "expert_query_confidence": expert_query_confidence,
        }
        with open(
            f"experiments/results/config_{run_id}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(config_to_save, f, indent=4)


def main(variant: str) -> None:
    """Run the experiment and generate plots."""


    # CONDITIONAL ACCEPTANCE CONFIGURATIONS (same as original paper, but with a new baseline.)
    # Combining the old grid into one, hopefully this is the best idea.
    # Running without any noise in the query cost.
    config = {
        "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
        "num_trials": 100,
        "num_failures_list": [3],
        "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
        "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
        "c_query_list": [0.08, 0.16, 0.32, 0.64],
        "query_cost_noise_width_fraction": 0.0,
    }

    # REBUTTAL CONFIGURATIONS:
    # 11/10/2025: increasing query cost noise width fraction to 0.2, then 0.4, 0.6.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
    #     "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.32],
    #     "query_cost_noise_width_fraction": 0.6,
    # }
    # 11/10/2025: looking at tigher confidences (close to (0.8, 0.3) and (0.7, 0.4) )
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(0.8, 0.3), (0.75, 0.35), (0.7, 0.4), (0.65, 0.45), (0.6, 0.5)],
    #     "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.32],
    #     "query_cost_noise_width_fraction": 0.6,
    # }

    ### OLD CONFIGURATIONS:
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
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "c_query": 0.08,
    #     "num_failures_list": [1],
    # }
    # if variant == "all-variants":
    #     for variant_to_use in ["balanced", "greedy", "conservative", "balanced-2"]:
    #         exp_vary_num_failures(variant_to_use, config)
    # else:
    #     exp_vary_num_failures(variant, config)

    # Run the experiment with a mixed failure population.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "failure_populations":
    #     [np.array([100, 0]), np.array([0, 100]), np.array([50, 50])],
    # }
    # exp_mixed_failure_population(variant)

    # Noisy-experts test: we want to try c_expert = {1, 0.8, 0.6, 0.4}.
    config = {
        "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
        "num_trials": 100,
        "num_failures_list": [3],
        "confidences_list": [(1.0, 0.1)],
        "redundancy_list": ["all_AND"],
        "c_query_list": [0.32],
        "expert_query_confidence_list": [1.0, 0.8, 0.6, 0.4],
    }

    # Run the FULL grid-search experiment
    # (4 variant x 4 failure counts x 4 confidence counts x
    #  4 redundancy counts x 4 query costs = 1024 runs)
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [0, 1, 2, 3],
    #     "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
    #     "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.08, 0.16, 0.32, 0.64],
    # }
    # 9/29/2025: test with a smaller grid size. Only vary # of modules,
    # redundancies, and query costs.
    # (4 variant x 1 failure count x 1 confidence count x
    # 4 redundancy counts x 4 query costs = 64 runs)
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(1.0, 0.1)],
    #     "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.08, 0.16, 0.32, 0.64],
    # }
    # 9/30/2025: only vary confidences, fix everything else.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
    #     "redundancy_list": ["all_AND"],
    #     "c_query_list": [0.32],
    # }
    # 9/29/2025: testing varying redundancies.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(1.0, 0.1)],
    #     "redundancy_list": ["all_AND", "all_OR", "AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.32],
    # }

    # 9/14/2025: much smaller test, 'canonical' setting for feeding.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [3],
    #     "confidences_list": [(0.7, 0.4)],
    #     "redundancy_list": ["all_AND"],
    #     "c_query_list": [0.64],
    # }
    # simple test with different redundancies, but everything else fixed.
    # config = {
    #     "graph_sizes": [5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [1],
    #     # "confidences_list": [(1.0, 0.1)],
    #     "confidences_list": [(0.9, 0.2)],
    #     # "redundancy_list": ["all_AND"],
    #     # "redundancy_list": ["all_AND", "all_OR"],
    #     "redundancy_list": ["AND_then_OR", "OR_then_AND"],
    #     "c_query_list": [0.08],
    # }
    # smaller test.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [1,2,3],
    #     "confidences_list": [(0.9, 0.2)],
    #     "redundancy_list": ["AND"],
    #     "c_query_list": [0.08],
    # }
    # 8/27: specifically run with only 0.08 query cost and 'AND' networks for now.
    # will also only run with the balanced variant.
    # (want to do this as a mini-experiment prior to running the full grid search)
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [0, 1, 2, 3],
    #     "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
    #     "redundancy_list": ["AND"],
    #     "c_query_list": [0.08],
    # }
    # 8/28: run only with the above, but also only num_failures = 0.
    # config = {
    #     "graph_sizes": [3, 5, 10, 15, 18, 25, 50, 75, 100],
    #     "num_trials": 100,
    #     "num_failures_list": [0],
    #     "confidences_list": [(1.0, 0.1), (0.9, 0.2), (0.8, 0.3), (0.7, 0.4)],
    #     "redundancy_list": ["AND"],
    #     "c_query_list": [0.08],
    # }
    if variant == "all-variants":
        for variant_to_use in ["balanced", "greedy", "conservative", "balanced-2"]:
            exp_grid_search_parallel(variant_to_use, config)
    else:
        exp_grid_search_parallel(variant, config)
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
