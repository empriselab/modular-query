#!/usr/bin/env python3
"""Experiment to measure and compare performance of different querying
strategies."""

import abc
import argparse
import os
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    time_horizon: int = 5,
    incorrect_module_count: np.ndarray | None = None,
    variant: str = "balanced",
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Run experiments with different graph sizes and querying strategies.

    Querying costs: not used for the polynomial module graph,
    but used for the logic gate module graph.

    incorrect_module_count: array of number of failures,
    where each value is the number of trials with that many failures.
    sum(incorrect_module_count) must = num_trials.
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
    for strategy_name in strategies:
        results[strategy_name] = {
            "query_cost": {size: [] for size in graph_sizes},
            "task_cost": {size: [] for size in graph_sizes},
            "total_cost": {size: [] for size in graph_sizes},
            "proxy_obj_1": {size: [] for size in graph_sizes},
            "proxy_obj_2": {size: [] for size in graph_sizes},
            "execution_time": {size: [] for size in graph_sizes},
            "mean_queries": {size: [] for size in graph_sizes},
            "total_queries": {size: [] for size in graph_sizes},
            "total_correct": {size: [] for size in graph_sizes},
            "total_timesteps": {size: [] for size in graph_sizes},
            "total_executions": {size: [] for size in graph_sizes},
            "total_task_cost": {size: [] for size in graph_sizes},
        }

    # Create cumulative distribution of incorrect module counts.
    incorrect_module_counts_cumsum = np.cumsum(incorrect_module_count)

    # Run experiments for each graph size.
    for size in graph_sizes:
        print(f"Running experiments for graph size {size}")

        # Reset num incorrect modules for each graph size.
        num_incorrect_modules = 0

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
                expert_query_module_names=all_queryable_module_names
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
                if variant == "greedy":
                    # Forward pass only.
                    action, _, computed_confidences = policy.forward_pass_only(state)
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
                    # Measure execution time.
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

                    # Record execution time.
                    execution_time = time.perf_counter() - start_time

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
                    acc_execution_time += execution_time

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
                results[strategy_name]["mean_queries"][size].append(mean_queries)
                results[strategy_name]["total_queries"][size].append(acc_queried)
                results[strategy_name]["total_correct"][size].append(correct)
                results[strategy_name]["total_timesteps"][size].append(
                    timesteps_elapsed
                )
                results[strategy_name]["total_executions"][size].append(num_executions)
                results[strategy_name]["total_task_cost"][size].append(acc_task_cost)

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


def plot_results(
    results: dict[str, dict[str, dict[int, list[float]]]],
    graph_sizes: list[int],
    plot_name: str = "strategy_comparison.png",
) -> None:
    """Create plots showing the performance of different querying strategies.

    Includes two proxy objectives:
    - Proxy objective 1 (uses 1 - product of confidences)
    - Proxy objective 2 (uses sum of uncertainties)

    Args:
        results: dictionary with results from run_experiment
        graph_sizes: list of graph sizes that were tested
    """
    # Set up figure with six subplots
    num_rows = 4
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 12), sharex=True)

    metrics = [
        "query_cost",
        "task_cost",
        "total_cost",
        "proxy_obj_1",
        "proxy_obj_2",
        "execution_time",
        "mean_queries",
        "total_queries",
        "total_correct",
        "total_timesteps",
        "total_executions",
    ]
    titles = [
        "Query Cost",
        "Task Cost",
        "Total Cost",
        "Product-of-Confidences Total Cost",
        "Sum-of-Uncertainties Total Cost",
        "Execution Time (s)",
        "Mean Queries per Time Step",
        "Total Queries",
        "Total Correct",
        "Total Timesteps",
        "Total Executions",
    ]
    ylabels = [
        "Cost",
        "Cost",
        "Cost",
        "Cost",
        "Cost",
        "Time (s)",
        "Mean Queries",
        "Total Queries",
        "Total Correct",
        "Total Timesteps",
        "Total Executions",
    ]

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
        "Binary Tree Query": {
            "color": "cyan",
            "linestyle": "-",
            "marker": "v",
            "linewidth": 2,
        },
    }

    # Plot each metric
    lines = []
    labels = []
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // num_cols][i % num_cols]

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
        ax.set_ylabel(ylabels[i])
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


def exp_vary_cquery(variant: str) -> None:
    """Run the experiment with varying query cost."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]
    time_horizon = 5

    # 6/12: vary c_query now, but keep workload_eps=1.0
    workload_eps = 1.0
    c_query_list = [0.1]
    num_trials = 100

    for c_query in c_query_list:
        print_and_log(f"Running experiments with c_query = {c_query:.2f}")
        results = run_experiment(
            graph_sizes=graph_sizes,
            num_trials=num_trials,
            query_cost=c_query,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            workload_eps=workload_eps,
            time_horizon=time_horizon,
            incorrect_module_count=np.array([0, num_trials]),
            variant=variant,
        )

        # Plot the results
        plot_results(
            results,
            graph_sizes,
            plot_name=f"strategy_comparison_c_query_{c_query:.2f}.png",
        )

    print("Experiment with varying query cost complete!")


def exp_vary_num_failures(variant: str) -> None:
    """Run the experiment with varying number of failures; also saves results
    to a pickle file and plots the results."""
    graph_sizes = [10, 15, 18, 25, 50, 75, 100]
    num_failures_list = [0, 1, 2, 3]
    num_trials = 100
    time_horizon = 10

    for num_failures in num_failures_list:
        # Convert num_failures to a one-hot vector.
        incorrect_module_count = np.zeros(num_failures + 1)
        incorrect_module_count[-1] = num_trials
        # Don't include graph sizes that are too small for the number of failures.
        graph_sizes = [size for size in graph_sizes if size > num_failures]
        print_and_log(f"Running experiments with num_failures = {num_failures}")
        results = run_experiment(
            graph_sizes=graph_sizes,
            num_trials=num_trials,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            query_cost=0.08,
            incorrect_module_count=incorrect_module_count,
            workload_eps=1.0,
            time_horizon=time_horizon,
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
        plot_results(
            results,
            graph_sizes,
            plot_name=f"strategy_comparison_num_failures_{variant}_{num_failures}.png",
        )


def exp_mixed_failure_population(variant: str) -> None:
    """Run the experiment with a mixed failure population."""
    graph_sizes = [3, 5, 10, 15, 18, 25, 50, 75, 100]

    num_trials = 100

    # 0-failure, 1-failure, and mixed failure population.
    failure_populations = [np.array([100, 0]), np.array([0, 100]), np.array([50, 50])]
    for failure_population in failure_populations:
        print_and_log(
            f"Running experiments with failure population {failure_population}"
        )
        results = run_experiment(
            graph_sizes=graph_sizes,
            num_trials=num_trials,
            correct_answer_cost=0.0,
            incorrect_answer_cost=1.0,
            query_cost=0.08,
            incorrect_module_count=failure_population,
            variant=variant,
        )

        # Plot the results
        plot_results(
            results,
            graph_sizes,
            plot_name=f"strategy_comparison_mixed_failure_population"
            f"_{variant}_{failure_population}.png",
        )


def main(variant: str) -> None:
    """Run the experiment and generate plots."""
    # Run the experiment with varying query cost.
    # exp_vary_cquery(variant)

    # Run the experiment with varying number of failures.
    # exp_vary_num_failures(variant)

    # Run for all variants.
    if variant == "all-variants":
        for variant_to_use in ["balanced", "greedy", "conservative", "balanced-2"]:
            exp_vary_num_failures(variant_to_use)
    else:
        exp_vary_num_failures(variant)

    # Run the experiment with a mixed failure population.
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
