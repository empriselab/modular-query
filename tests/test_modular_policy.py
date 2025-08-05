"""Tests for modular_policy.py."""

from pathlib import Path
from typing import Any

import numpy as np

from modular_query.modular_policy import ModularPolicy
from modular_query.module_graph import ModuleGraph
from modular_query.module_utils import generate_random_and_gate_module_graph
from modular_query.modules import ActionModule, Module, StateModule
from modular_query.query_strategies.binary_tree_query import BinaryTreeQueryStrategy
from modular_query.query_strategies.brute_force import BruteForceQueryStrategy
from modular_query.query_strategies.graph_query import GraphQueryStrategy
from modular_query.query_strategies.mip import MIPQueryStrategy
from modular_query.query_strategies.never_query import NeverQueryStrategy


def test_modular_policy():
    """Tests for ModularPolicy()."""

    class _MiddleModule(Module):

        @classmethod
        def get_name(cls) -> str:
            return "middle"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"state": 1} in this test
            return inputs["state"] + 1, 0.5

        def get_expert_query_cost(self) -> float:
            return 100.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    class _ActionModule(ActionModule):

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"middle": 2} in this test
            return inputs["middle"] + 1, 0.5

        def get_expert_query_cost(self) -> float:
            return 1.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return inputs["middle"] + 2

    state_module = StateModule()
    middle_module = _MiddleModule()
    action_module = _ActionModule()

    module_to_parents = {
        state_module: [],
        middle_module: [state_module],
        action_module: [middle_module],
    }
    correct_answer_cost = 0.0
    incorrect_answer_cost = 10.0

    graph = ModuleGraph(module_to_parents)
    query_strategy = NeverQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )

    action, cost, _, _, _ = policy.get_action(state=1)
    assert action == 3, f"Expected action to be 3, got {action}"
    assert abs(cost - 0.0) < 1e-6, f"Expected total query cost to be 0.0, got {cost}"

    query_strategy = BruteForceQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )

    action, cost, _, _, _ = policy.get_action(state=1)
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    query_strategy = MIPQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )
    action, cost, _, _, _ = policy.get_action(state=1)
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    # Test for GraphQueryStrategy
    query_strategy = GraphQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )
    action, cost, _, _, _ = policy.get_action(state=1)

    # NOTE: holds for epsilon=0.1 (assumption for this test).
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    # If we decide to query again (even though in this case the first query leads to
    # success), we will not query for any module, but because of the sticky query
    # behavior, we'll use expert advice for both modules, leading to the action still
    # being 4.

    action, cost, _, _, _ = policy.get_action(state=1)
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 0.0) < 1e-6, f"Expected total query cost to be 0.0, got {cost}"

    # Test for BinaryTreeQueryStrategy
    query_strategy = BinaryTreeQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )
    # Visualize the planning graph with dummy confidences.
    dummy_confidences = {
        action_module: 0.5,
        middle_module: 0.5,
        state_module: 0.5,
    }
    query_strategy.visualize_planning_graph(
        query_strategy.create_query_graph(graph, dummy_confidences),
        Path("tests/test_planning_graph.png"),
    )

    # Because we force a single query, we should query for the action module.
    action, cost, queried, _, _ = policy.get_action(state=1)
    assert queried, f"Expected queried to be True, got {queried}"
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"


def test_graph_query_strategy():
    """Test for GraphQueryStrategy() with the random AND gate graph."""
    # Reproduces the current setup in run_experiment.
    # So we can understand why if there are 2 incorrect modules,
    # the graph query strategy
    # may not necessarily query for them in sequence as it should.
    module_graph = generate_random_and_gate_module_graph(
        num_modules=5,
        edge_probability=0.3,
        query_cost=0.08,
        rng=np.random.default_rng(42),
        num_incorrect_modules=2,
    )

    state = True

    # Get the correct expected output.
    all_queryable_module_names = {m.get_name() for m in module_graph.get_modules()}
    all_queryable_module_names.remove("state")
    assert isinstance(module_graph.root, StateModule)
    module_graph.root.set_state(state)
    computed_values, _, _ = module_graph.compute_values(
        expert_query_module_names=all_queryable_module_names
    )
    ground_truth_output = computed_values[module_graph.leaf]

    # Reset strategy's internal state.
    strategy = GraphQueryStrategy(correct_answer_cost=0.0, incorrect_answer_cost=1.0)
    strategy.reset()

    policy = ModularPolicy(
        module_graph=module_graph, query_strategy=strategy, verbose=True
    )

    # Temporal loop.
    # Initialize accumulators.
    timesteps_elapsed = 0
    time_horizon = 5
    correct = False
    while timesteps_elapsed < time_horizon and not correct:
        # Run the policy.
        action, _, _, _, _ = policy.get_action(state=state)

        correct = action == ground_truth_output

        # Increment timesteps elapsed.
        timesteps_elapsed += 1


def test_binary_tree_query_strategy():
    """Test for BinaryTreeQueryStrategy() with 2 middle modules, each with
    different confidences (0.5 and 0.25)."""

    class _MiddleModule1(Module):
        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            return inputs["state"] + 1, 0.25

        @classmethod
        def get_name(cls) -> str:
            return "middle1"

        def get_expert_query_cost(self) -> float:
            return 100.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    class _MiddleModule2(Module):
        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            return inputs["state"] + 1, 0.5

        @classmethod
        def get_name(cls) -> str:
            return "middle2"

        def get_expert_query_cost(self) -> float:
            return 100.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    class _ActionModule(ActionModule):
        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            return inputs["middle1"] + inputs["middle2"] + 1, 0.5

        @classmethod
        def get_name(cls) -> str:
            return "action"

        def get_expert_query_cost(self) -> float:
            return 1.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return inputs["middle1"] + inputs["middle2"] + 2

    state_module = StateModule()
    middle_module_1 = _MiddleModule1()
    middle_module_2 = _MiddleModule2()
    action_module = _ActionModule()

    module_to_parents = {
        state_module: [],
        middle_module_1: [state_module],
        middle_module_2: [state_module],
        action_module: [middle_module_1, middle_module_2],
    }

    graph = ModuleGraph(module_to_parents)
    correct_answer_cost = 0.0
    incorrect_answer_cost = 10.0
    query_strategy = BinaryTreeQueryStrategy(correct_answer_cost, incorrect_answer_cost)

    # Visualize the planning graph with dummy confidences.
    dummy_confidences = {
        action_module: 0.5,
        middle_module_1: 0.25,
        middle_module_2: 0.5,
        state_module: 0.5,
    }
    query_strategy.visualize_planning_graph(
        query_strategy.create_query_graph(graph, dummy_confidences),
        Path("tests/test_planning_graph_2.png"),
    )
