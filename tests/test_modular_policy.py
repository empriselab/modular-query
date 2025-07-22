"""Tests for modular_policy.py."""

from typing import Any

from modular_query.modular_policy import ModularPolicy
from modular_query.module_graph import ModuleGraph
from modular_query.modules import ActionModule, Module, StateModule
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

    action, cost, _, _ = policy.get_action(state=1)
    assert action == 3, f"Expected action to be 3, got {action}"
    assert abs(cost - 0.0) < 1e-6, f"Expected total query cost to be 0.0, got {cost}"

    query_strategy = BruteForceQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )

    action, cost, _, _ = policy.get_action(state=1)
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    query_strategy = MIPQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )
    action, cost, _, _ = policy.get_action(state=1)
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    # Test for GraphQueryStrategy
    query_strategy = GraphQueryStrategy(correct_answer_cost, incorrect_answer_cost)
    policy = ModularPolicy(
        module_graph=graph,
        query_strategy=query_strategy,
    )
    action, cost, _, _ = policy.get_action(state=1)

    # NOTE: holds for epsilon=0.1 (assumption for this test).
    assert action == 4, f"Expected action to be 4, got {action}"
    assert abs(cost - 1.0) < 1e-6, f"Expected total query cost to be 1.0, got {cost}"

    # If we decide to query again (even though in this case the first query leads to
    # success), we should query for the middle module, since the method forces
    # a query for subsequent timesteps.

    action, cost, _, _ = policy.get_action(state=1)
    assert action == 3, f"Expected action to be 3, got {action}"
    assert (
        abs(cost - 100.0) < 1e-6
    ), f"Expected total query cost to be 100.0, got {cost}"
