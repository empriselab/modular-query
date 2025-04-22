"""A modular policy."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import ActionModule, StateModule
from modular_query.query_strategies.base import QueryStrategy


class ModularPolicy:
    """A policy defined by a graph of modules and a querying strategy."""

    def __init__(
        self, module_graph: ModuleGraph, query_strategy: QueryStrategy
    ) -> None:
        self.module_graph = module_graph
        self.query_strategy = query_strategy
        assert isinstance(self.module_graph.root, StateModule)
        assert isinstance(self.module_graph.leaf, ActionModule)

    def get_action(self, state: Any) -> tuple[Any, float]:
        """Invoke the policy and return action and total querying cost."""
        # Set the state in the state module.
        if not isinstance(self.module_graph.root, StateModule):
            raise RuntimeError("Root module must be a StateModule.")
        self.module_graph.root.set_state(state)

        # Compute initial values for all modules first.
        computed_values, computed_confidences, _ = self.module_graph.compute_values(
            expert_query_module_names=set()
        )

        # Compute the expert query modules using the querying strategy.
        expert_query_module_name = self.query_strategy.get_expert_query_module(
            module_graph=self.module_graph,
            computed_values=computed_values,
            computed_confidences=computed_confidences,
        )

        # Recompute values with the chosen expert queries.
        computed_values, _, total_query_cost = self.module_graph.compute_values(
            expert_query_module_names=(
                set([expert_query_module_name]) if expert_query_module_name else set()
            )
        )

        # Get the action from the leaf module.
        if not isinstance(self.module_graph.leaf, ActionModule):
            raise RuntimeError("Leaf module must be an ActionModule.")

        action_value = computed_values[self.module_graph.leaf]

        return action_value, total_query_cost
