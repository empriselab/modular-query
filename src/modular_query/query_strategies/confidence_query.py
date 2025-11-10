"""A query strategy that queries for the module with the lowest confidence,
ignoring workload."""


from typing import Any
from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy


class ConfidenceQueryStrategy(QueryStrategy):
    """A query strategy that queries for the module with the lowest confidence,
    ignoring workload."""

    def get_expert_query_module(self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        # Filter out state modules, and modules that have already been queried.
        all_modules = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
            and module.get_name() not in self.queried_modules
        ]
        # Find the module with the lowest confidence.
        if len(all_modules) == 0:
            return None, {}, {}
        lowest_confidence_module = min(
            all_modules,
            key=lambda x: computed_confidences[module_graph.get_module_by_name(x)],
        )
        return lowest_confidence_module, {}, {}