"""A query strategy that uses a topological order to select the next module to
query.

This is a simple query strategy that just queries the next module in the
topological order.
"""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy


class TopoQueryStrategy(QueryStrategy):
    """A query strategy that uses a topological order to select the next module
    to query."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        """Get the next module to query."""
        # filter out state modules, and modules that have already been queried.
        all_modules = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
            and module.get_name() not in self.queried_modules
        ]
        if len(all_modules) == 0:
            return None, {}, {}
        return all_modules[0], {}, {}
