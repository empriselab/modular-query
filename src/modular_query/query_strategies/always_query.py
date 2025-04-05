"""A query strategy that always queries everything."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy


class AlwaysQueryStrategy(QueryStrategy):
    """A query strategy that always queries everything."""

    def get_expert_query_modules(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
    ) -> set[str]:
        to_query = set()
        for module in module_graph.get_modules():
            if isinstance(module, StateModule):
                continue
            to_query.add(module.get_name())
        return to_query
