"""A query strategy that always queries everything."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


class AlwaysQueryStrategy(QueryStrategy):
    """A query strategy that always queries everything."""

    def get_expert_query_modules(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> set[str]:
        return self.get_all_queryable_modules(module_graph)
