"""A query strategy that doesn't query anything."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


class NeverQueryStrategy(QueryStrategy):
    """A query strategy that doesn't query anything."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> str | None:
        return None
