"""A query strategy that considers all possible queries and chooses the
best."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.module_utils import get_query_set_expected_cost
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


class BruteForceQueryStrategy(QueryStrategy):
    """A query strategy that considers all possible queries and chooses the
    best."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None]:
        all_modules = self.get_all_queryable_modules(module_graph)
        # Filter out modules that have already been queried.
        all_modules = all_modules - self.queried_modules
        best_query_module = None
        best_query_cost = float("inf")
        # Force to ask at least one module.
        for query_module in list(all_modules):
            query_set = set([query_module]) if query_module else set()
            query_cost = get_query_set_expected_cost(
                query_set,
                module_graph,
                computed_confidences,
                self.correct_answer_cost,
                self.incorrect_answer_cost,
            )
            # Check if this query set is better than the best found so far.
            if query_cost < best_query_cost:
                best_query_cost = query_cost
                best_query_module = query_module
        return best_query_module, {}
