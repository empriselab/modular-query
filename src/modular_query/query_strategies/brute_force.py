"""A query strategy that considers all possible queries and chooses the
best."""

from itertools import combinations
from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy
from modular_query.utils import get_query_set_expected_cost


class BruteForceQueryStrategy(QueryStrategy):
    """A query strategy that considers all possible queries and chooses the
    best."""

    def get_expert_query_modules(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> set[str]:
        all_modules = self.get_all_queryable_modules(module_graph)
        best_query_set = set()
        best_query_cost = float("inf")
        for r in range(len(all_modules) + 1):
            for query_combo in combinations(all_modules, r):
                query_set = set(query_combo)
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
                    best_query_set = query_set
        return best_query_set
