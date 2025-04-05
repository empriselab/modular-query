"""A query strategy that considers all possible queries and chooses the
best."""

from itertools import combinations
from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


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
                query_cost = self._get_query_set_cost(
                    query_set, module_graph, computed_confidences
                )
                # Check if this query set is better than the best found so far.
                if query_cost < best_query_cost:
                    best_query_cost = query_cost
                    best_query_set = query_set
        return best_query_set

    def _get_query_set_cost(
        self,
        query_set: set[str],
        module_graph: ModuleGraph,
        computed_confidences: dict[Module, float],
    ) -> float:
        """Calculate the total cost of querying the modules in the given query
        set."""

        # Compute query cost.
        module_name_to_module = {m.get_name(): m for m in module_graph.get_modules()}
        total_query_cost = 0.0
        for module_name in query_set:
            module = module_name_to_module[module_name]
            total_query_cost += module.get_expert_query_cost()

        # Compute probability of being correct, assuming that a module confidence
        # is exactly the probability that it is correct.
        probability_of_correct_answer = 1.0
        nonquery_set = set(module_name_to_module) - query_set
        for module_name in nonquery_set:
            module = module_name_to_module[module_name]
            probability_of_correct_answer *= computed_confidences[module]

        # Compute total cost.
        combined_cost = total_query_cost + (
            self.correct_answer_cost * probability_of_correct_answer
            + self.incorrect_answer_cost * (1 - probability_of_correct_answer)
        )

        return combined_cost
