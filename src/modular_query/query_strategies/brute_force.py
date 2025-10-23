"""A query strategy that considers all possible queries and chooses the
best."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.module_utils import (
    compare_query_set_expected_costs,
    get_query_set_expected_cost,
)
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy


class BruteForceQueryStrategy(QueryStrategy):
    """A query strategy that considers all possible queries and chooses the
    best."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        # filter out state modules, and modules that have already been queried.
        all_modules = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
            and module.get_name() not in self.queried_modules
        ]
        best_query_module = None
        best_query_cost = float("inf")

        # Only use numerically-stable version if there are no 'OR' modules
        # (since it is only applicable to AND modules)
        use_numerically_stable_version = not bool(self.or_modules)

        if not use_numerically_stable_version:
            # Default version
            # (does not use numerically-stable version of the expected cost).
            for query_module in all_modules:
                query_set = set([query_module]) if query_module else set()
                query_cost = get_query_set_expected_cost(
                    query_set,
                    module_graph,
                    computed_confidences,
                    self.and_modules,
                    self.or_modules,
                )
                if query_cost < best_query_cost:
                    best_query_cost = query_cost
                    best_query_module = query_module
        else:
            # Will use the numerically-stable version of the expected cost.
            # Only usable if there are no 'OR' modules.
            for query_module in all_modules:
                query_set = set([query_module]) if query_module else set()
                if best_query_module:
                    comparison = compare_query_set_expected_costs(
                        query_set,
                        set([best_query_module]),
                        module_graph,
                        computed_confidences,
                    )
                    # Check if this query set is better than the best found so far.
                    if comparison == 1:
                        best_query_module = query_module
                else:
                    best_query_module = query_module

        return best_query_module, {}, {}
