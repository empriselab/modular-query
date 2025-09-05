"""A query strategy that considers all possible queries and chooses the
best."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.module_utils import compare_query_set_expected_costs
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
    ) -> tuple[str | None, dict[str, float] | None]:
        # filter out state modules, and modules that have already been queried.
        all_modules = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
            and module.get_name() not in self.queried_modules
        ]
        best_query_module = None
        # Force to ask at least one module.
        # Will use the numerically-stable version of the expected cost.
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

        return best_query_module, {}
