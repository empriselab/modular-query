"""A hybrid graph query strategy that takes into account the dependency
struture of the graph.

In particular:
- All modules that are 'AND' gates are used to populate a BinaryTreeQueryStrategy.
- All modules that are 'OR' gates are used to populate a GraphQueryStrategy.

The core querying algorithm will operate as follows:
(remember, the constraint is that we need to query for at least one module,
as long as there are still modules that we haven't queried for.)

1. We consider querying for a module using the BinaryTreeQueryStrategy,
and then assume we are autonomous for all of the modules in GraphQueryStrategy -
we then compute the total cost of this query configuration.
2. We consider querying for a module using the GraphQueryStrategy
and then assume we are autonomous for all of the modules in BinaryTreeQueryStrategy -
we then compute the total cost of this query configuration.
3. We compare the total costs of the two arrangements,
and select the one with the lower cost.
"""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy
from modular_query.query_strategies.binary_tree_query import BinaryTreeQueryStrategy
from modular_query.query_strategies.graph_query import GraphQueryStrategy
from modular_query.utils import product_of_confidences, sum_of_uncertainties


class HybridGraphQueryStrategy(QueryStrategy):
    """A hybrid graph query strategy that takes into account the dependency
    struture of the graph."""

    def __init__(
        self,
        correct_answer_cost: float,
        incorrect_answer_cost: float,
        and_modules: set[str] | None = None,
        or_modules: set[str] | None = None,
    ):
        super().__init__(
            correct_answer_cost, incorrect_answer_cost, and_modules, or_modules
        )
        # Store the binary tree query strategy.
        self.binary_tree_query_strategy = BinaryTreeQueryStrategy(
            correct_answer_cost, incorrect_answer_cost, and_modules, or_modules
        )
        # Store the graph query strategy.
        self.graph_query_strategy = GraphQueryStrategy(
            correct_answer_cost, incorrect_answer_cost, and_modules, or_modules
        )

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        # Configuration 1: Query for a module using the BinaryTreeQueryStrategy,
        # and then assume we are autonomous
        # for all of the modules in GraphQueryStrategy.
        # Configuration 2: Query for a module using the GraphQueryStrategy,
        # and then assume we are autonomous
        # for all of the modules in BinaryTreeQueryStrategy.

        ## First, partition the original ModuleGraph into two,
        ## one for the AND gates,
        ## and one for the OR gates.
        full_module_to_parents = module_graph.module_to_parents
        and_module_to_parents = {
            module: parents
            for module, parents in full_module_to_parents.items()
            if module.get_name() in self.and_modules
        }
        or_module_to_parents = {
            module: parents
            for module, parents in full_module_to_parents.items()
            if module.get_name() in self.or_modules
        }
        # filter out invalid parents
        for module in list(and_module_to_parents.keys()):
            and_module_to_parents[module] = [
                parent
                for parent in and_module_to_parents[module]
                if parent in and_module_to_parents
            ]
        for module in list(or_module_to_parents.keys()):
            or_module_to_parents[module] = [
                parent
                for parent in or_module_to_parents[module]
                if parent in or_module_to_parents
            ]
        and_graph = ModuleGraph(and_module_to_parents, root_leaf_check=False)
        or_graph = ModuleGraph(or_module_to_parents, root_leaf_check=False)

        # Configuration 1.
        # Find the optimal querying strategy for the 'AND' graph.
        AND_module, and_timing_info, and_solution_info = (
            self.binary_tree_query_strategy.get_expert_query_module(
                and_graph, computed_values, computed_confidences
            )
        )
        # Compute the autonomous cost for the 'OR' gates
        # (use the sum-of-uncertainties proxy cost)
        restricted_confidences = {
            module: conf
            for module, conf in computed_confidences.items()
            if module.get_name() in self.or_modules
        }
        autonomous_OR_cost = sum_of_uncertainties(restricted_confidences)
        configuration_1_cost = and_solution_info["path_cost"] + autonomous_OR_cost

        # Configuration 2.
        restricted_confidences = {
            module: conf
            for module, conf in computed_confidences.items()
            if module.get_name() in self.and_modules
        }
        autonomous_AND_cost = 1 - product_of_confidences(restricted_confidences)
        OR_module, or_timing_info, or_solution_info = (
            self.graph_query_strategy.get_expert_query_module(
                or_graph, computed_values, restricted_confidences
            )
        )
        configuration_2_cost = or_solution_info["path_cost"] + autonomous_AND_cost

        if AND_module is None and OR_module is None:
            return None, {}, {}
        if AND_module is None:
            # If only the OR module is available, we should query for it.
            return OR_module, or_timing_info, {}
        if OR_module is None:
            # If only the AND module is available, we should query for it.
            return AND_module, and_timing_info, {}
        if configuration_1_cost < configuration_2_cost:
            return AND_module, and_timing_info, {}
        return OR_module, or_timing_info, {}
