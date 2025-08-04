"""A modular policy."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import ActionModule, Module, StateModule
from modular_query.query_strategies.base import QueryStrategy
from modular_query.utils import print_and_log

## Includes 'sticky query' behavior:
## where if we query for module Mi at time t,
## we continue to use the expert for Mi at future t
## without incurring the query cost for module Mi.
##
## This assumes that if we invoke the expert once,
## we always have access to their 'oracle function' for free
## (i.e. even if the inputs to the module change,
## we can costlessly get the expert output)
## That is, once we query a module, we 'repair' it permanently.


class ModularPolicy:
    """A policy defined by a graph of modules and a querying strategy."""

    def __init__(
        self,
        module_graph: ModuleGraph,
        query_strategy: QueryStrategy,
        verbose: bool = False,
    ) -> None:
        self.module_graph = module_graph
        self.query_strategy = query_strategy
        assert isinstance(self.module_graph.root, StateModule)
        assert isinstance(self.module_graph.leaf, ActionModule)
        self.verbose = verbose
        # Track the set of modules that have been queried.
        self.queried_modules: set[str] = set()

    def forward_pass_only(
        self, state: Any
    ) -> tuple[Any, dict[Module, float], dict[Module, float]]:
        """Compute the values for all modules in the graph (without any
        queries) for a given state.

        Returns action value, computed values, and computed confidences.
        """
        # Set the state in the state module.
        if not isinstance(self.module_graph.root, StateModule):
            raise RuntimeError("Root module must be a StateModule.")
        self.module_graph.root.set_state(state)

        # Compute initial values for all modules first.
        if self.verbose:
            print_and_log("Computing initial values for all modules...")
        computed_values, computed_confidences, _ = self.module_graph.compute_values(
            expert_query_module_names=set()
        )
        return (
            computed_values[self.module_graph.leaf],
            computed_values,
            computed_confidences,
        )

    def get_action(
        self, state: Any
    ) -> tuple[Any, float, bool, dict[Module, float], dict[str, float] | None]:
        """Invoke the policy and return action and total querying cost and
        whether we queried for a module, as well as the post-query
        confidences."""
        _, computed_values, computed_confidences = self.forward_pass_only(state)

        # Compute the expert query modules using the querying strategy.
        expert_query_module_name, timing_info = (
            self.query_strategy.get_expert_query_module(
                module_graph=self.module_graph,
                computed_values=computed_values,
                computed_confidences=computed_confidences,
            )
        )

        # Determine whether we queried for a module.
        queried = expert_query_module_name is not None

        # Recompute values with the chosen expert queries.
        if self.verbose:
            print_and_log("Recomputing values with expert queries...")
            print_and_log(
                f"Expert query module:"
                f"{expert_query_module_name if expert_query_module_name else 'None'}"
            )
        # Get the set of modules to query.
        expert_query_module_names = (
            set([expert_query_module_name]) if expert_query_module_name else set()
        )
        # Add sticky queries when computing the graph values.
        expert_query_module_names.update(self.queried_modules)
        computed_values, post_query_confidences, total_query_cost = (
            self.module_graph.compute_values(
                expert_query_module_names=expert_query_module_names
            )
        )
        # But, subtract the query cost for the sticky queries
        # when computing the total query cost.
        total_query_cost_adjusted = total_query_cost - sum(
            (
                module.get_expert_query_cost()
                for module in self.module_graph.topo_order
                if module.get_name() in self.queried_modules
            )
        )

        # Get the action from the leaf module.
        if not isinstance(self.module_graph.leaf, ActionModule):
            raise RuntimeError("Leaf module must be an ActionModule.")

        action_value = computed_values[self.module_graph.leaf]

        if queried:
            assert (
                total_query_cost > 0
            ), "Raw total query cost should be positive if we query."
            # Add the queried module to the set of queried modules.
            assert expert_query_module_name is not None  # type narrowing for mypy
            self.queried_modules.add(expert_query_module_name)

        return (
            action_value,
            total_query_cost_adjusted,
            queried,
            post_query_confidences,
            timing_info,
        )
