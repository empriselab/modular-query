"""A modular policy."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import ActionModule, Module, StateModule
from modular_query.query_strategies.base import QueryStrategy
from modular_query.utils import print_and_log


class ModularPolicy:
    """A policy defined by a graph of modules and a querying strategy."""

    def __init__(
        self,
        module_graph: ModuleGraph,
        query_strategy: QueryStrategy,
        verbose: bool = False,
        variant: str = "balanced",
    ) -> None:
        self.module_graph = module_graph
        self.query_strategy = query_strategy
        assert isinstance(self.module_graph.root, StateModule)
        assert isinstance(self.module_graph.leaf, ActionModule)
        self.verbose = verbose
        self.variant = variant
        # Keep a cache of the expert values for each queried module.
        self.expert_values_cache: dict[str, Any] = {}

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
            expert_query_module_names=set(),
            expert_values_cache=self.expert_values_cache,
        )
        return (
            computed_values[self.module_graph.leaf],
            computed_values,
            computed_confidences,
        )

    def get_action(self, state: Any) -> tuple[
        Any,
        float,
        bool,
        Module | None,
        dict[Module, float],
        dict[Module, float],
        dict[str, float] | None,
    ]:
        """Invoke the policy and return action and total querying cost and
        whether we queried for a module, as well as the post-query
        confidences."""

        # Passing through with the expert cache 
        # (where we know that expert cached values will have a confidence of p_expert)
        self.module_graph.set_state(state)
        computed_values, computed_confidences, total_query_cost = (
            self.module_graph.compute_values(
                expert_query_module_names=set(),
                expert_values_cache=self.expert_values_cache,
            )
        )

        # Compute the expert query modules using the querying strategy.
        expert_query_module_name, timing_info, _ = (
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
        
        # Balanced-2 variant intervention: check termination condition before making query
        if (self.variant == "balanced-2" and 
            expert_query_module_name is not None and 
            expert_query_module_names):
            # Get the module and its current confidence
            queried_module = self.module_graph.get_module_by_name(expert_query_module_name)
            current_confidence = computed_confidences[queried_module]
            query_cost = queried_module.get_expert_query_cost()
            
            # Check if confidence gain is less than query cost
            confidence_gain = self.module_graph.expert_query_confidence - current_confidence
            if confidence_gain < query_cost:
                # Don't query - set expert_query_module_names to empty set
                # Also, set queried to False.
                expert_query_module_names = set()
                queried = False
                if self.verbose:
                    print_and_log(f"Balanced-2: Not querying {expert_query_module_name} "
                                f"(confidence gain {confidence_gain:.3f} < cost {query_cost:.3f})")
        
        computed_values, post_query_confidences, total_query_cost = (
            self.module_graph.compute_values(
                expert_query_module_names=expert_query_module_names,
                expert_values_cache=self.expert_values_cache,
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
            assert (
                expert_query_module_name is not None
            ), "Expert query module name should not be None if we query."
            # Post querying logic (querying state and cache updates)
            # - identify modules that are downstream of the queried module
            downstream_modules = self.module_graph.get_downstream_modules(
                expert_query_module_name
            )
            # - identify the intersection between the downstream set
            # and the strategy's current queried_modules set.
            downstream_modules_to_remove = downstream_modules.intersection(
                self.query_strategy.queried_modules
            )
            # - add the queried module to the strategy's queried_modules set
            #  [triggers internal strategy changes]
            self.query_strategy.add_queried_module(expert_query_module_name)
            # - remove any downstream modules from the strategy's queried_modules set
            #  [triggers internal strategy changes]
            self.query_strategy.remove_queried_modules(
                self.module_graph, downstream_modules_to_remove
            )

            # Cache the expert value for the queried module.
            expert_query_module = self.module_graph.get_module_str_to_module()[
                expert_query_module_name
            ]
            self.expert_values_cache[expert_query_module_name] = computed_values[
                expert_query_module
            ]
            # Reset the cache for downstream modules in the expert_values_cache.
            for module in downstream_modules_to_remove:
                self.expert_values_cache[module] = None

        return (
            action_value,
            total_query_cost,
            queried,
            self.module_graph.get_module_by_name(expert_query_module_name),
            computed_confidences,
            post_query_confidences,
            timing_info,
        )
