"""A query strategy that uses a graph-based approach to select the most
informative queries to ask."""

import itertools
from typing import Any, Iterable

import networkx as nx

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


def pairwise(iterable: Iterable) -> zip:
    """Yields pairs of consecutive items from an iterable.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class BinaryTreeQueryStrategy(QueryStrategy):
    """A query strategy that uses a graph-based approach (A*) in a binary tree
    to select the most informative queries to ask."""

    def __init__(
        self,
        correct_answer_cost: float,
        incorrect_answer_cost: float,
    ) -> None:
        super().__init__(correct_answer_cost, incorrect_answer_cost)
        # State variable: Set of modules that we have queried for.
        # We add to this set when we query a module.
        self.queried_modules: dict[str, bool] = {}

    def reset(self) -> None:
        """Reset the state variable."""
        self.queried_modules = {}

    def create_query_graph(
        self,
        module_graph: ModuleGraph,
        computed_confidences: dict[Module, float],
    ) -> nx.DiGraph:
        """Create a query graph for the given module graph, excluding modules
        that have already been queried when making query edges."""

        # Create failure recovery graph.
        graph: nx.DiGraph = nx.DiGraph()

        ## Add nodes. s_init,
        ## for each module in the module graph, create a nodes that track
        ## the full history of queries for previous modules.

        modules_list = sorted(list(self.get_all_queryable_modules(module_graph)))

        graph.add_node("s_init")
        for i, module in enumerate(modules_list):
            for j in range(
                2 ** (i + 1)
            ):  # j is an integer whose binary encoding stores the query history.
                # 0s correspond to querying, 1s to not querying.
                graph.add_node(f"s_{module},{j}")
        graph.add_node("s_final")

        ## Add edges
        computed_confidences_str = {
            module.get_name(): confidence
            for (module, confidence) in computed_confidences.items()
        }

        ### Add query edges.
        query_costs_str = {
            module.get_name(): module.get_expert_query_cost()
            for module in module_graph.get_modules()
            if module.get_name() in self.get_all_queryable_modules(module_graph)
        }
        ### Query edges between consecutive levels.
        query_cost = query_costs_str[modules_list[0]]
        # First level.
        if modules_list[0] not in self.queried_modules:
            graph.add_edge(
                "s_init",
                f"s_{modules_list[0]},0",
                key="a_query",
                cost=query_cost,
            )
        for module_start, module_end in pairwise(modules_list):
            if module_end not in self.queried_modules:
                query_cost = query_costs_str[module_end]
                # Query edges start from all s_{module_start},j nodes,
                # and end at s_{module_end},k nodes, where k is even.
                # (even k corresponds to querying the module_end module,)
                for j in range(2 ** (modules_list.index(module_start) + 1)):
                    for k in range(0, 2 ** (modules_list.index(module_end) + 1), 2):
                        # Add query edges from all s_{module_start},j nodes
                        # to all s_{module_end},k nodes.
                        graph.add_edge(
                            f"s_{module_start},{j}",
                            f"s_{module_end},{k}",
                            key="a_query",
                            cost=query_cost,
                        )

        ### Add autonomous edges.
        ## Autonomous edges between consecutive levels.
        auto_cost = 1 - computed_confidences_str[modules_list[0]]
        graph.add_edge("s_init", f"s_{modules_list[0]},1", key="a_auto", cost=auto_cost)
        for module_start, module_end in pairwise(modules_list):
            auto_cost = 1 - computed_confidences_str[module_end]
            # Query edges start from all s_{module_start},j nodes,
            # and end at s_{module_end},k nodes, where k is odd.
            # (odd k corresponds to not querying the module_end module,)

            # The cost itself is a bit tricky to compute.
            # If k = 1 (which corresponds to query for all modules before module_end),
            # we should just put 1 - confidence of the module_end module.
            # But if k > 1, the cost is special (to create telescoping).
            # If we set A = product of confidences of all modules
            # before module_end that we did not query
            # (which we have to deduce from the binary encoding of j),
            # then the cost is - confidence of module_end * A + A.
            for j in range(2 ** (modules_list.index(module_start) + 1)):
                for k in range(1, 2 ** (modules_list.index(module_end) + 1), 2):
                    if k == 1:
                        # If k = 1, the cost is just 1 - confidence of module_end.
                        graph.add_edge(
                            f"s_{module_start},{j}",
                            f"s_{module_end},{k}",
                            key="a_auto",
                            cost=auto_cost,
                        )
                    else:
                        # If k > 1, we need to compute the cost
                        # based on the binary encoding of j.
                        # Get the product of confidences of all modules
                        # before module_end that we did not query.
                        A = 1.0
                        raw_binary_encoding = bin(j)[
                            2:
                        ]  # Get the binary encoding of j, without the '0b' prefix.
                        # 0 pad so that it has the same length
                        # as the number of modules before module_end.
                        modified_binary_encoding = raw_binary_encoding.zfill(
                            len(modules_list[: modules_list.index(module_end)])
                        )
                        for idx, bit in enumerate(modified_binary_encoding):
                            if bit == "1":  # '0' represents querying the module,
                                # '1' represents not querying.
                                A *= computed_confidences_str[modules_list[idx]]
                        # Compute the cost.
                        cost = -computed_confidences_str[module_end] * A + A
                        graph.add_edge(
                            f"s_{module_start},{j}",
                            f"s_{module_end},{k}",
                            key="a_auto",
                            cost=cost,
                        )
        ## Autonomous edges from the last level to the final node.
        # Cost is 0.
        for j in range(2 ** (len(modules_list))):
            graph.add_edge(
                f"s_{modules_list[-1]},{j}",
                "s_final",
                key="a_auto",
                cost=0,
            )
        return graph

    def run_a_star(self, graph: nx.DiGraph, source: str, dest: str) -> dict[str, bool]:
        """Run A* on the graph to find the best path."""

        # Step 1: Create + run A* search algorithm (returns list of nodes in the graph)
        a_star_path = nx.astar_path(graph, source, dest, weight="cost")

        # Step 2: Get the path from the root to the leaf.
        # Determine whether we went down query path or not.
        path = {}
        for _, next_node in pairwise(a_star_path):
            if next_node != "s_final":
                # Get the module name and the binary encoding of the query history.
                raw_module_name, binary_encoding = next_node.split(",")
                # Remove the 's_' prefix from the module name.
                module_name = raw_module_name[2:]  # Remove 's_' prefix.
                # Convert binary encoding to integer.
                binary_encoding = int(binary_encoding)
                # If the last bit is 0, we queried the module, otherwise we did not.
                if binary_encoding % 2 == 0:
                    path[module_name] = True
                else:
                    path[module_name] = False

        # Step 3: Return the path.
        return path

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> str | None:
        # Step 1: Create the query graph structure
        graph = self.create_query_graph(module_graph, computed_confidences)
        # Step 2: Run A* in the graph to return the best path
        path = self.run_a_star(graph, "s_init", "s_final")
        # Step 3: Get the query from the result
        # (just choose the first one in the sequence)
        for module, is_query in path.items():
            if is_query:
                # Update the state variable
                self.queried_modules[module] = True
                return module
        # If no module is found, return None
        return None
