"""A query strategy that uses a graph-based approach to select the most
informative queries to ask."""

import itertools
from typing import Any

import networkx as nx

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


def pairwise(iterable: Any) -> zip:
    """Yields pairs of consecutive items from an iterable.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# NOTE: for now, i'll just implement a 1 time-step version of this. That is,
# create the graph structure, run A* once, and just get a query from the result.
# We'll set epsilon = 0.1 for demonstration purposes.


class GraphQueryStrategy(QueryStrategy):
    """A query strategy that uses a graph-based approach (A*) to select the
    most informative queries to ask.

    Uses an epsilon value of 0.1 by default.
    """

    def __init__(
        self,
        correct_answer_cost: float,
        incorrect_answer_cost: float,
        workload_eps: float = 0.1,
    ) -> None:
        super().__init__(correct_answer_cost, incorrect_answer_cost)
        self.workload_eps = workload_eps
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
    ) -> nx.MultiDiGraph:
        """Create a query graph for the given module graph, excluding modules
        that have already been queried when making query edges."""

        # Create failure recovery graph.
        graph: nx.MultiDiGraph = nx.MultiDiGraph()

        ## Add nodes. s_init, [module] success, [module] failure
        ## for each module.
        graph.add_node("s_init")
        for module in self.get_all_queryable_modules(module_graph):
            graph.add_node(f"s_{module},success")
            graph.add_node(f"s_{module},failure")

        ## Add edges
        modules_list = sorted(list(self.get_all_queryable_modules(module_graph)))
        computed_confidences_str = {
            module.get_name(): confidence
            for (module, confidence) in computed_confidences.items()
        }

        ## Add autonomous edges.
        ## (NOTE: no self-loop currently for the last module.)
        ## Autonomous edges between consecutive levels.
        auto_cost = 1 - computed_confidences_str[modules_list[0]]
        graph.add_edge(
            "s_init", f"s_{modules_list[0]},success", key="a_auto", cost=auto_cost
        )
        graph.add_edge(
            "s_init", f"s_{modules_list[0]},failure", key="a_auto", cost=auto_cost
        )
        for module_start, module_end in pairwise(modules_list):
            auto_cost = 1 - computed_confidences_str[module_end]
            # Add autonomous edges to the success and failure nodes.
            graph.add_edge(
                f"s_{module_start},success",
                f"s_{module_end},success",
                key="a_auto",
                cost=auto_cost,
            )
            graph.add_edge(
                f"s_{module_start},success",
                f"s_{module_end},failure",
                key="a_auto",
                cost=auto_cost,
            )
            graph.add_edge(
                f"s_{module_start},failure",
                f"s_{module_end},success",
                key="a_auto",
                cost=auto_cost,
            )
            graph.add_edge(
                f"s_{module_start},failure",
                f"s_{module_end},failure",
                key="a_auto",
                cost=auto_cost,
            )

        ## Add query edges.
        query_costs_str = {
            module.get_name(): module.get_expert_query_cost()
            for module in module_graph.get_modules()
            if module.get_name() in self.get_all_queryable_modules(module_graph)
        }
        ### Query edges between consecutive levels.
        query_cost = query_costs_str[modules_list[0]] * self.workload_eps
        if modules_list[0] not in self.queried_modules:
            graph.add_edge(
                "s_init", f"s_{modules_list[0]},success", key="a_query", cost=query_cost
            )
        for module_start, module_end in pairwise(modules_list):
            if module_end not in self.queried_modules:
                query_cost = query_costs_str[module_end] * self.workload_eps
                graph.add_edge(
                    f"s_{module_start},success",
                    f"s_{module_end},success",
                    key="a_query",
                    cost=query_cost,
                )
                graph.add_edge(
                    f"s_{module_start},failure",
                    f"s_{module_end},success",
                    key="a_query",
                    cost=query_cost,
                )
        ### Query edges from last failure node to each of the success nodes.
        num_back_edges = 0
        for module in modules_list:
            if module not in self.queried_modules:
                query_cost = query_costs_str[module] * self.workload_eps
                graph.add_edge(
                    f"s_{modules_list[-1]},failure",
                    f"s_{module},success",
                    key="a_query",
                    cost=query_cost,
                )
                num_back_edges += 1
        # print(f"Number of back edges: {num_back_edges}")
        return graph

    def run_a_star(
        self, graph: nx.MultiDiGraph, source: str, final_module: str
    ) -> dict[str, bool]:
        """Run A* on the graph to find the best path."""
        # Step 1: Create + run A* search algorithm (returns list of nodes in the graph)
        try:
            # A* search algorithm
            a_star_path = nx.astar_path(
                graph, source, f"s_{final_module},success", weight="cost"
            )
        except nx.NetworkXNoPath:
            # Return empty path
            # (ideally, should have been a dict where all modules map to False,
            # but this is easier to implement for now)
            # print("No path found in the graph.")
            return {}

        # Step 2: Get the path from the root to the leaf.
        # Determine whether we went down query path or not.
        path = {}
        for current_node, next_node in pairwise(a_star_path):
            # Get the module name from the current node.
            module_name = next_node.split(",")[0][2:]
            # Get all edges between the two nodes
            all_edges = graph[current_node][next_node]
            # Get the edge with the lowest cost.
            edge_key, _edge_attr = min(
                all_edges.items(),  # (<edge_key>, <attr_dict>) pairs
                key=lambda kv: kv[1]["cost"],  # kv[1] is the attr-dict
            )
            min_cost_edge = edge_key
            # Mark path[module_name] as True
            # if the edge is a query edge, False otherwise.
            if min_cost_edge == "a_query":
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
        modules_list = sorted(list(self.get_all_queryable_modules(module_graph)))
        final_module = modules_list[-1]
        # If we've already queried for one module, assume that we failed,
        # so the source node for A* has to change.
        source = (
            "s_init" if len(self.queried_modules) == 0 else f"s_{final_module},failure"
        )
        # print(f"Source node: {source}, Final module: {final_module}")
        path = self.run_a_star(graph, source, final_module)
        # print_and_log(f"Path: {path}")
        # Step 3: Get the query from the result
        # (just choose the first one in the sequence)
        for module, is_query in path.items():
            if is_query:
                # Update the state variable
                self.queried_modules[module] = True
                return module
        # If no module is found, return None
        return None
