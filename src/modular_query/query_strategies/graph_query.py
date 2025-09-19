"""A query strategy that uses a graph-based approach to select the most
informative queries to ask."""

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
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
        and_modules: set[str] | None = None,
        or_modules: set[str] | None = None,
        workload_eps: float = 0.1,
    ) -> None:
        super().__init__(
            correct_answer_cost, incorrect_answer_cost, and_modules, or_modules
        )
        self.workload_eps = workload_eps

    def create_query_graph(
        self,
        module_graph: ModuleGraph,
        computed_confidences: dict[Module, float],
        force_query_for_module: str | None = None,
    ) -> nx.MultiDiGraph:
        """Create a query graph for the given module graph, excluding modules
        that have already been queried when making query edges.

        If force_query_for_module is provided, then the graph will be
        created with only a query edge for the module at the given
        index.
        """
        # Create failure recovery graph.
        graph: nx.MultiDiGraph = nx.MultiDiGraph()

        ## Add nodes. s_init, [module] success, [module] failure
        ## for each module.
        graph.add_node("s_init")
        for module in self.get_all_queryable_modules(module_graph):
            graph.add_node(f"s_{module},success")
            graph.add_node(f"s_{module},failure")

        ## Add edges
        modules_list = [module.get_name() for module in module_graph.topo_order]
        # Remove the first module, which we will always assume to be the 'state' module.
        modules_list = modules_list[1:]
        computed_confidences_str = {
            module.get_name(): confidence
            for (module, confidence) in computed_confidences.items()
        }

        ## Add autonomous edges.
        ## (NOTE: no self-loop currently for the last module.)
        ## Autonomous edges between consecutive levels.
        if force_query_for_module != modules_list[0]:
            auto_cost = 1 - computed_confidences_str[modules_list[0]]
            graph.add_edge(
                "s_init", f"s_{modules_list[0]},success", key="a_auto", cost=auto_cost
            )
            graph.add_edge(
                "s_init", f"s_{modules_list[0]},failure", key="a_auto", cost=auto_cost
            )
        for module_start, module_end in pairwise(modules_list):
            if force_query_for_module != module_end:
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

        ## Add query edges (only for modules that have not been queried yet).
        query_costs_str = {
            module.get_name(): module.get_expert_query_cost()
            for module in module_graph.get_modules()
            if module.get_name() in self.get_all_queryable_modules(module_graph)
        }
        ### Query edges between consecutive levels.
        query_cost = query_costs_str[modules_list[0]] * self.workload_eps
        graph.add_edge(
            "s_init", f"s_{modules_list[0]},success", key="a_query", cost=query_cost
        )
        for module_start, module_end in pairwise(modules_list):
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
        return graph

    def run_a_star(
        self, graph: nx.MultiDiGraph, source: str, final_module: str
    ) -> tuple[dict[str, bool], float]:
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
            return {}, float("inf")

        # Step 2: Get the path from the root to the leaf.
        # Determine whether we went down query path or not.
        path = {}
        path_cost = 0
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
            path_cost += graph[current_node][next_node][min_cost_edge]["cost"]

        # Step 3: Return the path.
        return path, path_cost

    def visualize_query_graph(self, graph: nx.MultiDiGraph, outfile: Path) -> None:
        """Visualize the query graph."""
        nx.draw(graph, with_labels=True)
        plt.savefig(outfile)
        plt.close()

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        ## The key to constraining this method to query once is - we will
        ## create N - |Q| copies of the graph (where N is the number of modules, and
        ## |Q| is the number of modules that have been queried so far),
        ## where copy i will not have autonomous edges for module i (i.e. forcing
        ## the query strategy to query for module i).
        ##
        ## We'll just run A* in each of the N copies of the graph, and return the
        ## best path from each of the N copies.

        # Step 1: Create N - |Q| copies of the graph.
        modules_list = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
        ]
        graphs: dict[int, nx.MultiDiGraph] = {}
        for i in range(len(modules_list)):
            if modules_list[i] not in self.queried_modules:
                graph = self.create_query_graph(
                    module_graph,
                    computed_confidences,
                    force_query_for_module=modules_list[i],
                )
                graphs[i] = graph

        # Step 2: Run A* in the graphs to return the best path.
        # Edge case: if modules_list is empty, then we return None.
        if len(modules_list) == 0:
            return None, {}, {"path_cost": 0.0}
        final_module = modules_list[-1]
        source = "s_init"

        best_path = None
        best_path_cost = float("inf")
        for i, graph in graphs.items():
            path, path_cost = self.run_a_star(graph, source, final_module)
            if path_cost < best_path_cost:
                best_path = path
                best_path_cost = path_cost

        if best_path is None:
            return None, {}, {"path_cost": 0.0}
        for module, is_query in best_path.items():
            if is_query:
                return module, {}, {"path_cost": best_path_cost}
        return None, {}, {"path_cost": 0.0}
