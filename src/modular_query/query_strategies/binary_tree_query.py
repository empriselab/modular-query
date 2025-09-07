"""A query strategy that uses a graph-based approach to select the most
informative queries to ask."""

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy
from modular_query.utils import timer


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
        and_modules: set[str] | None = None,
        or_modules: set[str] | None = None,
    ) -> None:
        super().__init__(
            correct_answer_cost, incorrect_answer_cost, and_modules, or_modules
        )
        # Store the graph.
        self.graph: nx.DiGraph | None = None
        # Store the list of modules in the module graph.
        self.modules_list: list[str] = []

    def reset(self) -> None:
        """Reset the state variable (need to also reset the graph.)"""
        super().reset()
        self.graph = None
        self.modules_list = []

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
        self.modules_list = [
            module.get_name()
            for module in module_graph.topo_order
            if not isinstance(module, StateModule)
        ]

        ## Create mapping from module name to list of valid node IDs (in binary)
        module_to_node_ids: dict[str, list[str]] = {}

        graph.add_node("s_init")
        for _, module in enumerate(self.modules_list):
            # We index nodes as s_{module_name},j,
            # where j is an integer whose binary encoding stores the query history.
            # 0s correspond to querying, 1s to not querying.
            # Constraint is query history can only contain at most 1 query in total.

            # Construct valid j values for the module.
            module_to_node_ids[module] = []
            base_str = "1" * (
                self.modules_list.index(module) + 1
            )  # Base string for the module.
            node_binary_strs = [base_str]
            for index in range(len(base_str)):
                one_query_str = (
                    base_str[:index] + "0" + base_str[index + 1 :]
                )  # Replace the index with '0' to indicate querying.
                node_binary_strs.append(one_query_str)
            module_to_node_ids[module] = node_binary_strs

            for binary_str in node_binary_strs:
                # Convert binary string to integer.
                j = int(binary_str, 2)
                # Add the node to the graph.
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
        query_cost = query_costs_str[self.modules_list[0]]
        # First level.
        if self.modules_list[0] not in self.queried_modules:
            graph.add_edge(
                "s_init",
                f"s_{self.modules_list[0]},0",
                key="a_query",
                cost=query_cost,
            )
        for module_start, module_end in pairwise(self.modules_list):
            if module_end not in self.queried_modules:
                query_cost = query_costs_str[module_end]
                # Only the special node j = "111...1" (all 1s),
                # corresponding to not querying any previous modules,
                # can have query edges to the next module.
                # And it will go to k = "111...10" (all 1s, last bit is 0),
                # which corresponds to querying the module_end module.
                j_binary_str = "1" * (self.modules_list.index(module_start) + 1)
                k_binary_str = j_binary_str + "0"
                j = int(j_binary_str, 2)
                k = int(k_binary_str, 2)
                # Add the query edge from s_{module_start},j to s_{module_end},k.
                graph.add_edge(
                    f"s_{module_start},{j}",
                    f"s_{module_end},{k}",
                    key="a_query",
                    cost=query_cost,
                )

        ### Add autonomous edges.
        ## Autonomous edges between consecutive levels.
        auto_cost = 1 - computed_confidences_str[self.modules_list[0]]
        graph.add_edge(
            "s_init", f"s_{self.modules_list[0]},1", key="a_auto", cost=auto_cost
        )
        for module_start, module_end in pairwise(self.modules_list):
            # Autonomous edges start from all s_{module_start},j nodes,
            # and end at s_{module_end},k nodes, where k = concat(j, "1")
            # (odd k corresponds to not querying the module_end module,)

            # Cost function computation.
            # - If k = 1 (query for all modules before module_end),
            #   cost = 1 - confidence(module_end)
            # - If k > 1, the cost is special (to create telescoping).
            #   Let A = product(confidences of all prior modules that we did not query)
            #   (which we have to deduce from the binary encoding of j),
            #   cost = - confidence(module_end) * A + A.
            for j_binary_str in module_to_node_ids[module_start]:
                j = int(j_binary_str, 2)
                k_binary_str = j_binary_str + "1"
                k = int(k_binary_str, 2)
                if k == 1:
                    cost = 1 - computed_confidences_str[module_end]
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
                        len(self.modules_list[: self.modules_list.index(module_end)])
                    )
                    for idx, bit in enumerate(modified_binary_encoding):
                        if bit == "1":  # '0' represents querying the module,
                            # '1' represents not querying.
                            A *= computed_confidences_str[self.modules_list[idx]]
                    # Compute the cost.
                    cost = -computed_confidences_str[module_end] * A + A
                # Add edge
                graph.add_edge(
                    f"s_{module_start},{j}",
                    f"s_{module_end},{k}",
                    key="a_auto",
                    cost=cost,
                )

        ## Autonomous edges from the last level to the final node.
        # Cost is 0.
        for j_binary_str in module_to_node_ids[self.modules_list[-1]]:
            j = int(j_binary_str, 2)
            graph.add_edge(
                f"s_{self.modules_list[-1]},{j}",
                "s_final",
                key="a_auto",
                cost=0,
            )

        ## Forcing exactly one query.
        ## Remove the edge corresponding to the all fully-autonomous path.
        final_node = "1" * len(self.modules_list)
        final_node_int = int(final_node, 2)
        graph.remove_edge(
            f"s_{self.modules_list[-1]},{final_node_int}",
            "s_final",
        )

        return graph

    def visualize_planning_graph(self, graph: nx.MultiDiGraph, outfile: Path) -> None:
        """Visualize the graph."""
        # Increase the size of the plot.
        plt.figure(figsize=(20, 6))

        # Group nodes by prefix
        prefix_to_nodes = defaultdict(list)
        for node in graph.nodes:
            if node != "s_init" or node != "s_final":
                prefix = node.split(",")[0]  # e.g., 's_action'
                prefix_to_nodes[prefix].append(node)

        # Assign positions: same x for same prefix, y spread out
        pos = {}
        xscale = 1.75
        for i, (prefix, nodes) in enumerate(prefix_to_nodes.items()):
            for j, node in enumerate(sorted(nodes)):
                xindex = i * xscale
                pos[node] = (xindex, -j)  # x = group index, y = -order in group

        # Put in s_init and s_final at the top and bottom.
        pos["s_init"] = (0, 0)
        pos["s_final"] = (xscale * (len(prefix_to_nodes) - 1), 0)

        # Draw the graph.
        nx.draw(graph, pos, with_labels=True)

        # Put in edge labels.
        edge_labels = {
            (u, v): f"{d['key']} ({d['cost']})" for u, v, d in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        plt.savefig(outfile)
        plt.close()

    def run_a_star(
        self, graph: nx.DiGraph, source: str, dest: str
    ) -> tuple[dict[str, bool], float]:
        """Run A* on the graph to find the best path."""

        # Step 1: Create + run A* search algorithm (returns list of nodes in the graph)
        a_star_path = nx.astar_path(graph, source, dest, weight="cost")

        # Step 2: Get the path from the root to the leaf.
        # Determine whether we went down query path or not.
        path = {}
        path_cost = 0.0
        for current_node, next_node in pairwise(a_star_path):
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
                path_cost += graph.get_edge_data(current_node, next_node)["cost"]

        # Step 3: Return the path.
        return path, path_cost

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float], dict[str, Any]]:
        """
        Returns:
            - The module to query,
            - The timing information,
            - Solution information (for now, just path cost.)
        """
        # Step 1: Create the query graph structure
        # Only do this if we have not created the graph yet.
        if self.graph is None:
            with timer(
                "BinaryTreeQueryStrategy: Create query graph",
                verbose=False,
            ) as result:  # type: dict[str, float]
                self.graph = self.create_query_graph(module_graph, computed_confidences)
            t_create_graph = result["time"]
        else:
            t_create_graph = 0.0

        assert self.graph is not None, "Graph should not be None."

        # Step 2: Run A* in the graph to return the best path.
        with timer("BinaryTreeQueryStrategy: Run A* search", verbose=False) as result:
            try:
                path, path_cost = self.run_a_star(self.graph, "s_init", "s_final")
            except nx.NetworkXNoPath:
                # This can happen if we have queried all possible modules
                # in the module graph.
                path = {}
                path_cost = 0.0
        t_run_a_star = result["time"]
        timing_info = {
            "t_create_graph": t_create_graph,
            "t_run_a_star": t_run_a_star,
        }
        # Step 3: Get the query from the result
        # (just choose the first one in the sequence)
        for module, is_query in path.items():
            if is_query:
                return module, timing_info, {"path_cost": path_cost}

        # If no module is found, return None
        return None, timing_info, {"path_cost": path_cost}

    def add_queried_module(self, module_name: str) -> None:
        """Add a module to the set of queried modules, and update the internal
        state of the strategy."""
        assert self.graph is not None, "Graph should not be None."
        super().add_queried_module(module_name)

        # Update the graph to remove the query edge
        # for the module that we just queried.
        query_module_index = self.modules_list.index(module_name)
        if query_module_index == 0:
            # Remove the query edge from s_init to s_{module},0.
            self.graph.remove_edge("s_init", f"s_{module_name},0")
        else:
            # Remove the query edge from s_{module_start},j to s_{module},k,
            # where k = concat(j, "1").
            j_binary_str = "1" * (query_module_index)
            k_binary_str = j_binary_str + "0"
            j = int(j_binary_str, 2)
            k = int(k_binary_str, 2)
            self.graph.remove_edge(
                f"s_{self.modules_list[query_module_index - 1]},{j}",
                f"s_{module_name},{k}",
            )

    def remove_queried_modules(
        self, module_graph: ModuleGraph, module_names: set[str]
    ) -> None:
        """Remove a set of modules from the set of queried modules, and update
        the internal state of the strategy."""
        assert self.graph is not None, "Graph should not be None."
        super().remove_queried_modules(module_graph, module_names)

        query_costs_str = {
            module.get_name(): module.get_expert_query_cost()
            for module in module_graph.get_modules()
            if module.get_name() in self.get_all_queryable_modules(module_graph)
        }

        # Update the graph to add back in the query edges
        # for all of the modules in the module_names set.
        for module_name in module_names:
            query_module_index = self.modules_list.index(module_name)
            if query_module_index == 0:
                # Add the query edge from s_init to s_{module},0.
                query_cost = query_costs_str[self.modules_list[0]]
                self.graph.add_edge(
                    "s_init",
                    f"s_{module_name},0",
                    key="a_query",
                    cost=query_cost,
                )
            else:
                # Add the query edge from s_{module_start},j to s_{module},k,
                # where k = concat(j, "1").
                j_binary_str = "1" * (query_module_index)
                k_binary_str = j_binary_str + "0"
                j = int(j_binary_str, 2)
                k = int(k_binary_str, 2)
                query_cost = query_costs_str[module_name]
                self.graph.add_edge(
                    f"s_{self.modules_list[query_module_index - 1]},{j}",
                    f"s_{module_name},{k}",
                    key="a_query",
                    cost=query_cost,
                )
