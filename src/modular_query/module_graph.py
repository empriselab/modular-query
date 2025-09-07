"""Module graph."""

from collections import deque
from typing import Any

from modular_query.modules import Module
from modular_query.utils import print_and_log


class ModuleGraph:
    """A graph of modules."""

    def __init__(
        self,
        module_to_parents: dict[Module, list[Module]],
        root_leaf_check: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        root_leaf_check: if True,
        then we assert that there is a unique root and leaf module.
        If False, then we do not assert this.
        """
        self.module_to_parents = module_to_parents
        self.module_to_children: dict[Module, list[Module]] = {
            m: [] for m in module_to_parents
        }
        for module, parents in self.module_to_parents.items():
            for parent in parents:
                if parent not in self.module_to_children:
                    self.module_to_children[parent] = []
                self.module_to_children[parent].append(module)

        # Derive the unique root and the unique leaf, which we assume correspond
        # to state and action respectively.
        roots = [m for m, p in self.module_to_parents.items() if not p]
        if root_leaf_check:
            assert len(roots) == 1, "Root module must be unique"
        if len(roots) > 0:
            self.root = roots[0]
        leaves = [m for m, c in self.module_to_children.items() if not c]
        if root_leaf_check:
            assert len(leaves) == 1, "Leaf module must be unique."
        if len(leaves) > 0:
            self.leaf = leaves[0]

        # Compute a topological order of the modules from the root to the leaf.
        self.topo_order: list[Module] = []
        queue = deque(roots)
        visited = set()

        while queue:
            module = queue.popleft()
            if module in visited:
                continue
            visited.add(module)
            self.topo_order.append(module)

            # For each child, only enqueue it if *all* its parents are visited.
            for child in self.module_to_children.get(module, []):
                if all(parent in visited for parent in self.module_to_parents[child]):
                    queue.append(child)

        # Set verbose flag. Outputs model forward passes.
        self.verbose = verbose

    def get_modules(self) -> set[Module]:
        """Get all modules in the graph."""
        return set(self.module_to_parents)

    def get_module_str_to_module(self) -> dict[str, Module]:
        """Get a dictionary mapping module strings to modules."""
        return {m.get_name(): m for m in self.module_to_parents}

    def get_module_by_name(self, name: str | None) -> Module | None:
        """Get a module by name."""
        for module in self.get_modules():
            if module.get_name() == name:
                return module
        return None

    def get_downstream_modules(self, module_name: str) -> set[str]:
        """Get all downstream modules of a given module."""
        module = self.get_module_str_to_module()[module_name]
        # successively get all children of the module, and all downstream modules.
        downstream_modules = set()
        queue = deque([module])
        while queue:
            module = queue.popleft()
            downstream_modules.add(module.get_name())
            child_modules = self.module_to_children[module]
            # don't add child modules that are already in the queue,
            # or already in the downstream modules set.
            child_modules = [
                c
                for c in child_modules
                if c not in queue and c.get_name() not in downstream_modules
            ]
            queue.extend(child_modules)
        return downstream_modules

    def validate_graph_connectivity(self) -> dict[str, Any]:
        """Validate the graph connectivity and return diagnostic information.

        Returns a dictionary with information about:
        - Number of modules
        - Number of root modules (no parents)
        - Number of leaf modules (no children)
        - Number of isolated modules (no parents and no children)
        - Number of modules in topological order
        - Any connectivity issues found
        """
        all_modules = set(self.module_to_parents.keys())
        root_modules = [m for m in all_modules if not self.module_to_parents[m]]
        leaf_modules = [
            m for m in all_modules if not self.module_to_children.get(m, [])
        ]

        # Find isolated modules (no parents and no children)
        isolated_modules = []
        for module in all_modules:
            if not self.module_to_parents[module] and not self.module_to_children.get(
                module, []
            ):
                isolated_modules.append(module)

        # Check if all modules are in topological order
        modules_in_topo = set(self.topo_order)
        missing_from_topo = all_modules - modules_in_topo
        extra_in_topo = modules_in_topo - all_modules

        # Check for cycles (simplified check)
        has_cycle = False
        if len(self.topo_order) != len(all_modules):
            has_cycle = True

        return {
            "total_modules": len(all_modules),
            "root_modules": len(root_modules),
            "leaf_modules": len(leaf_modules),
            "isolated_modules": len(isolated_modules),
            "modules_in_topo": len(self.topo_order),
            "missing_from_topo": len(missing_from_topo),
            "extra_in_topo": len(extra_in_topo),
            "has_cycle": has_cycle,
            "root_module_names": [m.get_name() for m in root_modules],
            "isolated_module_names": [m.get_name() for m in isolated_modules],
            "missing_module_names": [m.get_name() for m in missing_from_topo],
        }

    def compute_values(
        self, expert_query_module_names: set[str], expert_values_cache: dict[str, Any]
    ) -> tuple[dict[Module, Any], dict[Module, float], float]:
        """Recompute values and confidences for all modules.

        Return the values and also the overall expert query cost.
        """

        # Total query cost.
        total_query_cost = 0.0

        # Traverse in topological order, computing each module’s output exactly once.
        computed_values: dict[Module, Any] = {}
        computed_confidences: dict[Module, float] = {}
        for module in self.topo_order:
            parent_outputs = {}
            # Gather all parent outputs from previously computed modules.
            for parent in self.module_to_parents[module]:
                parent_outputs[parent.get_name()] = computed_values[parent]

            # Invoke the module call.
            if (
                module.get_name() in expert_values_cache
                and expert_values_cache[module.get_name()] is not None
            ):
                # Use the cached expert value, and set confidence to 1.0.
                value = expert_values_cache[module.get_name()]
                computed_values[module] = value
                computed_confidences[module] = 1.0
            elif module.get_name() in expert_query_module_names:
                # If this module is the module to query, call the expert.
                value = module.call_expert(parent_outputs)
                # Use the expert's value, and set confidence to 1.0.
                computed_values[module] = value
                computed_confidences[module] = 1.0
                # Add the expert query cost for this module.
                query_cost = module.get_expert_query_cost()
                assert (
                    query_cost > 0
                ), f"Module {module.get_name()}: Query cost must be positive."
                total_query_cost += query_cost
            else:
                value, confidence = module.call(parent_outputs)
                computed_values[module] = value
                computed_confidences[module] = confidence

            # For logging purposes, print the module's inputs,
            # the output, and the ground-truth output.
            if self.verbose:
                try:
                    expert_value = module.call_expert(parent_outputs)
                except NotImplementedError:
                    expert_value = None
                print_and_log(
                    f"Module: {module.get_name()}, Inputs: {parent_outputs}, "
                    f"Output: {value}, Ground-truth value (for given input): "
                    f"{expert_value}"
                )

        return computed_values, computed_confidences, total_query_cost
