"""Module graph."""

from collections import deque
from typing import Any

from modular_query.modules import Module
from modular_query.utils import print_and_log


class ModuleGraph:
    """A graph of modules."""

    def __init__(
        self, module_to_parents: dict[Module, list[Module]], verbose: bool = False
    ) -> None:
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
        assert len(roots) == 1, "Root module must be unique"
        self.root = roots[0]
        leaves = [m for m, c in self.module_to_children.items() if not c]
        assert len(leaves) == 1, "Leaf module must be unique"
        self.leaf = leaves[0]

        # Compute a topological order of the modules from the root to the leaf.
        self.topo_order: list[Module] = []
        queue = deque([self.root])
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

    def get_module_by_name(self, name: str | None) -> Module | None:
        """Get a module by name."""
        for module in self.get_modules():
            if module.get_name() == name:
                return module
        return None

    def compute_values(
        self, expert_query_module_names: set[str]
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
            if module.get_name() in expert_query_module_names:
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
