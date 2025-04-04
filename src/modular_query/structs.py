"""Data structures."""

import abc
from collections import deque
from typing import Any, Generic, TypeVar

State = TypeVar("State")
Action = TypeVar("Action")


class Module(abc.ABC):
    """One module in the overall modular policy."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the module."""

    @abc.abstractmethod
    def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
        """Given inputs of {other module name: value}, return a value for this
        module, and also a confidence score between 0 and 1 that indicates the
        probability that the returned value is correct."""


class GroundTruthModule(Module):
    """A ground-truth module that always returns the correct value."""

    @abc.abstractmethod
    def get_query_cost(self) -> float:
        """Get the cost of querying this module (higher is worse)."""

    @abc.abstractmethod
    def get_ground_truth(self, inputs: dict[str, Any]) -> Any:
        """Get the ground-truth value for this module."""

    def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
        return self.get_ground_truth(inputs), 1.0


class ModularPolicy(Generic[State, Action]):
    """A modular policy defined by a graph of modules."""

    def __init__(self, module_to_parents: dict[Module, list[Module]]) -> None:
        self._module_to_parents = module_to_parents
        self._module_to_children: dict[Module, list[Module]] = {
            m: [] for m in module_to_parents
        }
        for module, parents in self._module_to_parents.items():
            for parent in parents:
                if parent not in self._module_to_children:
                    self._module_to_children[parent] = []
                self._module_to_children[parent].append(module)

        # Derive the unique root and the unique leaf, which we assume correspond
        # to state and action respectively.
        roots = [m for m, p in self._module_to_parents.items() if not p]
        assert len(roots) == 1, "Root module must be unique"
        self._root = roots[0]
        leaves = [m for m, c in self._module_to_children.items() if not c]
        assert len(leaves) == 1, "Leaf module must be unique"
        self._leaf = leaves[0]

        # Compute a topological order of the modules from the root to the leaf.
        self._topo_order: list[Module] = []
        queue = deque([self._root])
        visited = set()

        while queue:
            module = queue.popleft()
            if module in visited:
                continue
            visited.add(module)
            self._topo_order.append(module)

            # For each child, only enqueue it if *all* its parents are visited.
            for child in self._module_to_children.get(module, []):
                if all(parent in visited for parent in self._module_to_parents[child]):
                    queue.append(child)

    def get_action(self, state: State) -> Action:
        """Get the action from the modular policy given the state."""

        # Traverse in topological order, computing each module’s output exactly once.
        computed_values: dict[Module, Any] = {}
        for module in self._topo_order:
            parent_outputs = {}
            # Gather all parent outputs from previously computed modules.
            for parent in self._module_to_parents[module]:
                parent_outputs[parent.get_name()] = computed_values[parent]

            # The root module also gets the raw state.
            if module == self._root:
                parent_outputs["state"] = state

            # Invoke the module call.
            value, _ = module.call(parent_outputs)
            computed_values[module] = value

        # 3) The leaf module’s output is our final action.
        return computed_values[self._leaf]
