"""Utility functions."""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Type

import numpy as np
from tomsutils.utils import draw_dag

from modular_query.module_graph import ModuleGraph
from modular_query.modules import ActionModule, Module, StateModule


def draw_module_graph(module_graph: ModuleGraph, outfile: Path) -> None:
    """Draw a visualization of the module graph."""
    edges: list[tuple[str, str]] = []
    for module, parents in module_graph.module_to_parents.items():
        for parent in parents:
            edges.append((parent.get_name(), module.get_name()))
    draw_dag(edges, outfile)


def create_module(
    name: str,
    fn: Callable[[dict[str, Any]], tuple[Any, float]],
    query_cost: float,
    expert_fn: Callable[[dict[str, Any]], Any],
    ParentModuleClass: Type[Module] = Module,
) -> Module:
    """Module factory."""

    class _CustomModule(ParentModuleClass):  # type: ignore

        @classmethod
        def get_name(cls) -> str:
            return name

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            return fn(inputs)

        def get_expert_query_cost(self) -> float:
            return query_cost

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return expert_fn(inputs)

    return _CustomModule()


def generate_random_logic_gate_module_graph(
    num_modules: int,
    edge_probability: float,
    query_cost_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    is_policy: bool = False,
    num_incorrect_modules: int = 1,
    incorrect_module_confidence: float = 0.1,
) -> ModuleGraph:
    """Generate a random module graph where all modules are logical."""

    # The expert function is the same for all modules.
    def expert(and_or_or: str, not_out: bool, inputs: dict[str, Any]) -> Any:
        """Expert function for the AND gate."""
        if and_or_or == "and":
            out = all(inputs.values())
        elif and_or_or == "or":
            out = any(inputs.values())
        else:
            raise NotImplementedError
        if not_out:
            out = not not_out
        return out

    def correct_fn(
        and_or_or: str, not_out: bool, confidence: float, inputs: dict[str, Any]
    ) -> tuple[Any, float]:
        """A function for a correct module."""
        return expert(and_or_or, not_out, inputs), confidence

    def incorrect_fn(
        and_or_or: str, not_out: bool, confidence: float, inputs: dict[str, Any]
    ) -> tuple[Any, float]:
        """A function for an incorrect module."""
        return not expert(and_or_or, not_out, inputs), confidence

    # Determine which modules will be unconfident and incorrect.
    incorrect_module_nums = rng.choice(
        num_modules, size=num_incorrect_modules, replace=False
    )

    # Create the modules.
    modules: list[Module] = []
    for num in range(num_modules):
        if num == 0 and is_policy:
            # Ensure the first module is a StateModule if this is a policy graph.
            module_name = "state"
            module: Module = StateModule()
            modules.append(module)
            continue

        if num == num_modules - 1 and is_policy:
            ParentModuleClass: Type[Module] = ActionModule  # type: ignore
            module_name = "action"
        else:
            ParentModuleClass = Module  # type: ignore
            module_name = f"module_{num}"

        and_or_or = rng.choice(["and", "or"])
        not_out = rng.choice([True, False])
        expert_fn = partial(expert, and_or_or, not_out)
        if num in incorrect_module_nums:
            fn = partial(incorrect_fn, and_or_or, not_out, incorrect_module_confidence)
        else:
            fn = partial(correct_fn, and_or_or, not_out, 1.0)
        query_cost = query_cost_sampler(rng)
        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost,
            expert_fn=expert_fn,
            ParentModuleClass=ParentModuleClass,
        )
        modules.append(module)

    # Create a random directed acyclic graph (DAG) for the module dependencies.
    module_to_parents: dict[Module, list[Module]] = {}
    leaves: set[Module] = set()
    for i in range(num_modules):
        module = modules[i]
        parents = []
        # Force the last module to be the only leaf.
        if i == num_modules - 1:
            parents = sorted(leaves, key=lambda m: m.get_name())
        else:
            for j in range(i):
                if rng.uniform() < edge_probability:
                    parents.append(modules[j])
        if i > 0 and not parents:
            # Ensure at least one parent for the first module to avoid isolated nodes.
            parent_idx = rng.choice(range(i))
            parents.append(modules[parent_idx])
        module_to_parents[module] = parents
        leaves -= set(parents)
        leaves.add(module)

    # Create the module graph.
    module_graph = ModuleGraph(module_to_parents)
    return module_graph


def get_query_set_expected_cost(
    query_set: set[str],
    module_graph: ModuleGraph,
    computed_confidences: dict[Module, float],
    correct_answer_cost: float,
    incorrect_answer_cost: float,
) -> float:
    """Calculate the total cost of querying the modules in the given query
    set."""

    # Compute query cost.
    module_name_to_module = {m.get_name(): m for m in module_graph.get_modules()}
    total_query_cost = 0.0
    for module_name in query_set:
        module = module_name_to_module[module_name]
        total_query_cost += module.get_expert_query_cost()

    # Compute probability of being correct, assuming that a module confidence
    # is exactly the probability that it is correct.
    probability_of_correct_answer = 1.0
    nonquery_set = set(module_name_to_module) - query_set
    for module_name in nonquery_set:
        module = module_name_to_module[module_name]
        probability_of_correct_answer *= computed_confidences[module]

    # Compute total cost.
    combined_cost = total_query_cost + (
        correct_answer_cost * probability_of_correct_answer
        + incorrect_answer_cost * (1 - probability_of_correct_answer)
    )

    return combined_cost
