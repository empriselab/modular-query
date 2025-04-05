"""Utility functions."""

from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tomsutils.utils import draw_dag

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module


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
) -> Module:
    """Module factory."""

    class _CustomModule(Module):

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


def generate_random_and_gate_module_graph(
    num_modules: int,
    edge_probability: float,
    confidence_sampler: Callable[[np.random.Generator], float],
    query_cost_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
) -> ModuleGraph:
    """Generate a random module graph where all modules are AND gates."""

    # The expert function is the same for all modules.
    def expert_fn(inputs: dict[str, Any]) -> Any:
        """Expert function for the AND gate."""
        return all(inputs.values())

    def correct_fn(confidence: float, inputs: dict[str, Any]) -> tuple[Any, float]:
        """A function for a correct module."""
        return expert_fn(inputs), confidence

    def incorrect_fn(confidence: float, inputs: dict[str, Any]) -> tuple[Any, float]:
        """A function for an incorrect module."""
        return not expert_fn(inputs), confidence

    # Create the modules.
    modules: list[Module] = []
    for num in range(num_modules):
        module_name = f"module_{num}"
        prob_correct = confidence_sampler(rng)
        assert 0 <= prob_correct <= 1
        if rng.uniform() < prob_correct:
            fn = partial(correct_fn, prob_correct)
        else:
            fn = partial(incorrect_fn, prob_correct)
        query_cost = query_cost_sampler(rng)
        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost,
            expert_fn=expert_fn,
        )
        modules.append(module)

    # Create a random directed acyclic graph (DAG) for the module dependencies.
    module_to_parents: dict[Module, list[Module]] = {}
    for i in range(num_modules):
        module = modules[i]
        parents = []
        for j in range(i):
            if rng.uniform() < edge_probability:
                parents.append(modules[j])
        if i > 0 and not parents:
            # Ensure at least one parent for the first module to avoid isolated nodes.
            parent_idx = rng.choice(range(i))
            parents.append(modules[parent_idx])
        module_to_parents[module] = parents

    # Create the module graph.
    module_graph = ModuleGraph(module_to_parents)
    return module_graph
