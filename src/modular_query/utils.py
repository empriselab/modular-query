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
    confidence_sampler: Callable[[np.random.Generator], float],
    query_cost_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    is_policy: bool = False,
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

        prob_correct = confidence_sampler(rng)
        assert 0 <= prob_correct <= 1
        and_or_or = rng.choice(["and", "or"])
        not_out = rng.choice([True, False])
        expert_fn = partial(expert, and_or_or, not_out)
        if rng.uniform() < prob_correct:
            fn = partial(correct_fn, and_or_or, not_out, prob_correct)
        else:
            fn = partial(incorrect_fn, and_or_or, not_out, prob_correct)
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
