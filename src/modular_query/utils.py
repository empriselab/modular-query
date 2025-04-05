"""Utility functions."""

from pathlib import Path

from tomsutils.utils import draw_dag

from modular_query.module_graph import ModuleGraph


def draw_module_graph(module_graph: ModuleGraph, outfile: Path) -> None:
    """Draw a visualization of the module graph."""
    edges: list[tuple[str, str]] = []
    for module, parents in module_graph.module_to_parents.items():
        for parent in parents:
            edges.append((parent.get_name(), module.get_name()))
    draw_dag(edges, outfile)
