"""Tests for utils.py."""

import numpy as np

from modular_query.utils import generate_random_logic_gate_module_graph


def test_generate_random_and_gate_module_graph():
    """Tests for generate_random_and_gate_module_graph()."""

    num_modules = 8
    module_graph = generate_random_logic_gate_module_graph(
        num_modules=num_modules,
        edge_probability=0.5,
        confidence_sampler=lambda rng: rng.uniform(0.5, 1.0),
        query_cost_sampler=lambda rng: rng.uniform(0.1, 10.0),
        rng=np.random.default_rng(seed=123),
    )

    # Ensure the module graph has the expected number of modules.
    assert (
        len(module_graph.get_modules()) == num_modules
    ), f"Expected {num_modules} modules in the graph."
    # Ensure that the modules are connected in some way.
    module_names = {m.get_name() for m in module_graph.get_modules()}
    assert (
        len(module_names) == num_modules
    ), "Expected unique module names in the graph."

    # Uncomment to visualize.
    # from modular_query.utils import draw_module_graph
    # from pathlib import Path
    # draw_module_graph(module_graph, Path("tests/random_graph.png"))
