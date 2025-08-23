"""Tests for module_utils.py."""

import numpy as np

from modular_query.module_utils import (
    generate_random_logic_gate_module_graph,
    generate_random_module_graph,
    generate_random_polynomial_module_graph,
)


def test_generate_random_logic_gate_module_graph():
    """Tests for generate_random_logic_gate_module_graph()."""

    num_modules = 8
    module_graph = generate_random_logic_gate_module_graph(
        num_modules=num_modules,
        edge_probability=0.5,
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


def test_generate_random_polynomial_module_graph():
    """Tests for generate_random_polynomial_module_graph()."""

    num_modules = 8
    module_graph = generate_random_polynomial_module_graph(
        num_modules=num_modules,
        edge_probability=0.5,
        query_cost=1.0,  # Uniform query cost for all modules of 1.0
        rng=np.random.default_rng(seed=123),
        num_incorrect_modules=1,
        incorrect_module_confidence=0.1,
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

    # Ensure that forward pass runs without errors.
    state = 0
    all_queryable_module_names = {m.get_name() for m in module_graph.get_modules()}
    all_queryable_module_names.remove("state")
    module_graph.root.set_state(state)
    _, _, _ = module_graph.compute_values(
        expert_query_module_names=all_queryable_module_names, expert_values_cache={}
    )


def test_generate_random_and_gate_module_graph():
    """Tests for generate_random_and_gate_module_graph()."""

    num_modules = 8
    module_graph = generate_random_module_graph(
        num_modules=num_modules,
        edge_probability=0.5,
        query_cost=1.0,  # Uniform query cost for all modules of 1.0
        rng=np.random.default_rng(seed=123),
        num_incorrect_modules=1,
        incorrect_module_confidence=0.1,
        redundancy="AND",
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

    # Ensure that forward pass runs without errors.
    state = 0  # False
    all_queryable_module_names = {m.get_name() for m in module_graph.get_modules()}
    all_queryable_module_names.remove("state")
    module_graph.root.set_state(state)
    computed_values, _, _ = module_graph.compute_values(
        expert_query_module_names=set(), expert_values_cache={}
    )

    # Verify that the final value is correct for the AND gates (i.e.
    # FALSE, given the state is 0).
    leaf_module = module_graph.leaf
    assert (
        computed_values[leaf_module] == 0
    ), f"Expected leaf module value to be 0, got {computed_values[leaf_module]}"

    # Even for state = 1, because there is a single incorrect module, the
    # leaf module value should still be 0.
    state = 1  # True
    module_graph.root.set_state(state)
    computed_values, _, _ = module_graph.compute_values(
        expert_query_module_names=set(), expert_values_cache={}
    )
    assert computed_values[leaf_module] == 0, (
        "Expected leaf module value to be 0 (even with true initial state), "
        f"got {computed_values[leaf_module]}"
    )
