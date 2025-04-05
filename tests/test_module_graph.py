"""Tests for module_graph.py."""

from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module


def test_module_graph():
    """Tests for ModuleGraph()."""

    class _M1(Module):

        @classmethod
        def get_name(cls) -> str:
            return "M1"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # Return a simple constant for demonstration.
            return 1, 1.0

        def get_expert_query_cost(self) -> float:
            return 0.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    class _M2(Module):

        @classmethod
        def get_name(cls) -> str:
            return "M2"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M1": 1} in this test
            return 2, 0.5

        def get_expert_query_cost(self) -> float:
            return 0.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    class _M3(Module):

        @classmethod
        def get_name(cls) -> str:
            return "M3"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M1": 1} in this test
            return 3, 0.5

        def get_expert_query_cost(self) -> float:
            return 1.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return 10

    class _M4(Module):
        @classmethod
        def get_name(cls) -> str:
            return "M4"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M2": 2, "M3": 3} in this test
            value_m2 = inputs["M2"]
            value_m3 = inputs["M3"]
            return value_m2 + value_m3, 1.0

        def get_expert_query_cost(self) -> float:
            return 0.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    # Create the diamond structure:
    #
    #   M1
    #  /  \
    # M2  M3
    #  \  /
    #   M4
    #
    # So M1 has no parents, M2 and M3 have [M1], M4 has [M2, M3].
    m1, m2, m3, m4 = _M1(), _M2(), _M3(), _M4()
    module_to_parents = {
        m1: [],
        m2: [m1],
        m3: [m1],
        m4: [m2, m3],
    }

    graph = ModuleGraph(module_to_parents)

    # Uncomment to create a visualization of the graph.
    # from modular_query.utils import draw_module_graph
    # from pathlib import Path
    # draw_module_graph(graph, Path("tests/test_graph.png"))

    values, confidences, query_cost = graph.compute_values(
        expert_query_module_names=set()
    )
    assert (
        abs(query_cost - 0.0) < 1e-6
    ), f"Expected total query cost to be 0.0, got {query_cost}"
    assert len(values) == 4, f"Expected 4 modules to be computed, got {len(values)}"
    # Ensure the values returned are correct.
    assert values[m1] == 1, f"Expected M1 to return 1, got {values[m1]}"
    assert values[m2] == 2, f"Expected M2 to return 2, got {values[m2]}"
    assert values[m3] == 3, f"Expected M3 to return 3, got {values[m3]}"
    assert values[m4] == 5, f"Expected M4 to return 5 (2 + 3), got {values[m4]}"
    assert (
        confidences[m1] == 1.0
    ), f"Expected M1 confidence to be 1.0, got {confidences[m1]}"
    assert (
        confidences[m2] == 0.5
    ), f"Expected M2 confidence to be 0.5, got {confidences[m2]}"
    assert (
        confidences[m3] == 0.5
    ), f"Expected M3 confidence to be 0.5, got {confidences[m3]}"
    assert (
        confidences[m4] == 1.0
    ), f"Expected M4 confidence to be 1.0, got {confidences[m4]}"

    # Now test with M3 as an expert query:
    values, confidences, query_cost = graph.compute_values(
        expert_query_module_names={"M3"}
    )
    # Now M3 should be computed using the expert, which returns 10.
    assert (
        abs(query_cost - 1.0) < 1e-6
    ), f"Expected total query cost to be 1.0 (for M3), got {query_cost}"
    assert len(values) == 4, f"Expected 4 modules to be computed, got {len(values)}"
    # Ensure the values returned are correct with M3 as an expert.
    assert values[m1] == 1, f"Expected M1 to return 1, got {values[m1]}"
    assert values[m2] == 2, f"Expected M2 to return 2, got {values[m2]}"
    # M3 should now return 10 due to the expert query.
    assert values[m3] == 10, f"Expected M3 to return 10 (expert), got {values[m3]}"
    # M4 should now return 12 (2 + 10) because M3 was computed as an expert.
    assert values[m4] == 12, f"Expected M4 to return 12 (2 + 10), got {values[m4]}"
    assert (
        confidences[m1] == 1.0
    ), f"Expected M1 confidence to be 1.0, got {confidences[m1]}"
    assert (
        confidences[m2] == 0.5
    ), f"Expected M2 confidence to be 0.5, got {confidences[m2]}"
    # M3 should now have confidence 1.0 because it was computed using the expert.
    assert (
        confidences[m3] == 1.0
    ), f"Expected M3 confidence to be 1.0, got {confidences[m3]}"
    assert (
        confidences[m4] == 1.0
    ), f"Expected M4 confidence to be 1.0, got {confidences[m4]}"
