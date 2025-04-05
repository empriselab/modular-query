"""Tests for modules.py."""

from typing import Any

from modular_query.modules import Module


def test_module():
    """Tests for Module()."""

    class MyCustomModule(Module):
        """A custom module for testing."""

        @classmethod
        def get_name(cls) -> str:
            return "MyCustomModule"

        def call(self, inputs: dict[str, str]) -> tuple[str, float]:
            return "test_value", 0.5

        def get_expert_query_cost(self) -> float:
            return 0.0

        def call_expert(self, inputs: dict[str, Any]) -> Any:
            return self.call(inputs)[0]

    module = MyCustomModule()
    assert module.get_name() == "MyCustomModule"
    value, confidence = module.call({"other_module": "value"})
    assert value == "test_value"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    expert_value = module.call_expert({"other_module": "value"})
    assert (
        expert_value == "test_value"
    ), f"Expected expert value to be 'test_value', got {expert_value}"
    query_cost = module.get_expert_query_cost()
    assert query_cost == 0.0, f"Expected query cost to be 0.0, got {query_cost}"
