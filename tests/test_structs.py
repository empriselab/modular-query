"""Tests for structs.py."""

from typing import Any

from modular_query.structs import GroundTruthModule, ModularPolicy, Module


def test_module():
    """Tests for Module()."""

    class MyCustomModule(Module):
        """A custom module for testing."""

        @classmethod
        def get_name(cls) -> str:
            return "MyCustomModule"

        def call(self, inputs: dict[str, str]) -> tuple[str, float]:
            return "test_value", 0.5

    module = MyCustomModule()
    assert module.get_name() == "MyCustomModule"
    value, confidence = module.call({"other_module": "value"})
    assert value == "test_value"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"


def test_ground_truth_module():
    """Tests for GroundTruthModule()."""

    class MyGroundTruthModule(GroundTruthModule):
        """A ground-truth module for testing."""

        @classmethod
        def get_name(cls) -> str:
            return "MyGroundTruthModule"

        def get_query_cost(self) -> float:
            return 1.0

        def get_ground_truth(self, inputs: dict[str, str]) -> str:
            return "ground_truth_value"

    module = MyGroundTruthModule()
    assert module.get_query_cost() == 1.0
    value, confidence = module.call({"other_module": "value"})
    assert value == "ground_truth_value"
    assert confidence == 1.0, "Confidence should be 1.0 for ground truth"


def test_modular_policy():
    """Tests for ModularPolicy()."""

    class _M1(Module):

        @classmethod
        def get_name(cls) -> str:
            return "M1"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # The root will receive {"state": some_state}.
            # Return a simple constant for demonstration.
            return 1, 1.0

    class _M2(Module):
        @classmethod
        def get_name(cls) -> str:
            return "M2"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M1": 1} in this test
            return 2, 1.0

    class _M3(Module):
        @classmethod
        def get_name(cls) -> str:
            return "M3"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M1": 1} in this test
            return 3, 1.0

    class _M4(Module):
        @classmethod
        def get_name(cls) -> str:
            return "M4"

        def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
            # inputs should be {"M2": 2, "M3": 3} in this test
            value_m2 = inputs["M2"]
            value_m3 = inputs["M3"]
            return value_m2 + value_m3, 1.0

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

    policy = ModularPolicy(module_to_parents)

    # The state we pass is arbitrary since M1 ignores it and returns 1 anyway.
    state = "dummy state"

    # The final result from M4 should be 2 + 3 = 5 in this example.
    action = policy.get_action(state)
    assert action == 5, f"Expected 5 from the final module, got {action}"
