"""Base class modules."""

import abc
from typing import Any


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

    @abc.abstractmethod
    def get_expert_query_cost(self) -> float:
        """Get the cost of querying an expert for this module."""

    @abc.abstractmethod
    def call_expert(self, inputs: dict[str, Any]) -> Any:
        """Get the ground-truth value for this module."""


class StateModule(Module):
    """Special type of module that represents the input state to a policy."""

    def __init__(self) -> None:
        self._current_state: Any | None = None

    def set_state(self, state: Any) -> None:
        """Set the current state for this state module."""
        self._current_state = state

    @classmethod
    def get_name(cls) -> str:
        return "state"

    def call(self, inputs: dict[str, Any]) -> tuple[Any, float]:
        assert self._current_state is not None
        return self._current_state, 1.0

    def get_expert_query_cost(self) -> float:
        raise NotImplementedError("No expert for state modules.")

    def call_expert(self, inputs: dict[str, Any]) -> Any:
        raise NotImplementedError("No expert for state modules.")


class ActionModule(Module, abc.ABC):
    """Special type of module that represents the output action of a policy."""

    @classmethod
    def get_name(cls) -> str:
        return "action"
