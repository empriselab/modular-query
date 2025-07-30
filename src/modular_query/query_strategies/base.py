"""Base class for query strategies."""

import abc
from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule


class QueryStrategy(abc.ABC):
    """A strategy for deciding which module (if any) to expert-query.

    At the moment, this assumes a single-round choice:
        1. The original modules are queried first.
        2. Given the computed values, this strategy chooses a set to query.
        3. The chosen module (if any) is queried for its expert value.
    """

    def __init__(
        self, correct_answer_cost: float, incorrect_answer_cost: float
    ) -> None:
        self.correct_answer_cost = correct_answer_cost
        self.incorrect_answer_cost = incorrect_answer_cost

    @abc.abstractmethod
    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None]:
        """Given a module graph and the already computed values, return a
        module name to query and an optional dictionary of timing information.

        Can also be None if we don't want to query for any of the
        modules.
        """

    def get_all_queryable_modules(
        self,
        module_graph: ModuleGraph,
    ) -> set[str]:
        """Return all queryable modules in the module graph."""
        to_query = set()
        for module in module_graph.get_modules():
            if isinstance(module, StateModule):
                continue
            to_query.add(module.get_name())
        return to_query

    def reset(self) -> None:
        """Reset the state of the query strategy."""
