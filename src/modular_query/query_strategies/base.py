"""Base class for query strategies."""

import abc
from typing import Any

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module


class QueryStrategy(abc.ABC):
    """A strategy for deciding which modules to expert-query.

    At the moment, this assumes a single-round choice:
        1. The original modules are queried first.
        2. Given the computed values, this strategy chooses a set to query.
        3. The chosen modules are queried for their expert values.
    """

    @abc.abstractmethod
    def get_expert_query_modules(
        self, module_graph: ModuleGraph, computed_values: dict[Module, Any]
    ) -> set[str]:
        """Given a module graph and the already computed values, return a set
        of module names to query."""
