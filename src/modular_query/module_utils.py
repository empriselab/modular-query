"""Utility functions (that use modules) for the modular query framework."""

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


def construct_graph(
    modules: list[Module],
    num_modules: int,
    edge_probability: float,
    rng: np.random.Generator,
    expert_query_confidence: float,
) -> ModuleGraph:
    """Construct a random directed acyclic graph (DAG) for the module
    dependencies."""
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
    module_graph = ModuleGraph(
        module_to_parents,
        expert_query_confidence=expert_query_confidence,
        expert_value_rng=None,
    )

    return module_graph


def construct_graph_top_bottom(
    state_module: StateModule,
    top_modules: list[Module],
    bottom_modules: list[Module],
    action_module: ActionModule,
    edge_probability: float,
    rng: np.random.Generator,
    expert_query_confidence: float,
) -> ModuleGraph:
    """Construct a module graph with a single top gate type and a single bottom
    gate type.

    The structure is as follows:
    - state_module is the root (has no parents)
    - top_modules are all descendants of state_module. One of the top_modules will have
    no top_modules as parents, and will be the common parent of the bottom_modules.
    - bottom_modules are all descendants of the final top_module.
    - action_module is the leaf (has no children)
    """
    module_to_parents: dict[Module, list[Module]] = {}
    leaves: set[Module] = set()

    # Ensure state module has no parents
    module_to_parents[state_module] = []
    leaves.add(state_module)

    # Construct top modules with guaranteed connectivity
    # The last top module should be the only leaf.
    for i, module in enumerate(top_modules):
        parents: list[Module] = []
        if i == 0:
            # First top module must have state_module as parent to ensure connectivity
            parents = [state_module]
        elif i == len(top_modules) - 1:
            # Last top module should be the only leaf.
            parents = sorted(leaves, key=lambda m: m.get_name())
        else:
            # Add random parents based on edge_probability
            for j in range(i):
                if rng.uniform() < edge_probability:
                    parents.append(top_modules[j])

            # Ensure at least one parent to avoid isolated nodes
            # (same as construct_graph)
            if not parents:
                parent_idx = rng.choice(range(i))
                parents.append(top_modules[parent_idx])

        module_to_parents[module] = parents
        leaves -= set(parents)
        leaves.add(module)

    # Reset 'leaves' and 'parents' before constructing the bottom modules.
    leaves = set()

    # Construct bottom modules with guaranteed connectivity
    # The action module should be the only leaf.
    for i, module in enumerate(bottom_modules + [action_module]):
        parents = []
        if i == 0:
            # The first bottom module must have at least one top module as parent
            if top_modules:
                parents = [top_modules[-1]]  # Connect to last top module
            else:
                # If no top modules, connect to state module
                parents = [state_module]
        elif i == len(bottom_modules + [action_module]) - 1:
            # Last bottom module should be the only leaf.
            parents = sorted(leaves, key=lambda m: m.get_name())
        else:
            # Add random parents based on edge_probability
            for j in range(i):
                if rng.uniform() < edge_probability:
                    parents.append(bottom_modules[j])

            # Ensure at least one parent to avoid isolated nodes
            # (same as construct_graph)
            if not parents:
                if i <= len(bottom_modules):
                    # Connect to a random previous bottom module
                    parent_idx = rng.choice(range(i))
                    parents.append(bottom_modules[parent_idx])
                else:
                    # Action module - connect to a random bottom module
                    if bottom_modules:
                        parent_idx = rng.choice(range(len(bottom_modules)))
                        parents.append(bottom_modules[parent_idx])
                    else:
                        # If no bottom modules, connect to state module
                        parents.append(state_module)

        module_to_parents[module] = parents
        leaves -= set(parents)
        leaves.add(module)

    # Create the module graph.
    module_graph = ModuleGraph(
        module_to_parents,
        expert_query_confidence=expert_query_confidence,
        expert_value_rng=None,
    )
    return module_graph


## Logic gate graph where we have:
## (1) AND gates only (not even negations)
## (2) Success nodes will just take logical AND over all inputs;
##     Failure nodes will just produce a False output.
def generate_random_module_graph(
    num_modules: int,
    edge_probability: float,
    query_cost: float,
    rng: np.random.Generator,
    num_incorrect_modules: int = 1,
    correct_module_confidence: float = 1.0,
    incorrect_module_confidence: float = 0.1,
    redundancy: str = "AND",
    expert_query_confidence: float = 1.0,
    query_cost_noise_width_fraction: float = 0.1,
) -> ModuleGraph:
    """Generate a random module graph where all modules are a single type of
    gate. (either AND or OR gates).

    Assumes uniform query cost for all modules.

    Another feature is that the incorrect modules can actually be
    correct with probability incorrect_module_confidence, and the
    correct modules will be correct with probability
    correct_module_confidence.
    """
    assert (
        query_cost > 0
    ), "Input query cost to generate_random_module_graph must be positive."

    # The expert function is the same for all modules.
    def expert(inputs: dict[str, Any]) -> Any:
        """Expert function for the AND gate."""
        if redundancy == "AND":
            out = all(inputs.values())
        elif redundancy == "OR":
            out = any(inputs.values())
        else:
            raise ValueError(f"Invalid redundancy: {redundancy}")
        return out

    def correct_fn(confidence: float, inputs: dict[str, Any]) -> tuple[Any, float]:
        """A function for a correct module."""
        return expert(inputs), confidence

    def incorrect_fn(confidence: float, _inputs: dict[str, Any]) -> tuple[Any, float]:
        """A function for an incorrect module."""
        return False, confidence

    # Determine which modules will be unconfident and incorrect.
    # NOTE: We have to select from the range [1, num_modules], because
    # currently we cannot model the StateModule as incorrect.
    incorrect_module_nums = rng.choice(
        np.arange(1, num_modules), size=num_incorrect_modules, replace=False
    )

    # Spawn an independent rng for seeding the modules with their actual states.
    rng_for_states = rng.spawn(1)[0]

    # And another independent rng for seeding modules with random uniform query costs.
    rng_for_query_costs = rng.spawn(1)[0]
    # query cost will be sampled from [(1-frac)*query_cost, (1+frac)*query_cost]

    # Create the modules.
    modules: list[Module] = []
    for num in range(num_modules):
        if num == 0:
            # Ensure the first module is a StateModule if this is a policy graph.
            module_name = "state"
            module: Module = StateModule()
            modules.append(module)
            continue

        if num == num_modules - 1:
            ParentModuleClass: Type[Module] = ActionModule  # type: ignore
            module_name = "action"
        else:
            ParentModuleClass = Module  # type: ignore
            module_name = f"module_{num}"

        if num in incorrect_module_nums:
            # Module is correct with probability incorrect_module_confidence.
            # Confidence is always 'incorrect_module_confidence' for these modules.
            if rng_for_states.uniform() < incorrect_module_confidence:
                fn = partial(correct_fn, incorrect_module_confidence)
            else:
                fn = partial(incorrect_fn, incorrect_module_confidence)
        else:
            # Confidence is always 'correct_module_confidence' for these modules.
            # Module is correct with probability correct_module_confidence.
            if rng_for_states.uniform() < correct_module_confidence:
                fn = partial(correct_fn, correct_module_confidence)
            else:
                fn = partial(incorrect_fn, correct_module_confidence)

        # sample query cost from [(1-frac)*query_cost, (1+frac)*query_cost]
        query_cost_rand = rng_for_query_costs.uniform(
            (1 - query_cost_noise_width_fraction) * query_cost,
            (1 + query_cost_noise_width_fraction) * query_cost,
        )

        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost_rand,
            expert_fn=expert,
            ParentModuleClass=ParentModuleClass,
        )
        modules.append(module)

    # For all modules except the state module, verify that the query cost is positive.
    for module in modules[1:]:
        assert module.get_expert_query_cost() > 0, (
            f"Module {module.get_name()}: "
            "Expert query cost must be positive"
            f"and equal to fn-provided cost of {query_cost_rand}"
        )

    module_graph = construct_graph(
        modules=modules,
        num_modules=num_modules,
        edge_probability=edge_probability,
        rng=rng,
        expert_query_confidence=expert_query_confidence,
    )

    return module_graph


def generate_random_top_bottom_module_graph(
    num_modules: int,
    edge_probability: float,
    query_cost: float,
    rng: np.random.Generator,
    num_incorrect_modules: int = 1,
    correct_module_confidence: float = 1.0,
    incorrect_module_confidence: float = 0.1,
    gate_top: str = "AND",
    gate_bottom: str = "OR",
    expert_query_confidence: float = 1.0,
    query_cost_noise_width_fraction: float = 0.1,
) -> ModuleGraph:
    """Generate a random module graph where the first group of modules are a
    single gate type ("top"), and the last set of modules are a different gate
    type ("bottom")."""

    # Create the modules.
    # Module 0 is the state module.
    # Modules 1 to num_modules/2 - 1 are the "top" gate type.
    # Modules num_modules/2 to num_modules - 1 are the "bottom" gate type.
    # Module num_modules is the action module.

    def expert(gate_type: str, inputs: dict[str, Any]) -> Any:
        """Expert function for the AND or OR gate."""
        if gate_type == "AND":
            return all(inputs.values())
        if gate_type == "OR":
            return any(inputs.values())
        raise NotImplementedError

    def correct_fn(
        confidence: float, gate_type: str, inputs: dict[str, Any]
    ) -> tuple[Any, float]:
        """A function for a correct module."""
        return expert(gate_type, inputs), confidence

    def incorrect_fn(
        confidence: float, _gate_type: str, _inputs: dict[str, Any]
    ) -> tuple[Any, float]:
        """A function for an incorrect module."""
        return False, confidence

    # Determine which modules will be unconfident and incorrect.
    # NOTE: We have to select from the range [1, num_modules], because
    # currently we cannot model the StateModule as incorrect.
    incorrect_module_nums = rng.choice(
        np.arange(1, num_modules), size=num_incorrect_modules, replace=False
    )

    # Spawn an independent rng for seeding the modules with their actual states.
    rng_for_states = rng.spawn(1)[0]

    # And another independent rng for seeding modules with random uniform query costs.
    rng_for_query_costs = rng.spawn(1)[0]
    # query cost will be sampled from [(1-frac)*query_cost, (1+frac)*query_cost]

    # Create the state module.
    state_module = StateModule()
    # Create the "top" gate type.
    top_modules: list[Module] = []
    for num in range(1, num_modules // 2):
        parent_module_class = Module  # type: ignore
        module_name = f"{gate_top}_module_{num}"
        if num in incorrect_module_nums:
            # Module is correct with probability incorrect_module_confidence.
            # Confidence is always 'incorrect_module_confidence' for these modules.
            if rng_for_states.uniform() < incorrect_module_confidence:
                fn = partial(correct_fn, incorrect_module_confidence, gate_top)
            else:
                fn = partial(incorrect_fn, incorrect_module_confidence, gate_top)
        else:
            # Confidence is always 'correct_module_confidence' for these modules.
            # Module is correct with probability correct_module_confidence.
            if rng_for_states.uniform() < correct_module_confidence:
                fn = partial(correct_fn, correct_module_confidence, gate_top)
            else:
                fn = partial(incorrect_fn, correct_module_confidence, gate_top)

        # sample query cost from [(1-frac)*query_cost, (1+frac)*query_cost]
        query_cost_rand = rng_for_query_costs.uniform(
            (1 - query_cost_noise_width_fraction) * query_cost,
            (1 + query_cost_noise_width_fraction) * query_cost,
        )

        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost_rand,
            expert_fn=partial(expert, gate_top),
            ParentModuleClass=parent_module_class,  # type: ignore
        )
        top_modules.append(module)

    # Create the OR gates.
    bottom_modules: list[Module] = []
    for num in range(num_modules // 2, num_modules - 1):
        parent_module_class = Module
        module_name = f"{gate_bottom}_module_{num}"
        if num in incorrect_module_nums:
            # Module is correct with probability incorrect_module_confidence.
            # Confidence is always 'incorrect_module_confidence' for these modules.
            if rng_for_states.uniform() < incorrect_module_confidence:
                fn = partial(correct_fn, incorrect_module_confidence, gate_bottom)
            else:
                fn = partial(incorrect_fn, incorrect_module_confidence, gate_bottom)
        else:
            # Confidence is always 'correct_module_confidence' for these modules.
            # Module is correct with probability correct_module_confidence.
            if rng_for_states.uniform() < correct_module_confidence:
                fn = partial(correct_fn, correct_module_confidence, gate_bottom)
            else:
                fn = partial(incorrect_fn, correct_module_confidence, gate_bottom)

        # sample query cost from [(1-frac)*query_cost, (1+frac)*query_cost]
        query_cost_rand = rng_for_query_costs.uniform(
            (1 - query_cost_noise_width_fraction) * query_cost,
            (1 + query_cost_noise_width_fraction) * query_cost,
        )

        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost_rand,
            expert_fn=partial(expert, gate_bottom),
            ParentModuleClass=parent_module_class,  # type: ignore
        )
        bottom_modules.append(module)

    # Create the action module.
    if num_modules - 1 in incorrect_module_nums:
        # Module is correct with probability incorrect_module_confidence.
        # Confidence is always 'incorrect_module_confidence' for these modules.
        if rng_for_states.uniform() < incorrect_module_confidence:
            fn = partial(correct_fn, incorrect_module_confidence, gate_bottom)
        else:
            fn = partial(incorrect_fn, incorrect_module_confidence, gate_bottom)
    else:
        # Confidence is always 'correct_module_confidence' for these modules.
        # Module is correct with probability correct_module_confidence.
        if rng_for_states.uniform() < correct_module_confidence:
            fn = partial(correct_fn, correct_module_confidence, gate_bottom)
        else:
            fn = partial(incorrect_fn, correct_module_confidence, gate_bottom)

    # sample query cost from [(1-frac)*query_cost, (1+frac)*query_cost]
    query_cost_rand = rng_for_query_costs.uniform(
        (1 - query_cost_noise_width_fraction) * query_cost,
        (1 + query_cost_noise_width_fraction) * query_cost,
    )

    action_module = create_module(
        name="action",
        fn=fn,
        query_cost=query_cost_rand,
        expert_fn=partial(expert, gate_bottom),
        ParentModuleClass=ActionModule,  # type: ignore
    )

    # Create the module graph.
    module_graph = construct_graph_top_bottom(
        state_module=state_module,
        top_modules=top_modules,
        bottom_modules=bottom_modules,
        action_module=action_module,  # type: ignore
        edge_probability=edge_probability,
        rng=rng,
        expert_query_confidence=expert_query_confidence,
    )
    return module_graph


##### NOT ACTIVELY USED #####


def generate_random_polynomial_module_graph(
    num_modules: int,
    edge_probability: float,
    query_cost: float,
    rng: np.random.Generator,
    num_incorrect_modules: int = 1,
    incorrect_module_confidence: float = 0.1,
) -> ModuleGraph:
    """Generate a random module graph where all modules are polynomial (just
    summing all inputs for now).

    Assumes uniform query cost for all modules.
    """

    def expert_fn(inputs: dict[str, Any]) -> Any:
        """Expert function for the polynomial module."""
        return sum(list(inputs.values()))

    def correct_fn(inputs: dict[str, Any]) -> tuple[Any, float]:
        """A function for a correct module."""
        return expert_fn(inputs), 1.0

    def incorrect_fn(
        rng: np.random.Generator, inputs: dict[str, Any]
    ) -> tuple[Any, float]:
        """A function for an incorrect module.

        Adds uniform integer noise
        """
        return expert_fn(inputs) + rng.integers(-10, 10), incorrect_module_confidence

    # Determine which modules will be unconfident and incorrect.
    incorrect_module_nums = rng.choice(
        num_modules, size=num_incorrect_modules, replace=False
    )

    # Create the modules.
    modules: list[Module] = []
    for num in range(num_modules):
        if num == 0:
            # Ensure the first module is a StateModule.
            module_name = "state"
            module: Module = StateModule()
            modules.append(module)
            continue

        if num == num_modules - 1:
            ParentModuleClass: Type[Module] = ActionModule  # type: ignore
            module_name = "action"
        else:
            ParentModuleClass = Module  # type: ignore
            module_name = f"module_{num}"

        if num in incorrect_module_nums:
            fn = partial(incorrect_fn, rng)
        else:
            fn = partial(correct_fn)

        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost,
            expert_fn=expert_fn,
            ParentModuleClass=ParentModuleClass,
        )
        modules.append(module)

    module_graph = construct_graph(
        modules=modules,
        num_modules=num_modules,
        edge_probability=edge_probability,
        rng=rng,
    )
    return module_graph


def generate_random_logic_gate_module_graph(
    num_modules: int,
    edge_probability: float,
    query_cost_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    is_policy: bool = False,
    num_incorrect_modules: int = 1,
    incorrect_module_confidence: float = 0.1,
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

    # Determine which modules will be unconfident and incorrect.
    incorrect_module_nums = rng.choice(
        num_modules, size=num_incorrect_modules, replace=False
    )

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

        and_or_or = rng.choice(["and", "or"])
        not_out = rng.choice([True, False])
        expert_fn = partial(expert, and_or_or, not_out)
        if num in incorrect_module_nums:
            fn = partial(incorrect_fn, and_or_or, not_out, incorrect_module_confidence)
        else:
            fn = partial(correct_fn, and_or_or, not_out, 1.0)
        query_cost = query_cost_sampler(rng)
        module = create_module(
            name=module_name,
            fn=fn,
            query_cost=query_cost,
            expert_fn=expert_fn,
            ParentModuleClass=ParentModuleClass,
        )
        modules.append(module)

    module_graph = construct_graph(
        modules=modules,
        num_modules=num_modules,
        edge_probability=edge_probability,
        rng=rng,
    )
    return module_graph


#### HELPFUL UTILITIES ####


def task_cost_proxy(
    computed_confidences: dict[Module, float],
    module_name_to_module: dict[str, Module],
    nonquery_and_set: set[str],
    nonquery_or_set: set[str],
) -> float:
    """Estimate the task cost of a given module graph.

    This is a proxy for the task cost, and is used to estimate the
    expected cost of querying a module graph.
    """
    # AND proxy.
    # Compute probability of being correct, assuming that a module confidence
    # is exactly the probability that it is correct.
    probability_of_correct_answer = 1.0
    for module_name in nonquery_and_set:
        module = module_name_to_module[module_name]
        probability_of_correct_answer *= computed_confidences[module]
    and_proxy = 1 - probability_of_correct_answer

    # OR proxy.
    or_proxy = 0.0
    for module_name in nonquery_or_set:
        module = module_name_to_module[module_name]
        or_proxy += 1 - computed_confidences[module]

    return and_proxy + or_proxy


def get_query_set_expected_cost(
    query_set: set[str],
    module_graph: ModuleGraph,
    computed_confidences: dict[Module, float],
    and_modules: set[str],
    or_modules: set[str],
) -> float:
    """Calculate the total cost of querying the modules in the given query
    set."""

    # Compute query cost.
    module_name_to_module = {m.get_name(): m for m in module_graph.get_modules()}
    total_query_cost = 0.0
    for module_name in query_set:
        module = module_name_to_module[module_name]
        total_query_cost += module.get_expert_query_cost()

    nonquery_set = set(module_name_to_module) - query_set

    # Partition the nonquery set into AND and OR modules.
    nonquery_and_set = nonquery_set & and_modules
    nonquery_or_set = nonquery_set & or_modules

    # Compute total cost.
    combined_cost = total_query_cost + task_cost_proxy(
        computed_confidences,
        module_name_to_module,
        nonquery_and_set,
        nonquery_or_set,
    )

    return combined_cost


def compare_query_set_expected_costs(
    query_set_1: set[str],
    query_set_2: set[str],
    module_graph: ModuleGraph,
    computed_confidences: dict[Module, float],
    tolerance: float = 1e-6,
) -> int:
    """(assumes the product-of-confidences proxy objective)

    Compare the expected costs of two query sets.

    Implemented in a numerically-stable manner (to handle cases where
    the probability of correct answer would underflow).

    Assume that correct_answer_cost and incorrect_answer_cost are 0.0
    and 1.0, respectively.

    Return 1 if query_set_1 is better than query_set_2, -1 if
    query_set_2 is better than query_set_1, and 0 if they are the same.

    Make sure that it's a total pre-order (reflexive, transitive, and
    total)
    """
    # Edge case: if one query set is empty, return the other query set.
    if not query_set_1:
        return -1
    if not query_set_2:
        return 1

    module_name_to_module = {m.get_name(): m for m in module_graph.get_modules()}
    # Compute the query costs for each query option.
    query_cost_1 = sum(
        module_name_to_module[module_name].get_expert_query_cost()
        for module_name in query_set_1
    )
    query_cost_2 = sum(
        module_name_to_module[module_name].get_expert_query_cost()
        for module_name in query_set_2
    )

    # Compute approximate probability of correct answer for each query option.
    nonquery_set_1 = set(module_name_to_module) - query_set_1
    nonquery_set_2 = set(module_name_to_module) - query_set_2

    # Compute the sum of logs of the confidences for each query option.
    log_confidence_1 = sum(
        np.log(computed_confidences[module_name_to_module[module_name]])
        for module_name in nonquery_set_1
    )
    log_confidence_2 = sum(
        np.log(computed_confidences[module_name_to_module[module_name]])
        for module_name in nonquery_set_2
    )

    # Case-by-case analysis based on query_cost_i and log_confidence_i values.
    if query_cost_1 > query_cost_2 and log_confidence_1 < log_confidence_2:
        # Case 1: query_set_1 is worse than query_set_2.
        return -1
    if query_cost_1 < query_cost_2 and log_confidence_1 > log_confidence_2:
        # Case 2: query_set_2 is worse than query_set_1.
        return 1

    # If the absolute values of both log-confidences are smaller than the tolerance,
    # we should compare the query costs directly.
    if log_confidence_1 < np.log(tolerance) and log_confidence_2 < np.log(tolerance):
        return 1 if query_cost_1 < query_cost_2 else -1

    # Otherwise we should compare the raw objective values (without taking logs).
    total_cost_1 = query_cost_1 + (1 - np.exp(log_confidence_1))
    total_cost_2 = query_cost_2 + (1 - np.exp(log_confidence_2))
    return 1 if total_cost_1 < total_cost_2 else -1
