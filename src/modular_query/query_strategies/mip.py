"""A query strategy that formulates subset selection as a mixed integer
program."""

from typing import Any

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Reals,
    SolverFactory,
    Var,
    value,
)

from modular_query.module_graph import ModuleGraph
from modular_query.modules import Module, StateModule
from modular_query.query_strategies.base import QueryStrategy
from modular_query.utils import timer


class MIPQueryStrategy(QueryStrategy):
    """A query strategy that formulates subset selection as an MIP."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> tuple[str | None, dict[str, float] | None, dict[str, Any]]:
        with timer(
            "MIPQueryStrategy: Construct problem",
            verbose=False,
        ) as result:  # type: dict[str, float]
            # Extract module info (use topological order).
            all_module_names = [
                module.get_name()
                for module in module_graph.topo_order
                if not isinstance(module, StateModule)
            ]
            module_name_to_module = {
                module.get_name(): module for module in module_graph.get_modules()
            }
            all_modules = [module_name_to_module[n] for n in all_module_names]
            query_costs = [m.get_expert_query_cost() for m in all_modules]

            model = ConcreteModel()

            # Create binary variables for each module that we might query.
            model.I = range(len(all_module_names))
            model.x = Var(model.I, domain=Binary)

            # Create one continuous variable
            # that should be equal to the proxy of task cost.
            # (no longer bounded in [0, 1]
            # because we are now using a proxy of task cost.)
            model.y = Var(domain=Reals)

            # Objective: minimize the total expected cost.
            def obj_expression(m):
                expert_cost_sum = sum(query_costs[i] * m.x[i] for i in m.I)
                proxy_task_cost = m.y
                return expert_cost_sum + proxy_task_cost

            model.obj = Objective(rule=obj_expression)

            # Constraint: select exactly one module (i.e. force a query.)
            def one_module_rule(m):
                # Handle special case where there are no modules to query.
                if len(m.I) == 0:
                    return Constraint.Feasible
                return sum(m.x[i] for i in m.I) == 1

            try:
                model.one_module_constraint = Constraint(rule=one_module_rule)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"model.I: {model.I}")
                print(f"model.x: {model.x}")
                print(f"model.y: {model.y}")
                print(f"model.one_module_constraint: {model.one_module_constraint}")
                raise e

            # Create auxiliary variables for AND and OR proxies.
            model.and_proxy = Var(domain=Reals)
            model.or_proxy = Var(domain=Reals)

            # Constraint: y = and_proxy + or_proxy.
            def y_constraint_rule(m):
                return m.y == m.and_proxy + m.or_proxy

            model.y_constraint = Constraint(rule=y_constraint_rule)

            # For AND modules.
            and_module_indices = [
                i for i, name in enumerate(all_module_names) if name in self.and_modules
            ]

            def and_proxy_rule(m):
                and_terms = []
                for i in and_module_indices:
                    module_name = all_module_names[i]
                    module = module_name_to_module[module_name]
                    confidence = computed_confidences[module]
                    # If x[i] = 0 (not queried), multiply by (1 - confidence).
                    # If x[i] = 1 (queried), multiply by 1.
                    and_terms.append(confidence + (1 - confidence) * m.x[i])

                product_expr = 1
                for term in and_terms:
                    product_expr *= term
                and_proxy = 1 - product_expr

                return m.and_proxy == and_proxy

            model.and_proxy_constraint = Constraint(rule=and_proxy_rule)

            # For OR modules.
            or_module_indices = [
                i for i, name in enumerate(all_module_names) if name in self.or_modules
            ]

            def or_proxy_rule(m):
                or_terms = []
                for i in or_module_indices:
                    module_name = all_module_names[i]
                    module = module_name_to_module[module_name]
                    confidence = computed_confidences[module]
                    # If x[i] = 0 (not queried), add (1 - confidence)
                    # If x[i] = 1 (queried), add 0
                    or_terms.append((1 - m.x[i]) * (1 - confidence))
                or_proxy = sum(or_terms)

                return m.or_proxy == or_proxy

            model.or_proxy_constraint = Constraint(rule=or_proxy_rule)

        t_construct_problem = result["time"]

        with timer("MIPQueryStrategy: Solve problem", verbose=False) as result:
            # Run optimization.
            # pip install amplpy pyomo -q
            # python -m amplpy.modules install coin highs scip gcg -q

            try:
                solver = SolverFactory(
                    "scip",
                    solve_io="nl",
                )
            except:  # pylint: disable=bare-except
                print("Failed to load solver.")
                print("Please install it:")
                print("python -m amplpy.modules install coin -q")
            solver.solve(model)
            query_mask = [value(model.x[i]) >= 0.5 for i in model.I]
        t_solve_problem = result["time"]

        timing_info = {
            "t_construct_problem": t_construct_problem,
            "t_solve_problem": t_solve_problem,
        }

        # Return selected module name, or None if none selected.
        if sum(query_mask) == 0:
            return None, timing_info, {}

        return all_module_names[query_mask.index(True)], timing_info, {}
