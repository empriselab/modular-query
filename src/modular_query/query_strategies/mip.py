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
    ) -> tuple[str | None, dict[str, float] | None]:
        with timer(
            "MIPQueryStrategy: Construct problem",
            verbose=False,
        ) as result:  # type: dict[str, float]
            # Extract module info (use topological order).
            all_module_names = [
                module.get_name()
                for module in module_graph.topo_order
                if not isinstance(module, StateModule)
                and module.get_name() not in self.queried_modules
            ]
            module_name_to_module = {
                module.get_name(): module for module in module_graph.get_modules()
            }
            all_modules = [module_name_to_module[n] for n in all_module_names]
            query_costs = [m.get_expert_query_cost() for m in all_modules]
            probs_correct = [computed_confidences[m] for m in all_modules]

            model = ConcreteModel()

            # Create binary variables for each module that we might query.
            model.I = range(len(all_module_names))
            model.x = Var(model.I, domain=Binary)

            # Create one continuous variable that should be equal to the probability
            # of "all correct".
            model.y = Var(domain=Reals, bounds=(0, 1))

            # Objective: minimize the total expected cost.
            def obj_expression(m):
                expert_cost_sum = sum(query_costs[i] * m.x[i] for i in m.I)
                task_cost = (
                    self.incorrect_answer_cost
                    + (self.correct_answer_cost - self.incorrect_answer_cost) * m.y
                )
                return expert_cost_sum + task_cost

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

            # Constraint: define y.
            def product_constraint_rule(m):
                prod_terms = []
                for i in m.I:
                    prod_terms.append(
                        probs_correct[i] + (1 - probs_correct[i]) * m.x[i]
                    )

                # We can multiply them with Pyomo's built-in ProductExpression:
                #   product_expr = prod_terms[0] * prod_terms[1] * ...
                # or we can do a quick reduce():
                product_expr = 1
                for term in prod_terms:
                    product_expr = product_expr * term

                # Enforce m.y == product
                return m.y - product_expr == 0

            model.product_constraint = Constraint(rule=product_constraint_rule)

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
            return None, timing_info

        return all_module_names[query_mask.index(True)], timing_info
