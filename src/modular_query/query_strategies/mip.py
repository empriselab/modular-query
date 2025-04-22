"""A query strategy that formulates subset selection as a mixed integer
program."""

from typing import Any

from amplpy import modules
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
from modular_query.modules import Module
from modular_query.query_strategies.base import QueryStrategy


class MIPQueryStrategy(QueryStrategy):
    """A query strategy that formulates subset selection as an MIP."""

    def get_expert_query_module(
        self,
        module_graph: ModuleGraph,
        computed_values: dict[Module, Any],
        computed_confidences: dict[Module, float],
    ) -> str | None:

        # Extract module info.
        all_module_names = sorted(self.get_all_queryable_modules(module_graph))
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

        # Constraint: only select at most one module.
        def one_module_rule(m):
            return sum(m.x[i] for i in m.I) <= 1

        model.one_module_constraint = Constraint(rule=one_module_rule)

        # Constraint: define y.
        def product_constraint_rule(m):
            prod_terms = []
            for i in m.I:
                prod_terms.append(probs_correct[i] + (1 - probs_correct[i]) * m.x[i])

            # We can multiply them with Pyomo's built-in ProductExpression:
            #   product_expr = prod_terms[0] * prod_terms[1] * ...
            # or we can do a quick reduce():
            product_expr = prod_terms[0]
            for term in prod_terms[1:]:
                product_expr = product_expr * term

            # Enforce m.y == product
            return m.y - product_expr == 0

        model.product_constraint = Constraint(rule=product_constraint_rule)

        # Run optimization.
        # pip install amplpy pyomo -q
        # python -m amplpy.modules install coin highs scip gcg -q

        try:
            solver = SolverFactory(
                "bonminnl", executable=modules.find("bonmin"), solve_io="nl"
            )
        except:  # pylint: disable=bare-except
            print("Failed to load bonmin solver.")
            print("Please install it:")
            print("python -m amplpy.modules install coin -q")
        solver.solve(model)
        query_mask = [value(model.x[i]) >= 0.5 for i in model.I]

        # Return selected module name, or None if none selected.
        if sum(query_mask) == 0:
            return None
        return all_module_names[query_mask.index(True)]
