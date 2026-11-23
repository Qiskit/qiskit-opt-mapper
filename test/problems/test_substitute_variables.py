# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for objective-name handling in ``substitute_variables``."""

import pytest
from qiskit_addon_opt_mapper import MultiObjectiveProblem, OptimizationProblem
from qiskit_addon_opt_mapper.problems import ObjSense


@pytest.fixture(name="named_problem", params=["minimize", "maximize"])
def fixture_named_problem(request) -> OptimizationProblem:
    """Problem with a named objective for each sense."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    getattr(problem, request.param)(linear={"x": 1, "y": 2}, quadratic={("x", "y"): 3}, name="f0")
    return problem


def test_constants_substitution_preserves_name_and_sense(named_problem):
    """Substituting a variable by a constant keeps the objective name and sense."""
    result = named_problem.substitute_variables(constants={"x": 1})
    assert result.objective.name == "f0"
    assert result.objective.sense == named_problem.objective.sense


def test_variables_substitution_preserves_name_and_sense(named_problem):
    """Substituting a variable by another variable keeps the objective name and sense."""
    result = named_problem.substitute_variables(variables={"x": ("y", 2)})
    assert result.objective.name == "f0"
    assert result.objective.sense == named_problem.objective.sense


def test_substitution_keeps_coefficients_of_named_objective(named_problem):
    """The named objective is substituted with the same coefficients as an unnamed one."""
    unnamed = OptimizationProblem("p")
    unnamed.binary_var("x")
    unnamed.binary_var("y")
    maker = (
        unnamed.minimize if named_problem.objective.sense == ObjSense.MINIMIZE else unnamed.maximize
    )
    maker(linear={"x": 1, "y": 2}, quadratic={("x", "y"): 3})

    named_result = named_problem.substitute_variables(constants={"x": 1})
    unnamed_result = unnamed.substitute_variables(constants={"x": 1})

    assert named_result.objective.constant == unnamed_result.objective.constant
    assert named_result.objective.linear.to_dict() == unnamed_result.objective.linear.to_dict()
    assert (
        named_result.objective.quadratic.to_dict() == unnamed_result.objective.quadratic.to_dict()
    )


@pytest.mark.parametrize("method", ["minimize", "maximize"])
def test_unnamed_objective_stays_unnamed(method):
    """Substitution on an unnamed objective yields an unnamed objective."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    getattr(problem, method)(linear={"x": 1, "y": 2})
    result = problem.substitute_variables(constants={"x": 1})
    assert result.objective.name == ""


def test_substitution_keeps_problem_name():
    """The substituted problem keeps the name of the source problem."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.minimize(linear={"x": 1}, name="f0")
    assert problem.substitute_variables(constants={"x": 1}).name == "p"


def _multi_objective_problem() -> MultiObjectiveProblem:
    problem = MultiObjectiveProblem("m")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.add_objective(constant=1, linear={"x": 2}, name="a")
    problem.add_objective(linear={"y": 1}, quadratic={("x", "y"): 3}, sense="max", name="b")
    return problem


def test_multi_objective_substitution_preserves_names_order_and_senses():
    """Every objective keeps its name, position and sense after substitution."""
    result = _multi_objective_problem().substitute_variables(constants={"x": 1})
    assert isinstance(result, MultiObjectiveProblem)
    assert [o.name for o in result.objectives] == ["a", "b"]
    assert [o.sense for o in result.objectives] == [ObjSense.MINIMIZE, ObjSense.MAXIMIZE]


def test_multi_objective_substitution_folds_constants_per_objective():
    """Each objective is substituted independently."""
    result = _multi_objective_problem().substitute_variables(constants={"x": 1})
    first, second = result.objectives
    assert first.constant == 3
    assert first.linear.to_dict() == {}
    assert second.linear.to_dict(use_name=True) == {"y": 4}


def test_substitution_retains_vanished_objective_as_constant():
    """An objective that loses all variables is kept, a vanished constraint is dropped."""
    problem = _multi_objective_problem()
    problem.linear_constraint({"x": 1}, "<=", 1, "gone")
    problem.linear_constraint({"y": 1}, "<=", 1, "kept")
    result = problem.substitute_variables(constants={"x": 1, "y": 1})
    assert result.get_num_objectives() == 2
    assert [o.constant for o in result.objectives] == [3, 4]
    assert result.linear_constraints_index == {}
    assert result.status == MultiObjectiveProblem.Status.VALID
