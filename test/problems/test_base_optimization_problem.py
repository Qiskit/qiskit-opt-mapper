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
"""Tests specific to BaseOptimizationProblem (abstractness and hierarchy)."""

import warnings

import pytest
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.problems import ObjSense, OptimizationProblemElement
from qiskit_addon_opt_mapper.problems.base_optimization_problem import BaseOptimizationProblem


class _MissingHooks(BaseOptimizationProblem):
    """Subclass that implements none of the abstract objective hooks."""


class _MissingStrHook(BaseOptimizationProblem):
    """Subclass that implements all abstract objective hooks except ``_objectives_str``."""

    def _reset_objectives(self) -> None:
        pass

    def _objectives_repr(self) -> str:
        return ""

    def _append_or_set_objective_from_parts(self, sense, **kwargs) -> None:
        pass


class _MissingFromPartsHook(BaseOptimizationProblem):
    """Subclass that implements all abstract objective hooks except the reconstruction hook."""

    def _reset_objectives(self) -> None:
        pass

    def _objectives_repr(self) -> str:
        return ""

    def _objectives_str(self) -> str:
        return ""


class _SubProblem(OptimizationProblem):
    """Concrete subclass used to check that the class is preserved."""


def test_base_is_abstract():
    """BaseOptimizationProblem cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseOptimizationProblem("p")


@pytest.mark.parametrize("subclass", [_MissingHooks, _MissingStrHook, _MissingFromPartsHook])
def test_subclass_missing_hooks_is_abstract(subclass):
    """A subclass that does not implement every objective hook cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        subclass("p")


def test_is_subclass_of_base():
    """OptimizationProblem is a subclass of BaseOptimizationProblem."""
    assert issubclass(OptimizationProblem, BaseOptimizationProblem)


def test_repr_of_empty_problem():
    """The empty-problem repr is byte-identical to the one before the base class existed."""
    assert (
        repr(OptimizationProblem("p"))
        == "<OptimizationProblem: minimize 0, 0 variables, 0 constraints, 'p'>"
    )


def test_element_accepts_optimization_problem_parent():
    """An element can be created with, and re-parented to, an OptimizationProblem."""
    element = OptimizationProblemElement(OptimizationProblem("p"))
    new_parent = OptimizationProblem("q")
    element.optimization_problem = new_parent
    assert element.optimization_problem is new_parent


def test_element_rejects_non_problem_parent():
    """Elements reject parents that are not a BaseOptimizationProblem, in init and setter."""
    with pytest.raises(TypeError, match="BaseOptimizationProblem instance expected"):
        OptimizationProblemElement("not a problem")
    element = OptimizationProblemElement(OptimizationProblem("p"))
    with pytest.raises(TypeError, match="BaseOptimizationProblem instance expected"):
        element.optimization_problem = "not a problem"


def test_new_empty_keeps_class_and_name_but_not_content():
    """`_new_empty` returns a problem of the same class and name without variables or objective."""
    problem = _SubProblem("p")
    problem.binary_var("x")
    problem.minimize(linear={"x": 1}, name="cost")
    empty = problem._new_empty()
    assert type(empty) is _SubProblem
    assert empty.name == "p"
    assert empty.get_num_vars() == 0
    assert not empty.objective.name


def test_substitute_variables_keeps_problem_class():
    """`substitute_variables` returns a problem of the same class as its input."""
    problem = _SubProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.minimize(linear={"x": 1, "y": 2})
    assert type(problem.substitute_variables({"x": 1})) is _SubProblem


def test_from_parts_hook_does_not_check_name():
    """The hook skips the name check, and `minimize` and `maximize` check once before it."""
    problem = OptimizationProblem()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem._append_or_set_objective_from_parts(ObjSense.MINIMIZE, name="bad\nname")
    assert not caught
    for method in (problem.minimize, problem.maximize):
        with pytest.warns(UserWarning, match="Objective name is not printable") as record:
            method(name="bad\nname")
        assert len(record) == 1


def test_from_parts_hook_sets_objective():
    """The hook replaces the objective of a single-objective problem."""
    problem = OptimizationProblem()
    problem.binary_var("x")
    problem._append_or_set_objective_from_parts(
        ObjSense.MAXIMIZE, constant=1.0, linear={"x": 2}, name="cost"
    )
    objective = problem.objective
    assert objective.sense == ObjSense.MAXIMIZE
    assert objective.constant == 1.0
    assert objective.linear.to_dict(use_name=True) == {"x": 2}
    assert objective.name == "cost"
