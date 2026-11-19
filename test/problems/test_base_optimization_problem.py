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

from collections.abc import Iterator

import pytest
from qiskit_addon_opt_mapper import MultiObjectiveProblem, OptimizationError, OptimizationProblem
from qiskit_addon_opt_mapper.problems import (
    ObjSense,
    OptimizationObjective,
    OptimizationProblemElement,
)
from qiskit_addon_opt_mapper.problems.base_optimization_problem import BaseOptimizationProblem


class _MissingHooks(BaseOptimizationProblem):
    """Subclass that implements none of the abstract objective hooks."""


class _MissingStrHook(BaseOptimizationProblem):
    """Subclass that implements all abstract objective hooks except ``_objectives_str``."""

    def _reset_objectives(self) -> None:
        pass

    @property
    def is_multi_objective(self) -> bool:
        return False

    def iter_objectives(self) -> Iterator[OptimizationObjective]:
        return iter(())

    def _objectives_repr(self) -> str:
        return ""

    def _append_or_set_objective_from_parts(self, sense, **kwargs) -> None:
        pass


def test_base_is_abstract():
    """BaseOptimizationProblem cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseOptimizationProblem("p")


@pytest.mark.parametrize("subclass", [_MissingHooks, _MissingStrHook])
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


class _MissingSetObjectiveHook(BaseOptimizationProblem):
    """Subclass that implements every hook except ``_append_or_set_objective_from_parts``."""

    def _reset_objectives(self) -> None:
        pass

    @property
    def is_multi_objective(self) -> bool:
        return False

    def iter_objectives(self) -> Iterator[OptimizationObjective]:
        return iter(())

    def _objectives_repr(self) -> str:
        return ""

    def _objectives_str(self) -> str:
        return ""


class _MissingIsMultiObjectiveHook(BaseOptimizationProblem):
    """Subclass that implements every hook except ``is_multi_objective``."""

    def _reset_objectives(self) -> None:
        pass

    def iter_objectives(self) -> Iterator[OptimizationObjective]:
        return iter(())

    def _objectives_repr(self) -> str:
        return ""

    def _objectives_str(self) -> str:
        return ""

    def _append_or_set_objective_from_parts(self, sense, **kwargs) -> None:
        pass


class _MissingIterObjectivesHook(BaseOptimizationProblem):
    """Subclass that implements every hook except ``iter_objectives``."""

    def _reset_objectives(self) -> None:
        pass

    @property
    def is_multi_objective(self) -> bool:
        return False

    def _objectives_repr(self) -> str:
        return ""

    def _objectives_str(self) -> str:
        return ""

    def _append_or_set_objective_from_parts(self, sense, **kwargs) -> None:
        pass


def test_missing_set_objective_hook_is_abstract():
    """``_append_or_set_objective_from_parts`` is an abstract hook."""
    with pytest.raises(TypeError, match="abstract"):
        _MissingSetObjectiveHook("p")


@pytest.mark.parametrize(
    ("subclass", "hook"),
    [
        (_MissingIsMultiObjectiveHook, "is_multi_objective"),
        (_MissingIterObjectivesHook, "iter_objectives"),
    ],
)
def test_missing_single_new_hook_is_abstract(subclass, hook):
    """``is_multi_objective`` and ``iter_objectives`` are abstract hooks."""
    with pytest.raises(TypeError, match=hook):
        subclass("p")


def test_new_empty_preserves_class_and_name():
    """``_new_empty`` creates a fresh problem of the same class with the same name."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.minimize(linear={"x": 1}, name="f0")
    fresh = problem._new_empty()
    assert type(fresh) is OptimizationProblem
    assert fresh is not problem
    assert fresh.name == "p"
    assert fresh.get_num_vars() == 0
    assert fresh.objective.name == ""


@pytest.mark.parametrize("sense", [ObjSense.MINIMIZE, ObjSense.MAXIMIZE])
@pytest.mark.parametrize("name", ["", "f0"])
def test_set_objective_from_parts(sense, name):
    """``_append_or_set_objective_from_parts`` forwards sense, all parts and the name."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    problem._append_or_set_objective_from_parts(
        sense,
        constant=3.0,
        linear={"x": 1},
        quadratic={("x", "y"): 2},
        higher_order={3: {("x", "y", "x"): 4}},
        name=name,
    )
    objective = problem.objective
    assert objective.sense == sense
    assert objective.name == name
    assert objective.constant == 3.0
    assert objective.linear.to_dict(use_name=True) == {"x": 1}
    assert objective.quadratic.to_dict(use_name=True) == {("x", "y"): 2}
    assert objective.higher_order[3].to_dict(use_name=True) == {("x", "x", "y"): 4}


# ---------------------------------------------------------------------------
# Shared behavior, run against every concrete subclass.
# ---------------------------------------------------------------------------


def _add_objective(problem: BaseOptimizationProblem, **kwargs) -> None:
    """Add an objective to either subclass (``minimize`` or ``add_objective``)."""
    if problem.is_multi_objective:
        problem.add_objective(**kwargs)
    else:
        problem.minimize(**kwargs)


@pytest.fixture(name="problem", params=[OptimizationProblem, MultiObjectiveProblem])
def fixture_problem(request) -> BaseOptimizationProblem:
    """An empty problem of each concrete subclass."""
    return request.param("p")


def test_constructor_defaults(problem):
    """A new problem has a name, a valid status and nothing else."""
    assert problem.name == "p"
    assert problem.status == BaseOptimizationProblem.Status.VALID
    assert problem.get_num_vars() == 0
    assert problem.get_num_linear_constraints() == 0
    assert problem.get_num_quadratic_constraints() == 0
    assert problem.get_num_higher_order_constraints() == 0


def test_name_setter_warns_on_non_printable_name(problem):
    """A non-printable problem name is accepted with a warning."""
    with pytest.warns(UserWarning, match="Problem name is not printable"):
        problem.name = "a\nb"
    assert problem.name == "a\nb"


def test_variable_types_single_form(problem):
    """All four variable types can be created and are counted."""
    problem.binary_var("b")
    problem.integer_var(0, 5, "i")
    problem.continuous_var(0, 1, "c")
    problem.spin_var("s")
    assert problem.get_num_vars() == 4
    assert (
        problem.get_num_binary_vars(),
        problem.get_num_integer_vars(),
        problem.get_num_continuous_vars(),
        problem.get_num_spin_vars(),
    ) == (1, 1, 1, 1)
    assert [v.name for v in problem.variables] == ["b", "i", "c", "s"]
    assert problem.variables_index == {"b": 0, "i": 1, "c": 2, "s": 3}


@pytest.mark.parametrize("kind", ["binary", "integer", "continuous", "spin"])
def test_variable_list_and_dict_forms(problem, kind):
    """List and dict creators produce the expected names."""
    args = {"integer": (0, 3), "continuous": (0, 1)}.get(kind, ())
    as_list = getattr(problem, f"{kind}_var_list")(2, *args, name="a")
    as_dict = getattr(problem, f"{kind}_var_dict")(["u", "v"], *args, name="d")
    assert [v.name for v in as_list] == ["a0", "a1"]
    assert list(as_dict) == ["du", "dv"]
    assert problem.get_num_vars() == 4


def test_duplicate_variable_name_raises(problem):
    """Creating two variables with the same name raises."""
    problem.binary_var("x")
    with pytest.raises(OptimizationError, match="Variable name already exists: x"):
        problem.binary_var("x")


@pytest.mark.parametrize("keys", [0, -1])
def test_non_positive_keys_raise(problem, keys):
    """A non-positive number of keys raises."""
    with pytest.raises(OptimizationError, match="non-positive number of variables"):
        problem.binary_var_list(keys)
    with pytest.raises(OptimizationError, match="non-positive number of variables"):
        problem.binary_var_dict(keys)


def test_invalid_key_format_raises(problem):
    """A key format with two substitutions raises."""
    with pytest.raises(OptimizationError, match="more than one substitution"):
        problem.binary_var_list(2, key_format="{}{}")


def test_get_variable_by_index_and_name(problem):
    """Variables are retrievable by index and by name, with the standard errors."""
    x = problem.binary_var("x")
    assert problem.get_variable(0) is x
    assert problem.get_variable("x") is x
    with pytest.raises(KeyError):
        problem.get_variable("zz")
    with pytest.raises(IndexError):
        problem.get_variable(9)


@pytest.mark.parametrize(
    ("kind", "prefix"),
    [("linear", "c"), ("quadratic", "q"), ("higher_order", "h")],
)
def test_constraint_families(problem, kind, prefix):
    """Add, get, count, remove and reindex-after-remove work for every constraint family."""
    problem.binary_var("x")
    add = getattr(problem, f"{kind}_constraint")
    get = getattr(problem, f"get_{kind}_constraint")
    count = getattr(problem, f"get_num_{kind}_constraints")
    remove = getattr(problem, f"remove_{kind}_constraint")
    index = lambda: getattr(problem, f"{kind}_constraints_index")  # noqa: E731

    extra = {"higher_order": {3: {("x", "x", "x"): 1}}} if kind == "higher_order" else {}
    first, second, third = add(**extra), add(**extra), add(**extra)
    assert [first.name, second.name, third.name] == [f"{prefix}{i}" for i in range(3)]
    assert count() == 3
    assert get(f"{prefix}1") is second
    assert get(2) is third

    remove(f"{prefix}0")
    assert count() == 2
    assert index() == {f"{prefix}1": 0, f"{prefix}2": 1}
    assert get(0) is second

    remove(0)
    assert index() == {f"{prefix}2": 0}


def test_duplicate_constraint_name_raises(problem):
    """A duplicate constraint name raises."""
    problem.linear_constraint(name="a")
    with pytest.raises(OptimizationError, match="Linear constraint's name already exists: a"):
        problem.linear_constraint(name="a")


def test_feasibility(problem):
    """``is_feasible`` and ``get_feasibility_info`` behave the same on both subclasses."""
    problem.continuous_var(-1, 1, "x")
    problem.continuous_var(-10, 10, "y")
    problem.linear_constraint({"x": 1, "y": 1}, "<=", 10, "c0")
    problem.linear_constraint({"x": 1, "y": 1}, "==", 5, "c1")
    problem.quadratic_constraint({"y": 1}, {("x", "x"): 1}, "<=", 10, "q0")

    assert problem.is_feasible([0, 5])
    assert not problem.is_feasible([1, 10])

    feasible, variables, constraints = problem.get_feasibility_info([10, 0])
    assert not feasible
    assert [v.name for v in variables] == ["x"]
    assert [c.name for c in constraints] == ["c1", "q0"]


def test_feasibility_length_mismatch_raises(problem):
    """A solution of the wrong length raises OptimizationError."""
    problem.binary_var("x")
    with pytest.raises(OptimizationError, match="does not match the number of problem variables"):
        problem.is_feasible([1, 2])


def test_clear_resets_everything(problem):
    """``clear`` removes variables, constraints and objectives, and the name."""
    problem.binary_var("x")
    problem.binary_var("y")
    _add_objective(problem, linear={"x": 1})
    problem.linear_constraint({"x": 1}, "==", 1)
    problem.quadratic_constraint({"x": 1}, {("y", "y"): 2}, "<=", 1)
    problem.clear()

    assert problem.name == ""
    assert problem.status == BaseOptimizationProblem.Status.VALID
    assert problem.get_num_vars() == 0
    assert problem.get_num_linear_constraints() == 0
    assert problem.get_num_quadratic_constraints() == 0
    assert problem.get_num_higher_order_constraints() == 0
    objectives = list(problem.iter_objectives())
    assert len(objectives) == (0 if problem.is_multi_objective else 1)
    assert all(
        obj.constant == 0 and not obj.linear.to_dict() and not obj.quadratic.to_dict()
        for obj in objectives
    )


def test_iter_objectives_matches_subclass(problem):
    """``iter_objectives`` yields one objective for single, all objectives for multi."""
    _add_objective(problem, constant=1)
    if problem.is_multi_objective:
        _add_objective(problem, constant=2)
    objectives = list(problem.iter_objectives())
    assert len(objectives) == (2 if problem.is_multi_objective else 1)
    assert objectives[0].constant == 1


def test_new_empty_keeps_class_and_name(problem):
    """``_new_empty`` returns an empty problem of the same class and name."""
    problem.binary_var("x")
    fresh = problem._new_empty()  # pylint: disable=protected-access
    assert type(fresh) is type(problem)
    assert fresh.name == "p"
    assert fresh.get_num_vars() == 0


def test_copy_from_reparents_variables_constraints_and_objectives(problem):
    """``_copy_from`` re-parents variables, constraints and objectives to the target."""
    problem.binary_var("x")
    problem.binary_var("y")
    _add_objective(problem, linear={"x": 1}, name="o")
    problem.linear_constraint({"x": 1}, "<=", 1, "lc")
    problem.quadratic_constraint({"x": 1}, {("y", "y"): 1}, "<=", 1, "qc")
    problem.higher_order_constraint({"x": 1}, None, {3: {("x", "x", "x"): 1}}, "<=", 1, "hc")

    target = type(problem)()
    target._copy_from(problem, include_name=True)  # pylint: disable=protected-access

    assert target.name == "p"
    assert all(v.optimization_problem is target for v in target.variables)
    assert all(c.optimization_problem is target for c in target.linear_constraints)
    assert all(c.optimization_problem is target for c in target.quadratic_constraints)
    assert all(c.optimization_problem is target for c in target.higher_order_constraints)
    for obj in target.iter_objectives():
        assert obj.optimization_problem is target


def test_substitute_variables_keeps_class_and_status(problem):
    """``substitute_variables`` returns the same class with a valid status."""
    problem.binary_var("x")
    problem.binary_var("y")
    _add_objective(problem, linear={"x": 1, "y": 2}, name="o")
    result = problem.substitute_variables(constants={"x": 1})
    assert type(result) is type(problem)
    assert result.status == BaseOptimizationProblem.Status.VALID
    assert result.name == "p"


def test_prettyprint_is_available_on_both(problem):
    """``prettyprint`` is inherited from the base class by both subclasses."""
    problem.binary_var("x")
    _add_objective(problem, linear={"x": 1}, name="o")
    text = problem.prettyprint()
    assert text.startswith("Problem name: p\n\n")
    assert "  x 'o'\n\nSubject to" in text
