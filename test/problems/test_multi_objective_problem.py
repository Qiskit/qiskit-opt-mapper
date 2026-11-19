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
"""Tests for MultiObjectiveProblem."""

import warnings

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper import (
    BaseOptimizationProblem,
    MultiObjectiveProblem,
    OptimizationError,
    OptimizationProblem,
)
from qiskit_addon_opt_mapper.problems import ObjSense, OptimizationObjective
from qiskit_addon_opt_mapper.translators import to_isings


@pytest.fixture(name="problem")
def fixture_problem() -> MultiObjectiveProblem:
    """Two binary variables, no objectives."""
    problem = MultiObjectiveProblem("m")
    problem.binary_var("x")
    problem.binary_var("y")
    return problem


@pytest.fixture(name="two_objectives")
def fixture_two_objectives(problem) -> MultiObjectiveProblem:
    """Two binary variables and two objectives with different senses."""
    problem.add_objective(linear={"x": 1, "y": 2})
    problem.add_objective(linear={"x": 3}, sense="max")
    return problem


def test_is_multi_objective():
    """``is_multi_objective`` distinguishes the two subclasses."""
    assert MultiObjectiveProblem().is_multi_objective is True
    assert OptimizationProblem().is_multi_objective is False


def test_isinstance_of_base():
    """Invariant 1: both subclasses are BaseOptimizationProblem instances."""
    assert isinstance(MultiObjectiveProblem(), BaseOptimizationProblem)
    assert isinstance(OptimizationProblem(), BaseOptimizationProblem)


def test_base_cannot_be_instantiated():
    """Invariant 2: the abstract base cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseOptimizationProblem()  # pylint: disable=abstract-class-instantiated


OPTIMIZATION_PROBLEM_PUBLIC_NAMES = {
    "Status",
    "binary_var",
    "binary_var_dict",
    "binary_var_list",
    "clear",
    "continuous_var",
    "continuous_var_dict",
    "continuous_var_list",
    "get_feasibility_info",
    "get_higher_order_constraint",
    "get_linear_constraint",
    "get_num_binary_vars",
    "get_num_continuous_vars",
    "get_num_higher_order_constraints",
    "get_num_integer_vars",
    "get_num_linear_constraints",
    "get_num_quadratic_constraints",
    "get_num_spin_vars",
    "get_num_vars",
    "get_quadratic_constraint",
    "get_variable",
    "higher_order_constraint",
    "higher_order_constraints",
    "higher_order_constraints_index",
    "integer_var",
    "integer_var_dict",
    "integer_var_list",
    "is_feasible",
    "iter_higher_order_constraints",
    "linear_constraint",
    "linear_constraints",
    "linear_constraints_index",
    "maximize",
    "minimize",
    "name",
    "objective",
    "prettyprint",
    "quadratic_constraint",
    "quadratic_constraints",
    "quadratic_constraints_index",
    "remove_higher_order_constraint",
    "remove_linear_constraint",
    "remove_quadratic_constraint",
    "spin_var",
    "spin_var_dict",
    "spin_var_list",
    "status",
    "substitute_variables",
    "to_ising",
    "variables",
    "variables_index",
}


def test_optimization_problem_keeps_public_names():
    """Invariant 3: every public name of OptimizationProblem at the P4 tip is still there."""
    public = {name for name in dir(OptimizationProblem()) if not name.startswith("_")}
    assert public >= OPTIMIZATION_PROBLEM_PUBLIC_NAMES


def test_zero_objectives_is_legal():
    """A problem with no objectives is legal."""
    problem = MultiObjectiveProblem("m")
    assert problem.get_num_objectives() == 0
    assert problem.objectives == []
    assert problem.objectives_index == {}
    assert problem.to_ising() == []
    assert repr(problem) == "<MultiObjectiveProblem: 0 objectives, 0 variables, 0 constraints, 'm'>"
    assert str(problem) == "no objectives (0 variables, 0 constraints, 'm')"


def test_one_objective_repr(problem):
    """Singular form '1 objective' in repr."""
    problem.add_objective(linear={"x": 1})
    assert repr(problem) == "<MultiObjectiveProblem: 1 objective, 2 variables, 0 constraints, 'm'>"


def test_auto_names_and_returned_objective(problem):
    """Unnamed objectives are called f0, f1, ...; the returned object is the live one."""
    first = problem.add_objective(linear={"x": 1})
    second = problem.add_objective(linear={"y": 1})
    assert (first.name, second.name) == ("f0", "f1")
    assert problem.get_objective(0) is first
    assert problem.get_objective("f1") is second
    assert problem.objectives_index == {"f0": 0, "f1": 1}
    assert problem.get_num_objectives() == 2
    assert problem.objectives == [first, second]


def test_explicit_name(problem):
    """An explicit name is stored on the objective and in the index."""
    objective = problem.add_objective(linear={"x": 1}, name="cost")
    assert objective.name == "cost"
    assert problem.objectives_index == {"cost": 0}
    assert problem.get_objective("cost") is objective


def test_auto_name_collision_loop():
    """With ``f1`` taken and one objective present, the next auto name is ``f2``."""
    problem = MultiObjectiveProblem()
    problem.add_objective(name="f1")
    assert problem.add_objective().name == "f2"


def test_empty_name_is_auto_named(problem):
    """An empty name behaves like ``None``."""
    assert problem.add_objective(name="").name == "f0"


def test_duplicate_name_raises(problem):
    """Invariant 5: an explicit duplicate name raises OptimizationError."""
    problem.add_objective(name="cost")
    with pytest.raises(OptimizationError, match="Objective name already exists: cost"):
        problem.add_objective(name="cost")
    assert problem.get_num_objectives() == 1


def test_non_printable_name_warns(problem):
    """A non-printable objective name emits the same warning as other names."""
    with pytest.warns(UserWarning, match="Objective name is not printable"):
        problem.add_objective(name="a\nb")


def test_none_coefficients_become_empty(problem):
    """None coefficients are replaced by empty ones."""
    objective = problem.add_objective()
    assert objective.constant == 0
    assert objective.linear.to_dict() == {}
    assert objective.quadratic.to_dict() == {}
    assert objective.higher_order == {}


@pytest.mark.parametrize(
    ("sense", "expected"),
    [
        (ObjSense.MINIMIZE, ObjSense.MINIMIZE),
        (ObjSense.MAXIMIZE, ObjSense.MAXIMIZE),
        ("min", ObjSense.MINIMIZE),
        ("Minimize", ObjSense.MINIMIZE),
        ("max", ObjSense.MAXIMIZE),
        ("MAXIMIZE", ObjSense.MAXIMIZE),
    ],
)
def test_sense_conversion(problem, sense, expected):
    """The sense can be given as an ObjSense member or as a string."""
    assert problem.add_objective(sense=sense).sense == expected


def test_default_sense_is_minimize(problem):
    """The default sense is minimize."""
    assert problem.add_objective().sense == ObjSense.MINIMIZE


def test_invalid_sense_raises(problem):
    """An invalid sense string raises OptimizationError and adds nothing."""
    with pytest.raises(OptimizationError, match="Invalid sense: MINIMUM"):
        problem.add_objective(sense="Minimum")
    assert problem.get_num_objectives() == 0
    assert problem.objectives_index == {}


def test_higher_order_objective():
    """Objectives accept higher-order terms."""
    problem = MultiObjectiveProblem()
    for name in "xyz":
        problem.binary_var(name)
    objective = problem.add_objective(higher_order={3: {("x", "y", "z"): 4}})
    assert objective.higher_order[3].to_dict(use_name=True) == {("x", "y", "z"): 4}


def test_constant_linear_quadratic(problem):
    """All parts of an objective are stored."""
    objective = problem.add_objective(
        constant=5, linear={"x": 1}, quadratic={("x", "y"): 2}, sense="max", name="o"
    )
    assert objective.constant == 5
    assert objective.linear.to_dict(use_name=True) == {"x": 1}
    assert objective.quadratic.to_dict(use_name=True) == {("x", "y"): 2}


def test_mutating_returned_objective_is_visible(problem):
    """The returned objective is the live object held by the problem."""
    objective = problem.add_objective(linear={"x": 1})
    objective.sense = ObjSense.MAXIMIZE
    objective.linear = {"y": 7}
    assert problem.get_objective("f0").sense == ObjSense.MAXIMIZE
    assert problem.get_objective(0).linear.to_dict(use_name=True) == {"y": 7}


def test_get_objective_errors(problem):
    """Unknown names raise KeyError, out-of-range indices raise IndexError."""
    problem.add_objective()
    with pytest.raises(KeyError):
        problem.get_objective("nope")
    with pytest.raises(IndexError):
        problem.get_objective(5)


@pytest.mark.parametrize("index_type", [np.int32, np.int64, np.uint8])
def test_get_objective_accepts_numpy_integer_index(problem, index_type):
    """A NumPy integer is an index, not a name."""
    first = problem.add_objective(name="a")
    second = problem.add_objective(name="b")
    assert problem.get_objective(index_type(0)) is first
    assert problem.get_objective(index_type(1)) is second
    with pytest.raises(IndexError):
        problem.get_objective(index_type(2))


def test_get_objective_indexes_from_numpy_array(problem):
    """Indices taken from a NumPy array work."""
    for name in ("a", "b", "c"):
        problem.add_objective(name=name)
    order = np.array([2, 0])
    assert [problem.get_objective(k).name for k in order] == ["c", "a"]


def test_get_objective_negative_index(problem):
    """Negative integer indices follow Python sequence indexing."""
    first = problem.add_objective(name="a")
    second = problem.add_objective(name="b")
    assert problem.get_objective(-1) is second
    assert problem.get_objective(-2) is first
    with pytest.raises(IndexError):
        problem.get_objective(-3)


def test_remove_objective_negative_index(problem):
    """Negative integer indices in remove_objective follow Python sequence indexing."""
    problem.add_objective(name="a")
    problem.add_objective(name="b")
    problem.add_objective(name="c")
    problem.remove_objective(-1)
    assert [obj.name for obj in problem.objectives] == ["a", "b"]
    assert problem.objectives_index == {"a": 0, "b": 1}


def test_remove_objective_accepts_numpy_integer_index(problem):
    """``remove_objective`` also treats a NumPy integer as an index."""
    for name in ("a", "b", "c"):
        problem.add_objective(name=name)
    problem.remove_objective(np.int64(1))
    assert problem.objectives_index == {"a": 0, "c": 1}


def test_remove_by_index_rebuilds_index(problem):
    """Removing by index re-numbers the remaining objectives."""
    for name in ("a", "b", "c"):
        problem.add_objective(name=name)
    problem.remove_objective(0)
    assert problem.objectives_index == {"b": 0, "c": 1}
    assert [obj.name for obj in problem.objectives] == ["b", "c"]


def test_remove_by_name_rebuilds_index(problem):
    """Removing by name re-numbers the remaining objectives."""
    for name in ("a", "b", "c"):
        problem.add_objective(name=name)
    problem.remove_objective("b")
    assert problem.objectives_index == {"a": 0, "c": 1}
    assert problem.get_objective("c") is problem.get_objective(1)


def test_remove_errors(problem):
    """Invalid removals raise and leave the problem untouched."""
    problem.add_objective(name="a")
    with pytest.raises(KeyError):
        problem.remove_objective("nope")
    with pytest.raises(IndexError):
        problem.remove_objective(3)
    assert problem.objectives_index == {"a": 0}


def test_name_can_be_reused_after_removal(problem):
    """A removed objective's name can be added again."""
    problem.add_objective(name="a")
    problem.remove_objective("a")
    assert problem.add_objective(name="a").name == "a"
    assert problem.objectives_index == {"a": 0}


@pytest.mark.parametrize("remove_first", [True, False])
def test_names_always_agree_with_index(problem, remove_first):
    """Invariant 10: ``get_objective(k).name`` agrees with ``objectives_index`` for every k."""
    for _ in range(4):
        problem.add_objective()
    problem.remove_objective(0 if remove_first else "f2")
    problem.add_objective()
    problem.add_objective(name="named")
    for name, idx in problem.objectives_index.items():
        assert problem.get_objective(idx).name == name
    assert len(problem.objectives_index) == problem.get_num_objectives()


def test_objective_name_has_no_setter(problem):
    """Invariant 10: an objective cannot be renamed."""
    objective = problem.add_objective()
    with pytest.raises(AttributeError):
        objective.name = "other"  # type: ignore[misc]  # pylint: disable=attribute-defined-outside-init


def test_unnamed_objective_defaults():
    """Invariant 9: the name is opt-in on the single-objective class."""
    problem = OptimizationProblem()
    assert OptimizationObjective(problem).name == ""
    assert problem.objective.name == ""


def test_repr_and_str(two_objectives):
    """Exact ``repr`` and ``str`` of a two-objective problem."""
    assert repr(two_objectives) == (
        "<MultiObjectiveProblem: 2 objectives, 2 variables, 0 constraints, 'm'>"
    )
    assert str(two_objectives) == (
        "minimize x + 2*y 'f0'; maximize 3*x 'f1' (2 variables, 0 constraints, 'm')"
    )


def test_clear_resets_objectives(two_objectives):
    """``clear`` empties the objectives and the index."""
    two_objectives.clear()
    assert two_objectives.get_num_objectives() == 0
    assert two_objectives.objectives_index == {}


def test_to_ising_one_pair_per_objective(two_objectives):
    """Invariant 7: ``to_ising`` has one pair per objective, equal to the single-objective one."""
    result = two_objectives.to_ising()
    assert len(result) == two_objectives.get_num_objectives() == 2
    for (op, offset), objective in zip(result, two_objectives.objectives, strict=True):
        single = OptimizationProblem()
        single.binary_var("x")
        single.binary_var("y")
        getattr(single, objective.sense.name.lower())(
            linear=objective.linear.to_dict(use_name=True)
        )
        single_op, single_offset = single.to_ising()
        assert isinstance(op, SparsePauliOp)
        assert op == single_op
        assert offset == single_offset


def test_to_ising_matches_to_isings(two_objectives):
    """The method delegates to ``to_isings``."""
    assert two_objectives.to_ising() == to_isings(two_objectives)


def test_to_ising_rejects_constraints(two_objectives):
    """The constraint guard still raises."""
    two_objectives.linear_constraint({"x": 1}, "==", 1)
    with pytest.raises(OptimizationError, match="no constraint"):
        two_objectives.to_ising()


def test_to_ising_rejects_non_binary():
    """Non-binary variables are rejected."""
    problem = MultiObjectiveProblem()
    problem.integer_var(0, 3, "i")
    problem.add_objective(linear={"i": 1})
    with pytest.raises(OptimizationError):
        problem.to_ising()


def test_to_ising_without_variables():
    """Zero variables: each objective yields a 1-qubit zero operator."""
    problem = MultiObjectiveProblem()
    problem.add_objective(constant=2)
    problem.add_objective(constant=3, sense="max")
    result = problem.to_ising()
    assert len(result) == 2
    assert [offset for _, offset in result] == [2, -3]


def test_prettyprint_two_objectives(two_objectives):
    """Exact ``prettyprint`` of a two-objective problem: one blank line before Subject to."""
    assert two_objectives.prettyprint() == "\n".join(
        [
            "Problem name: m",
            "",
            "Minimize",
            "  x + 2*y 'f0'",
            "Maximize",
            "  3*x 'f1'",
            "",
            "Subject to",
            "  No constraints",
            "",
            "  Binary variables (2)",
            "    x y",
            "",
        ]
    )


def test_prettyprint_two_objectives_with_constraint(two_objectives):
    """The constraints section is unaffected by the objective block."""
    two_objectives.linear_constraint({"x": 1}, "<=", 1, "c")
    assert two_objectives.prettyprint() == "\n".join(
        [
            "Problem name: m",
            "",
            "Minimize",
            "  x + 2*y 'f0'",
            "Maximize",
            "  3*x 'f1'",
            "",
            "Subject to",
            "  Linear constraints (1)",
            "    x <= 1  'c'",
            "",
            "  Binary variables (2)",
            "    x y",
            "",
        ]
    )


def test_prettyprint_zero_objectives():
    """Exact ``prettyprint`` of a problem without objectives."""
    assert MultiObjectiveProblem("e").prettyprint() == "\n".join(
        [
            "Problem name: e",
            "",
            "  No objectives",
            "",
            "Subject to",
            "  No constraints",
            "",
            "  No variables",
            "",
        ]
    )


def test_prettyprint_single_multi_objective_keeps_one_blank_line(problem):
    """A single objective in a multi-objective problem also has one blank line."""
    problem.add_objective(linear={"x": 1}, name="only")
    text = problem.prettyprint()
    assert "  x 'only'\n\nSubject to" in text
    assert "\n\n\n" not in text


def test_prettyprint_rejects_non_printable_objective_name(problem):
    """A non-printable objective name raises when pretty printing."""
    with pytest.warns(UserWarning):
        problem.add_objective(name="a\nb")
    with pytest.raises(OptimizationError, match="Objective name is not printable"):
        problem.prettyprint()


def test_prettyprint_wraps_name_suffix(problem):
    """A long expression wraps the name suffix instead of overflowing the line."""
    for i in range(30):
        problem.binary_var(f"v{i}")
    problem.add_objective(linear={f"v{i}": 1 for i in range(30)}, name="long_objective_name")
    text = problem.prettyprint()
    objective_block = text.split("\n\nSubject to")[0]
    assert all(len(line) <= 80 for line in objective_block.splitlines())
    assert "'long_objective_name'" in objective_block


def test_substitute_returns_multi_objective_problem(two_objectives):
    """Invariant 6: names, order, senses are preserved and the class is kept."""
    result = two_objectives.substitute_variables(constants={"x": 1})
    assert type(result) is MultiObjectiveProblem
    assert result.name == "m"
    assert [o.name for o in result.objectives] == ["f0", "f1"]
    assert [o.sense for o in result.objectives] == [ObjSense.MINIMIZE, ObjSense.MAXIMIZE]
    assert result.objectives_index == {"f0": 0, "f1": 1}
    assert result.status == BaseOptimizationProblem.Status.VALID


def test_substitute_constants(two_objectives):
    """Constants are folded into each objective."""
    result = two_objectives.substitute_variables(constants={"x": 1})
    first, second = result.objectives
    assert (first.constant, first.linear.to_dict(use_name=True)) == (1, {"y": 2})
    assert (second.constant, second.linear.to_dict()) == (3, {})


def test_substitute_variables_by_variables(two_objectives):
    """A variable replaced by a scaled variable is rewritten in every objective."""
    result = two_objectives.substitute_variables(variables={"x": ("y", 2)})
    first, second = result.objectives
    assert first.linear.to_dict(use_name=True) == {"y": 4}
    assert second.linear.to_dict(use_name=True) == {"y": 6}


def test_substitute_infeasible_sets_status(problem):
    """An infeasible substitution sets the status of the result."""
    problem.add_objective(linear={"x": 1})
    problem.linear_constraint({"x": 1}, "==", 1, "c")
    result = problem.substitute_variables(constants={"x": 0})
    assert result.status == BaseOptimizationProblem.Status.INFEASIBLE


def test_substitute_invalid_bounds_sets_status(problem):
    """A constant outside the variable bounds is infeasible, objectives are kept."""
    problem.add_objective(linear={"x": 1})
    result = problem.substitute_variables(constants={"x": 5})
    assert result.status == BaseOptimizationProblem.Status.INFEASIBLE
    assert result.get_num_objectives() == 1


def test_substitute_keeps_constraint_and_objective_names(two_objectives):
    """Constraints survive next to the objectives."""
    two_objectives.linear_constraint({"x": 1, "y": 1}, "<=", 1, "c")
    result = two_objectives.substitute_variables(constants={"x": 1})
    assert result.get_linear_constraint("c").rhs == 0
    assert result.get_num_objectives() == 2


def test_substitute_zero_objectives():
    """Substituting a problem without objectives gives one without objectives."""
    problem = MultiObjectiveProblem("e")
    problem.binary_var("x")
    result = problem.substitute_variables(constants={"x": 1})
    assert type(result) is MultiObjectiveProblem
    assert result.get_num_objectives() == 0


def test_substitute_preserves_names_with_gaps(problem):
    """Round-tripping keeps existing names instead of re-running default naming.

    After ``f0`` is removed the remaining objectives are ``f1`` and ``f2``. Default naming
    would give ``f0`` and ``f1`` again, so equality of names shows the original names are
    passed through ``add_objective``.
    """
    problem.add_objective(linear={"x": 1})
    problem.add_objective(linear={"x": 2})
    problem.add_objective(linear={"x": 3}, sense="maximize")
    problem.remove_objective("f0")

    result = problem.substitute_variables(constants={"x": 1})

    assert [o.name for o in result.objectives] == ["f1", "f2"]
    assert result.objectives_index == {"f1": 0, "f2": 1}
    assert [o.constant for o in result.objectives] == [2, 3]
    assert [o.sense for o in result.objectives] == [ObjSense.MINIMIZE, ObjSense.MAXIMIZE]


def test_substitute_preserves_explicit_and_default_names(problem):
    """Explicit and auto-generated names both survive a substitution unchanged."""
    problem.add_objective(linear={"x": 1}, name="cost")
    problem.add_objective(linear={"x": 2})
    problem.add_objective(linear={"x": 3}, name="f0")

    result = problem.substitute_variables(constants={"x": 1})

    assert [o.name for o in result.objectives] == ["cost", "f1", "f0"]
    assert result.objectives_index == problem.objectives_index


def test_invariant_12_objective_retained_as_constant_constraint_dropped():
    """Invariant 12: a vanished objective stays as a constant, a vanished constraint is dropped."""
    problem = MultiObjectiveProblem("m")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.add_objective(linear={"x": 2}, constant=1, name="gone")
    problem.add_objective(linear={"y": 1}, name="kept")
    problem.linear_constraint({"x": 1}, "<=", 1, "vanishing")
    problem.linear_constraint({"y": 1}, "<=", 1, "surviving")

    result = problem.substitute_variables(constants={"x": 1})

    assert result.get_num_objectives() == problem.get_num_objectives() == 2
    assert result.objectives_index == {"gone": 0, "kept": 1}
    gone = result.get_objective("gone")
    assert gone.constant == 3
    assert gone.linear.to_dict() == {}
    assert gone.quadratic.to_dict() == {}
    assert gone.higher_order == {}
    assert result.get_objective("kept").linear.to_dict(use_name=True) == {"y": 1}
    assert result.linear_constraints_index == {"surviving": 0}
    assert result.status == BaseOptimizationProblem.Status.VALID


def test_invariant_12_objective_retained_even_if_constraint_infeasible():
    """Only constraints can make the substitution infeasible; the objective is still kept."""
    problem = MultiObjectiveProblem("m")
    problem.binary_var("x")
    problem.add_objective(linear={"x": 1}, name="o")
    problem.linear_constraint({"x": 1}, ">=", 2, "impossible")
    result = problem.substitute_variables(constants={"x": 1})
    assert result.status == BaseOptimizationProblem.Status.INFEASIBLE
    assert result.get_objective("o").constant == 1


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_invariant_12_single_objective_analogue(sense):
    """Invariant 12 on OptimizationProblem: the vanished objective is a constant, never dropped."""
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    getattr(problem, sense)(constant=1, linear={"x": 2}, name="o")
    problem.linear_constraint({"x": 1}, "<=", 1, "vanishing")
    problem.linear_constraint({"y": 1}, "<=", 1, "surviving")

    result = problem.substitute_variables(constants={"x": 1})

    assert result.objective.constant == 3
    assert result.objective.linear.to_dict() == {}
    assert result.objective.name == "o"
    assert result.objective.sense == problem.objective.sense
    assert result.linear_constraints_index == {"surviving": 0}
    assert result.status == BaseOptimizationProblem.Status.VALID


def test_copy_from_reparents_objectives(two_objectives):
    """``_copy_from`` re-parents the list of objectives to the new problem."""
    target = MultiObjectiveProblem()
    target._copy_from(two_objectives, include_name=True)  # pylint: disable=protected-access
    assert all(obj.optimization_problem is target for obj in target.objectives)
    assert target.objectives_index == {"f0": 0, "f1": 1}


def test_no_warning_for_regular_use(two_objectives):
    """Ordinary use emits no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        two_objectives.add_objective(name="plain")
        two_objectives.prettyprint()
        two_objectives.substitute_variables(constants={"x": 1})
