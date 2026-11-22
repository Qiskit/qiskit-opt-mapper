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
"""Tests for MultiObjectiveProblem support in all converters."""

import warnings

import numpy as np
import pytest
from qiskit_addon_opt_mapper import MultiObjectiveProblem, OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    BinaryToLinearBinary,
    BinaryToSpin,
    EqualityToPenalty,
    InequalityToEquality,
    IntegerToBinary,
    LinearInequalityToPenalty,
    MaximizeToMinimize,
    MinimizeToMaximize,
    OptimizationProblemToHubo,
    OptimizationProblemToQubo,
    SpinToBinary,
)
from qiskit_addon_opt_mapper.converters.optimization_problem_converter import (
    OptimizationProblemConverter,
)


def _all_subclasses(cls):
    """Return every direct and indirect subclass of `cls`."""
    subs = set(cls.__subclasses__())
    for sub in list(subs):
        subs |= _all_subclasses(sub)
    return subs


def _make_binary_mo():
    problem = MultiObjectiveProblem(name="binary_mo")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.add_objective(sense="min", name="cost", linear={"x": 1, "y": 2})
    problem.add_objective(sense="max", name="profit", linear={"x": 3, "y": 1}, constant=5.0)
    return problem


def _make_spin_mo():
    problem = MultiObjectiveProblem(name="spin_mo")
    problem.spin_var("s0")
    problem.spin_var("s1")
    problem.add_objective(sense="min", name="energy", linear={"s0": 1}, quadratic={("s0", "s1"): 2})
    problem.add_objective(sense="max", name="magnetization", linear={"s1": 3}, constant=-1.0)
    return problem


def _make_higher_order_mo():
    problem = MultiObjectiveProblem(name="ho_mo")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.binary_var("z")
    problem.add_objective(
        sense="min", name="ho_cost", linear={"x": 1}, higher_order={3: {("x", "y", "z"): 1.0}}
    )
    problem.add_objective(
        sense="max", name="ho_profit", linear={"y": 2, "z": 1}, quadratic={("x", "y"): -1.0}
    )
    return problem


def _make_integer_mo():
    problem = MultiObjectiveProblem(name="int_mo")
    problem.integer_var(0, 3, "i")
    problem.binary_var("b")
    problem.add_objective(sense="min", name="obj1", linear={"i": 1, "b": 2})
    problem.add_objective(
        sense="max", name="obj2", linear={"i": -2, "b": 1}, quadratic={("i", "b"): 1}
    )
    return problem


def _make_inequality_mo():
    problem = MultiObjectiveProblem(name="ineq_mo")
    problem.integer_var(0, 3, "i")
    problem.binary_var("b")
    problem.linear_constraint({"i": 1, "b": 1}, "<=", 2, name="c0")
    problem.add_objective(sense="min", name="obj1", linear={"i": 1, "b": 2})
    problem.add_objective(sense="max", name="obj2", linear={"i": 2, "b": -1})
    return problem


def _make_equality_mo():
    problem = MultiObjectiveProblem(name="eq_mo")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.linear_constraint({"x": 1, "y": 1}, "==", 1, name="c0")
    problem.add_objective(sense="min", name="obj1", linear={"x": 1, "y": -1})
    problem.add_objective(sense="max", name="obj2", linear={"x": -2, "y": 3}, constant=2.0)
    return problem


def _make_linear_inequality_mo():
    problem = MultiObjectiveProblem(name="lin_ineq_mo")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.linear_constraint({"x": 1, "y": 1}, "<=", 1, name="c0")
    problem.add_objective(sense="min", name="obj1", linear={"x": 1, "y": 2})
    problem.add_objective(sense="max", name="obj2", linear={"x": -1, "y": 1}, constant=4.0)
    return problem


CONVERTER_MO_BUILDERS = [
    (BinaryToSpin, _make_binary_mo),
    (SpinToBinary, _make_spin_mo),
    (BinaryToLinearBinary, _make_higher_order_mo),
    (IntegerToBinary, _make_integer_mo),
    (InequalityToEquality, _make_inequality_mo),
    (EqualityToPenalty, _make_equality_mo),
    (LinearInequalityToPenalty, _make_linear_inequality_mo),
    (MaximizeToMinimize, _make_binary_mo),
    (MinimizeToMaximize, _make_binary_mo),
    (OptimizationProblemToQubo, _make_equality_mo),
    (OptimizationProblemToHubo, _make_equality_mo),
]


def _build_so_from_mo(mo_problem: MultiObjectiveProblem, obj_idx: int) -> OptimizationProblem:
    """Build a single-objective problem matching the mo_problem with only one objective."""
    so_problem = OptimizationProblem(name=mo_problem.name)
    for v in mo_problem.variables:
        so_problem._add_variable(
            name=v.name, vartype=v.vartype, lowerbound=v.lowerbound, upperbound=v.upperbound
        )
    for lc in mo_problem.linear_constraints:
        so_problem.linear_constraint(lc.linear.to_dict(use_name=True), lc.sense, lc.rhs, lc.name)
    for qc in mo_problem.quadratic_constraints:
        so_problem.quadratic_constraint(
            qc.linear.to_dict(use_name=True),
            qc.quadratic.to_dict(use_name=True),
            qc.sense,
            qc.rhs,
            qc.name,
        )
    for hoc in mo_problem.higher_order_constraints:
        ho_dict = {d: expr.to_dict(use_name=True) for d, expr in hoc.higher_order.items()}
        so_problem.higher_order_constraint(
            hoc.linear.to_dict(use_name=True),
            hoc.quadratic.to_dict(use_name=True),
            ho_dict,
            hoc.sense,
            hoc.rhs,
            hoc.name,
        )

    obj = mo_problem.get_objective(obj_idx)
    so_problem._append_or_set_objective_from_parts(
        obj.sense,
        constant=obj.constant,
        linear=obj.linear.to_dict(use_name=True),
        quadratic=obj.quadratic.to_dict(use_name=True),
        higher_order={d: expr.to_dict(use_name=True) for d, expr in obj.higher_order.items()},
        name=obj.name,
    )
    return so_problem


def _assert_objectives_equal(obj1, obj2):
    """Assert that two objectives have identical senses, names, and coefficients."""
    assert obj1.name == obj2.name
    assert obj1.sense == obj2.sense
    assert obj1.constant == pytest.approx(obj2.constant)
    assert obj1.linear.to_dict(use_name=True) == pytest.approx(obj2.linear.to_dict(use_name=True))
    assert obj1.quadratic.to_dict(use_name=True) == pytest.approx(
        obj2.quadratic.to_dict(use_name=True)
    )
    ho1 = {d: expr.to_dict(use_name=True) for d, expr in obj1.higher_order.items()}
    ho2 = {d: expr.to_dict(use_name=True) for d, expr in obj2.higher_order.items()}
    assert ho1.keys() == ho2.keys()
    for d in ho1:
        assert ho1[d] == pytest.approx(ho2[d])


@pytest.mark.parametrize(
    ("converter_cls", "builder"),
    CONVERTER_MO_BUILDERS,
    ids=[c.__name__ for c, _ in CONVERTER_MO_BUILDERS],
)
def test_all_converters_convert_multi_objective_problem(converter_cls, builder):
    """Every converter preserves MultiObjectiveProblem class, objective order, names, and values."""
    mo_problem = builder()
    so_problem0 = _build_so_from_mo(mo_problem, 0)
    so_problem1 = _build_so_from_mo(mo_problem, 1)

    conv = converter_cls()
    converted_mo = conv.convert(mo_problem)

    # Use the penalty determined for the MO problem if applicable
    if hasattr(conv, "penalty") and conv.penalty is not None:
        conv0 = converter_cls(penalty=conv.penalty)
        conv1 = converter_cls(penalty=conv.penalty)
    else:
        conv0 = converter_cls()
        conv1 = converter_cls()

    converted_so0 = conv0.convert(so_problem0)
    converted_so1 = conv1.convert(so_problem1)

    assert isinstance(converted_mo, MultiObjectiveProblem)
    assert converted_mo.name == mo_problem.name
    assert converted_mo.get_num_objectives() == 2

    _assert_objectives_equal(converted_mo.get_objective(0), converted_so0.objective)
    _assert_objectives_equal(converted_mo.get_objective(1), converted_so1.objective)


def test_all_converters_are_registered_for_multi_objective():
    """Ensure any newly added converter is registered in the multi-objective test suite."""
    all_registered = {c for c, _ in CONVERTER_MO_BUILDERS}
    subs = _all_subclasses(OptimizationProblemConverter)
    concrete_public = {
        c
        for c in subs
        if not getattr(c, "__abstractmethods__", None) and not c.__name__.startswith("_")
    }
    unregistered = concrete_public - all_registered
    assert not unregistered, f"New converter(s) missing multi-objective tests: {unregistered}"


@pytest.mark.parametrize(
    ("converter_cls", "builder"),
    CONVERTER_MO_BUILDERS,
    ids=[c.__name__ for c, _ in CONVERTER_MO_BUILDERS],
)
def test_converter_with_zero_and_one_objective_multi_objective_problem(converter_cls, builder):
    """A MultiObjectiveProblem with 0 or 1 objective converts cleanly."""
    base_mo = builder()

    # Zero objectives
    zero_mo = MultiObjectiveProblem(name=base_mo.name)
    for v in base_mo.variables:
        zero_mo._add_variable(
            name=v.name, vartype=v.vartype, lowerbound=v.lowerbound, upperbound=v.upperbound
        )
    for lc in base_mo.linear_constraints:
        zero_mo.linear_constraint(lc.linear.to_dict(use_name=True), lc.sense, lc.rhs, lc.name)

    conv_zero = converter_cls().convert(zero_mo)
    assert isinstance(conv_zero, MultiObjectiveProblem)
    assert conv_zero.get_num_objectives() == 0
    assert conv_zero.name == base_mo.name

    # One objective
    one_mo = MultiObjectiveProblem(name=base_mo.name)
    for v in base_mo.variables:
        one_mo._add_variable(
            name=v.name, vartype=v.vartype, lowerbound=v.lowerbound, upperbound=v.upperbound
        )
    for lc in base_mo.linear_constraints:
        one_mo.linear_constraint(lc.linear.to_dict(use_name=True), lc.sense, lc.rhs, lc.name)
    obj0 = base_mo.get_objective(0)
    one_mo._append_or_set_objective_from_parts(
        obj0.sense,
        constant=obj0.constant,
        linear=obj0.linear.to_dict(use_name=True),
        quadratic=obj0.quadratic.to_dict(use_name=True),
        higher_order={d: expr.to_dict(use_name=True) for d, expr in obj0.higher_order.items()},
        name=obj0.name,
    )

    conv_one = converter_cls().convert(one_mo)
    assert isinstance(conv_one, MultiObjectiveProblem)
    assert conv_one.get_num_objectives() == 1
    assert conv_one.get_objective(0).name == obj0.name


@pytest.mark.parametrize(
    ("converter_cls", "builder"),
    CONVERTER_MO_BUILDERS,
    ids=[c.__name__ for c, _ in CONVERTER_MO_BUILDERS],
)
def test_converter_does_not_rewarn_for_non_printable_name_in_multi_objective(
    converter_cls, builder
):
    """Non-printable objective names in MultiObjectiveProblem do not re-warn during conversion."""
    mo_problem = builder()
    bad_mo = MultiObjectiveProblem()
    for v in mo_problem.variables:
        bad_mo._add_variable(
            name=v.name, vartype=v.vartype, lowerbound=v.lowerbound, upperbound=v.upperbound
        )
    for lc in mo_problem.linear_constraints:
        bad_mo.linear_constraint(lc.linear.to_dict(use_name=True), lc.sense, lc.rhs, lc.name)

    for i in range(mo_problem.get_num_objectives()):
        obj = mo_problem.get_objective(i)
        bad_mo._append_or_set_objective_from_parts(
            obj.sense,
            constant=obj.constant,
            linear=obj.linear.to_dict(use_name=True),
            quadratic=obj.quadratic.to_dict(use_name=True),
            higher_order={d: expr.to_dict(use_name=True) for d, expr in obj.higher_order.items()},
            name=f"bad\nname_{i}",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converter_cls().convert(bad_mo)
    assert not [w for w in caught if "not printable" in str(w.message)]


def test_equality_to_penalty_preserves_problem_name():
    """EqualityToPenalty preserves problem name for both OptimizationProblem and MultiObjectiveProblem."""
    so_problem = OptimizationProblem(name="my_so_problem")
    so_problem.binary_var("x")
    so_problem.linear_constraint({"x": 1}, "==", 1, name="c0")
    so_problem.minimize(linear={"x": 1})
    conv_so = EqualityToPenalty().convert(so_problem)
    assert conv_so.name == "my_so_problem"

    mo_problem = MultiObjectiveProblem(name="my_mo_problem")
    mo_problem.binary_var("x")
    mo_problem.linear_constraint({"x": 1}, "==", 1, name="c0")
    mo_problem.add_objective(sense="min", name="cost", linear={"x": 1})
    conv_mo = EqualityToPenalty().convert(mo_problem)
    assert conv_mo.name == "my_mo_problem"


@pytest.mark.parametrize(
    ("converter_cls", "builder"),
    CONVERTER_MO_BUILDERS,
    ids=[c.__name__ for c, _ in CONVERTER_MO_BUILDERS],
)
def test_interpret_matches_single_objective(converter_cls, builder):
    """interpret(x) on converted MultiObjectiveProblem matches single-objective interpret."""
    mo_problem = builder()
    so_problem = _build_so_from_mo(mo_problem, 0)

    conv_mo_instance = converter_cls()
    converted_mo = conv_mo_instance.convert(mo_problem)

    conv_so_instance = converter_cls()
    converted_so = conv_so_instance.convert(so_problem)

    n_vars_mo = converted_mo.get_num_vars()
    n_vars_so = converted_so.get_num_vars()
    test_sol_mo = [1.0 if i % 2 == 0 else 0.0 for i in range(n_vars_mo)]
    test_sol_so = [1.0 if i % 2 == 0 else 0.0 for i in range(n_vars_so)]

    res_mo = conv_mo_instance.interpret(test_sol_mo)
    res_so = conv_so_instance.interpret(test_sol_so)
    # Check that the original variable values returned match
    orig_num_vars = mo_problem.get_num_vars()
    np.testing.assert_array_equal(res_mo[:orig_num_vars], res_so[:orig_num_vars])


def test_shared_penalty_feasibility_guarantee():
    """Shared penalty in EqualityToPenalty and LinearInequalityToPenalty guarantees feasibility."""
    # Multi-objective problem with 2 binary variables and an equality constraint x + y == 1
    # Feasible solutions: (1, 0) and (0, 1). Infeasible: (0, 0) and (1, 1).
    mo = MultiObjectiveProblem()
    mo.binary_var("x")
    mo.binary_var("y")
    mo.linear_constraint({"x": 1, "y": 1}, "==", 1, name="c0")
    mo.add_objective(sense="min", name="f1", linear={"x": 10, "y": -5})
    mo.add_objective(sense="min", name="f2", linear={"x": -2, "y": 8})

    conv = EqualityToPenalty()
    penalized = conv.convert(mo)

    # For both objectives, feasible points must have lower penalized objective than infeasible points
    for obj_idx in range(2):
        obj = penalized.get_objective(obj_idx)
        feasible_vals = [obj.evaluate([1, 0]), obj.evaluate([0, 1])]
        infeasible_vals = [obj.evaluate([0, 0]), obj.evaluate([1, 1])]
        assert min(infeasible_vals) > min(feasible_vals)

    # Test LinearInequalityToPenalty on x <= y constraint
    mo_ineq = MultiObjectiveProblem()
    mo_ineq.binary_var("x")
    mo_ineq.binary_var("y")
    mo_ineq.linear_constraint({"x": 1, "y": -1}, "<=", 0, name="c0")
    mo_ineq.add_objective(sense="min", name="f1", linear={"x": 10, "y": -5})
    mo_ineq.add_objective(sense="min", name="f2", linear={"x": -2, "y": 8})

    conv_ineq = LinearInequalityToPenalty()
    penalized_ineq = conv_ineq.convert(mo_ineq)

    # Feasible: (0, 0), (0, 1), (1, 1). Infeasible: (1, 0)
    for obj_idx in range(2):
        obj = penalized_ineq.get_objective(obj_idx)
        feasible_vals = [obj.evaluate([0, 0]), obj.evaluate([0, 1]), obj.evaluate([1, 1])]
        infeasible_val = obj.evaluate([1, 0])
        assert infeasible_val > min(feasible_vals)


def test_qubo_compatibility_checks_all_objectives():
    """OptimizationProblemToQubo flags higher-order terms in any objective."""
    mo = MultiObjectiveProblem()
    mo.binary_var("x")
    mo.binary_var("y")
    mo.binary_var("z")
    mo.add_objective(sense="min", name="f1", linear={"x": 1})
    mo.add_objective(sense="min", name="f2", higher_order={3: {("x", "y", "z"): 1.0}})
    conv = OptimizationProblemToQubo()
    assert not conv.is_compatible(mo)
    msg = conv.get_compatibility_msg(mo)
    assert "Higher order objective functions are not supported" in msg
