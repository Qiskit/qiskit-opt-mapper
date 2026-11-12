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
"""Test that every converter preserves ``objective.name`` across ``convert()``."""

import pytest
from qiskit_addon_opt_mapper import OptimizationProblem
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

# Each builder returns a problem the corresponding converter actually applies to. `name` and
# `sense` are threaded through so every `if MINIMIZE / else` branch gets exercised.


def _set_objective(problem, sense, name, **kwargs):
    method = problem.minimize if sense == "min" else problem.maximize
    method(name=name, **kwargs)


def _binary(sense, name):
    problem = OptimizationProblem()
    problem.binary_var("x")
    problem.binary_var("y")
    _set_objective(problem, sense, name, linear={"x": 1, "y": 2})
    return problem


def _spin(sense, name):
    problem = OptimizationProblem()
    problem.spin_var("s0")
    problem.spin_var("s1")
    _set_objective(problem, sense, name, linear={"s0": 1}, quadratic={("s0", "s1"): 2})
    return problem


def _higher_order(sense, name):
    problem = OptimizationProblem()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.binary_var("z")
    _set_objective(problem, sense, name, linear={"x": 1}, higher_order={3: {("x", "y", "z"): 1.0}})
    return problem


def _integer(sense, name):
    problem = OptimizationProblem()
    problem.integer_var(0, 3, "i")
    _set_objective(problem, sense, name, linear={"i": 1})
    return problem


def _inequality(sense, name):
    problem = OptimizationProblem()
    problem.integer_var(0, 3, "i")
    problem.linear_constraint({"i": 1}, "<=", 2, name="c0")
    _set_objective(problem, sense, name, linear={"i": 1})
    return problem


def _equality(sense, name):
    problem = OptimizationProblem()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.linear_constraint({"x": 1, "y": 1}, "==", 1, name="c0")
    _set_objective(problem, sense, name, linear={"x": 1})
    return problem


def _linear_inequality(sense, name):
    problem = OptimizationProblem()
    problem.binary_var("x")
    problem.binary_var("y")
    problem.linear_constraint({"x": 1, "y": 1}, "<=", 1, name="c0")
    _set_objective(problem, sense, name, linear={"x": 1})
    return problem


# (converter factory, problem builder) for every converter that emits an objective.
CONVERTERS = [
    (BinaryToSpin, _binary),
    (SpinToBinary, _spin),
    (BinaryToLinearBinary, _higher_order),
    (IntegerToBinary, _integer),
    (InequalityToEquality, _inequality),
    (EqualityToPenalty, _equality),
    (LinearInequalityToPenalty, _linear_inequality),
]

# The two composite converters, which never build an objective themselves.
COMPOSITE_CONVERTERS = [
    (OptimizationProblemToQubo, _equality),
    (OptimizationProblemToHubo, _equality),
]

# FlipProblemSense subclasses, which deep-copy the problem and mutate the objective in place.
FLIP_CONVERTERS = [
    (MaximizeToMinimize, "max"),
    (MinimizeToMaximize, "min"),
]

converter_cases = pytest.mark.parametrize(
    ("converter_cls", "builder"),
    CONVERTERS,
    ids=[c.__name__ for c, _ in CONVERTERS],
)
composite_cases = pytest.mark.parametrize(
    ("converter_cls", "builder"),
    COMPOSITE_CONVERTERS,
    ids=[c.__name__ for c, _ in COMPOSITE_CONVERTERS],
)


@converter_cases
@pytest.mark.parametrize("sense", ["min", "max"])
def test_converter_preserves_objective_name(converter_cls, builder, sense):
    """Invariant 13: a named objective keeps its name across `convert()`, for both senses."""
    converted = converter_cls().convert(builder(sense, "cost"))
    assert converted.objective.name == "cost"


@converter_cases
@pytest.mark.parametrize("sense", ["min", "max"])
def test_converter_leaves_unnamed_objective_unnamed(converter_cls, builder, sense):
    """The today-behaviour guard: an unnamed objective converts to an unnamed objective."""
    converted = converter_cls().convert(builder(sense, ""))
    assert converted.objective.name == ""


@composite_cases
@pytest.mark.parametrize("sense", ["min", "max"])
def test_composite_converter_propagates_transitively(converter_cls, builder, sense):
    """`OptimizationProblemToQubo` / `ToHubo` propagate the name through their sub-converters."""
    assert converter_cls().convert(builder(sense, "cost")).objective.name == "cost"
    assert converter_cls().convert(builder(sense, "")).objective.name == ""


@pytest.mark.parametrize(
    ("converter_cls", "sense"), FLIP_CONVERTERS, ids=[c.__name__ for c, _ in FLIP_CONVERTERS]
)
def test_flip_problem_sense_preserves_name(converter_cls, sense):
    """`FlipProblemSense` carries the name along with its deep copy."""
    assert converter_cls().convert(_binary(sense, "cost")).objective.name == "cost"
    assert converter_cls().convert(_binary(sense, "")).objective.name == ""


def test_all_converters_are_registered_for_name_propagation():
    """Ensure any newly added converter is registered in the name propagation test suite."""
    all_registered = (
        {c for c, _ in CONVERTERS}
        | {c for c, _ in COMPOSITE_CONVERTERS}
        | {c for c, _ in FLIP_CONVERTERS}
    )
    subs = set(OptimizationProblemConverter.__subclasses__())
    for s in list(subs):
        subs.update(s.__subclasses__())

    concrete_public = {
        c
        for c in subs
        if not getattr(c, "__abstractmethods__", None) and not c.__name__.startswith("_")
    }
    unregistered = concrete_public - all_registered
    assert not unregistered, f"New converter(s) missing name propagation tests: {unregistered}"
