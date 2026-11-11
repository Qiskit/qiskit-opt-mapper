# This code is a Qiskit project.
#
# (C) Copyright IBM 2025, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import math

import numpy as np
import pytest

# --- adjust imports to your package layout ---
from qiskit_addon_opt_mapper import OptimizationError
from qiskit_addon_opt_mapper.problems import (
    ObjSense,
    OptimizationObjective,
    OptimizationProblem,
)

# ---------------------------------------------


@pytest.fixture
def op3():
    op = OptimizationProblem()
    op.integer_var(-10.0, 10.0, "x0")
    op.integer_var(-10.0, 10.0, "x1")
    op.integer_var(-10.0, 10.0, "x2")
    return op


@pytest.fixture
def obj(op3):
    """Unnamed objective on op3 with a single linear term."""
    return OptimizationObjective(optimization_problem=op3, linear={0: 1.0}, sense=ObjSense.MINIMIZE)


def test_objective_multiple_higher_orders_eval_and_grad(op3):
    """
    Objective with linear + quadratic + {cubic, quartic} blocks via higher_orders.
    Verify value and gradient at a test point.
    """
    linear = {0: 1.5, 1: -2.0}
    quadratic = {(2, 2): 3.0}
    higher_orders = {
        3: {(0, 0, 1): 2.0},  # 2 * x0^2 * x1
        4: {(0, 1, 1, 2): -0.5},  # -0.5 * x0 * x1^2 * x2
    }

    obj = OptimizationObjective(
        optimization_problem=op3,
        constant=0.0,
        linear=linear,
        quadratic=quadratic,
        higher_order=higher_orders,  # multiple blocks style
        sense=ObjSense.MINIMIZE,
    )

    x = [2.0, 1.0, -1.0]

    # Value:
    # linear:    1.5*2 + (-2)*1 = 1
    # quadratic: 3 * (-1)^2 = 3
    # cubic:     2 * (2^2 * 1) = 8
    # quartic:  -0.5 * (2 * 1^2 * -1) = +1
    # total: 1 + 3 + 8 + 1 = 13
    expected_val = 13.0
    assert math.isclose(obj.evaluate(x), expected_val, rel_tol=1e-12, abs_tol=1e-12)

    # Gradient:
    # linear grad = [1.5, -2.0, 0]
    # quadratic grad = [0, 0, 6*x2] = [0, 0, -6]
    # cubic grad (2*x0^2*x1):
    #   [8, 8, 0]
    # quartic grad (-0.5*x0*x1^2*x2):
    #   d/dx0 = -0.5 * x1^2 * x2 = 0.5
    #   d/dx1 = -0.5 * 2*x0*x1*x2 = 2.0
    #   d/dx2 = -0.5 * x0*x1^2 = -1.0
    expected_grad = np.array(
        [
            1.5 + 8.0 + 0.5,  # = 10.0
            -2.0 + 8.0 + 2.0,  # = 8.0
            0.0 - 6.0 - 1.0,  # = -7.0
        ]
    )
    g = obj.evaluate_gradient(x)
    assert np.allclose(g, expected_grad)


def test_objective_default_name_is_empty(obj):
    """Default name is the empty string (unnamed objective)."""
    assert obj.name == ""


def test_objective_repr_str_unnamed_unchanged(obj):
    """repr/str of an unnamed objective are unchanged by the optional name."""
    assert repr(obj) == "<OptimizationObjective: minimize x0>"
    assert str(obj) == "minimize x0"


def test_objective_name_in_constructor(op3):
    """A name passed to the constructor is stored and appears as ``'name'`` suffix."""
    named = OptimizationObjective(
        optimization_problem=op3, linear={0: 1.0}, sense=ObjSense.MINIMIZE, name="cost"
    )
    assert named.name == "cost"
    assert repr(named) == "<OptimizationObjective: minimize x0 'cost'>"
    assert str(named) == "minimize x0 'cost'"


def test_objective_name_no_setter(op3):
    """The name property has no setter: assignment raises AttributeError."""
    named = OptimizationObjective(
        optimization_problem=op3, linear={0: 1.0}, sense=ObjSense.MINIMIZE, name="cost"
    )
    with pytest.raises(AttributeError):
        named.name = "other"  # type: ignore[misc]


def test_obj_sense_convert_min_variants():
    """ObjSense.convert maps all MINIMIZE spellings correctly."""
    assert ObjSense.convert("min") is ObjSense.MINIMIZE
    assert ObjSense.convert("MIN") is ObjSense.MINIMIZE
    assert ObjSense.convert("Minimize") is ObjSense.MINIMIZE
    assert ObjSense.convert("minimize") is ObjSense.MINIMIZE


def test_obj_sense_convert_max_variants():
    """ObjSense.convert maps all MAXIMIZE spellings correctly."""
    assert ObjSense.convert("max") is ObjSense.MAXIMIZE
    assert ObjSense.convert("MAXIMIZE") is ObjSense.MAXIMIZE
    assert ObjSense.convert("Maximize") is ObjSense.MAXIMIZE
    assert ObjSense.convert("maximize") is ObjSense.MAXIMIZE


def test_obj_sense_convert_idempotent():
    """ObjSense.convert is idempotent on ObjSense members."""
    assert ObjSense.convert(ObjSense.MINIMIZE) is ObjSense.MINIMIZE
    assert ObjSense.convert(ObjSense.MAXIMIZE) is ObjSense.MAXIMIZE


def test_obj_sense_convert_invalid_uppercases_message():
    """An invalid string raises OptimizationError with the *uppercased* input in the message."""
    with pytest.raises(OptimizationError, match="MINIMUM"):
        ObjSense.convert("Minimum")


def test_obj_sense_convert_non_string_raises():
    """A non-string, non-ObjSense argument raises OptimizationError (not AttributeError)."""
    with pytest.raises(OptimizationError):
        ObjSense.convert(42)  # type: ignore[arg-type]


def test_obj_sense_convert_via_sense_alias():
    """OptimizationObjective.Sense.convert is reachable, mirroring Constraint.Sense.convert."""
    assert OptimizationObjective.Sense.convert("max") is ObjSense.MAXIMIZE


def test_import_module_does_not_raise():
    """Importing the module must not raise (guards the quoted annotation on ObjSense.convert)."""
    import importlib

    import qiskit_addon_opt_mapper.problems.optimization_objective as mod

    importlib.reload(mod)


def test_minimize_name_parameter():
    """minimize(name=...) sets the name and it surfaces in repr/str of the problem."""
    p = OptimizationProblem()
    p.binary_var("x")
    p.minimize(linear={"x": 1}, name="cost")
    assert p.objective.name == "cost"
    assert "minimize x 'cost'" in repr(p)
    assert "minimize x 'cost'" in str(p)


def test_maximize_name_parameter():
    """maximize(name=...) sets the name and it surfaces in repr/str of the problem."""
    p = OptimizationProblem()
    p.binary_var("x")
    p.maximize(linear={"x": 1}, name="profit")
    assert p.objective.name == "profit"
    assert "maximize x 'profit'" in repr(p)
    assert "maximize x 'profit'" in str(p)


def test_minimize_positional_args_unchanged(op3):
    """Positional-argument call sites (converter style) are unaffected by the new name= param."""
    op3.minimize(0.0, {0: 1.0}, {}, None)
    assert op3.objective.name == ""
    assert str(op3.objective) == "minimize x0"


def test_render_public_method(op3):
    """render() is public and produces the same output as str() for unnamed and named objectives."""
    op3.minimize(linear={0: 1.0})
    assert op3.objective.render() == str(op3.objective)

    op3.minimize(linear={0: 1.0}, name="cost")
    assert op3.objective.render() == "minimize x0 'cost'"


def test_non_printable_name_warns():
    """minimize/maximize emit a UserWarning for non-printable objective names."""
    p = OptimizationProblem()
    p.binary_var("x")
    with pytest.warns(UserWarning, match="Objective name is not printable"):
        p.minimize(linear={"x": 1}, name="bad\nname")
    with pytest.warns(UserWarning, match="Objective name is not printable"):
        p.maximize(linear={"x": 1}, name="bad\nname")
