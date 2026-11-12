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

"""Tests for LinearExpression class."""

import numpy as np
import pytest
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems import LinearExpression, OptimizationProblem, Representation
from scipy.sparse import coo_array, csr_array, dok_array

REPRESENTATION_TO_TYPE = {
    Representation.DENSE: np.ndarray,
    Representation.COO: coo_array,
    Representation.DOK: dok_array,
    Representation.CSR: csr_array,
}


def MAKE_INSTANCE(coeffs, nvars=4, representation=Representation.DENSE):
    """Helper to create LinearExpression instance with a fake OptimizationProblem."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return LinearExpression(
        optimization_problem=op, coefficients=coeffs, representation=representation
    )


# ---------- representation property ----------
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_representation_property(representation):
    expr = MAKE_INSTANCE([1.0, 0.0, -2.5, 3.0], representation=representation)
    assert expr.representation is representation


# ---------- happy path ----------
@pytest.mark.parametrize("as_array", [list, np.array])
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_1d_and_correct_length_for_dense(as_array, representation):
    n = 4
    coeffs = as_array([1.0, 0.0, -2.5, 3.0])
    expr = MAKE_INSTANCE(coeffs, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n,)


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_sparse_row_vector_correct_length(representation):
    n = 4
    m = dok_array((n,))
    m[0] = 1.0
    m[2] = -2.5
    expr = MAKE_INSTANCE(m, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n,)


def test_csc_representation_raises():
    op = OptimizationProblem()
    op.binary_var_list(range(4))
    with pytest.raises(OptimizationError):
        LinearExpression(
            optimization_problem=op,
            coefficients=[1.0, 0.0, 0.0, 0.0],
            representation=Representation.CSC,
        )


def test_dense_to_array_returns_copy():
    expr = MAKE_INSTANCE([1.0, 2.0, 3.0, 4.0], nvars=4)
    arr = expr.to_array()
    arr[0] = 999.0
    assert expr.coefficients[0] == 1.0


# ---------- error cases ----------
@pytest.mark.parametrize(
    "bad_dense",
    [
        [1.0, 2.0, 3.0],  # list, short
        np.array([1.0, 2.0, 3.0]),  # ndarray, short
        [1, 2, 3, 4, 5],  # list, long
        np.array([1, 2, 3, 4, 5]),  # ndarray, long
    ],
)
def test_rejects_wrong_length_for_dense(bad_dense):
    n = 4
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE(bad_dense, nvars=n)
    msg = str(e.value).lower()
    assert "one-dimensional" in msg or "row vector" in msg
    assert "length" in msg and "match" in msg


def test_rejects_ndarray_not_1d():
    n = 4
    bad = np.array([[1, 2], [3, 4]])  # 2D
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE(bad, nvars=n)
    assert "one-dimensional" in str(e.value).lower()


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_dot_evaluation_consistency(representation):
    n = 4
    coeffs = [1, 2, 3, 4]
    expr = MAKE_INSTANCE(coeffs, nvars=n, representation=representation)

    x = np.array([1, 1, 1, 1])
    expected = sum(coeffs)  # 1+2+3+4=10
    assert expr.evaluate(x) == expected


# ---------- item access by name ----------
def test_getitem_by_name():
    op = OptimizationProblem()
    op.binary_var_list(range(4))
    expr = LinearExpression(optimization_problem=op, coefficients=[1.0, 2.0, 3.0, 4.0])
    assert expr["x1"] == 2.0
    assert expr[2] == 3.0


def test_setitem_by_name_and_index():
    op = OptimizationProblem()
    op.binary_var_list(range(4))
    expr = LinearExpression(optimization_problem=op, coefficients=[1.0, 2.0, 3.0, 4.0])
    expr["x1"] = 20.0
    expr[3] = 40.0
    assert expr[1] == 20.0
    assert expr[3] == 40.0


# ---------- more error cases ----------
def test_rejects_wrong_length_sparse():
    n = 4
    bad = dok_array((3,))
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE(bad, nvars=n)
    assert "one-dimensional" in str(e.value).lower()


def test_rejects_unsupported_type():
    n = 4
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE({1, 2, 3}, nvars=n)
    assert "unsupported format" in str(e.value).lower()


# ---------- dict construction / evaluation ----------
def test_dict_construction_by_name_and_index():
    op = OptimizationProblem()
    op.binary_var_list(range(4))
    expr = LinearExpression(optimization_problem=op, coefficients={"x0": 1.0, 2: -2.5})
    assert expr[0] == 1.0
    assert expr[2] == -2.5
    assert expr[1] == 0.0


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_evaluate_with_dict_input(representation):
    n = 4
    expr = MAKE_INSTANCE([1.0, 2.0, 3.0, 4.0], nvars=n, representation=representation)
    assert expr.evaluate({"x0": 1, "x1": 1, 2: 1, 3: 1}) == 10.0


# ---------- to_array / to_dict across representations ----------
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_to_array_across_representations(representation):
    n = 4
    expr = MAKE_INSTANCE([1.0, 0.0, -2.5, 3.0], nvars=n, representation=representation)
    np.testing.assert_array_equal(expr.to_array(), np.array([1.0, 0.0, -2.5, 3.0]))


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_to_dict_across_representations(representation):
    n = 4
    expr = MAKE_INSTANCE([1.0, 0.0, -2.5, 3.0], nvars=n, representation=representation)
    assert expr.to_dict() == {0: 1.0, 2: -2.5, 3: 3.0}


def test_to_dict_excludes_explicit_zeros():
    n = 4
    expr = MAKE_INSTANCE([1.0, 2.0, 3.0, 4.0], nvars=n, representation=Representation.CSR)
    expr[1] = 0.0
    assert expr.to_dict() == {0: 1.0, 2: 3.0, 3: 4.0}


def test_to_dict_use_name():
    op = OptimizationProblem()
    op.binary_var_list(range(4))
    expr = LinearExpression(optimization_problem=op, coefficients=[1.0, 0.0, 0.0, 3.0])
    assert expr.to_dict(use_name=True) == {"x0": 1.0, "x3": 3.0}


# ---------- bounds ----------
def test_bounds_computation():
    op = OptimizationProblem()
    op.binary_var_list(range(2))
    expr = LinearExpression(optimization_problem=op, coefficients=[2.0, -3.0])
    bounds = expr.bounds
    assert bounds.lowerbound == -3.0
    assert bounds.upperbound == 2.0


def test_bounds_raises_for_unbounded_variable():
    from qiskit_addon_opt_mapper.infinity import INFINITY

    op = OptimizationProblem()
    op.continuous_var(name="x0", lowerbound=-INFINITY, upperbound=INFINITY)
    expr = LinearExpression(optimization_problem=op, coefficients=[1.0])
    with pytest.raises(OptimizationError):
        _ = expr.bounds


# ---------- repr / str ----------
def test_evaluate_gradient_returns_array():
    expr = MAKE_INSTANCE([1.0, 2.0, -3.0, 4.0], nvars=4)
    np.testing.assert_array_equal(
        expr.evaluate_gradient(np.array([0, 0, 0, 0])), np.array([1.0, 2.0, -3.0, 4.0])
    )


def test_repr_and_str():
    expr = MAKE_INSTANCE([1.0, 0.0, -2.0, 0.0], nvars=4)
    assert "LinearExpression" in repr(expr)
    assert isinstance(str(expr), str)
