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

"""Tests for QuadraticExpression class."""

import numpy as np
import pytest
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.infinity import INFINITY
from qiskit_addon_opt_mapper.problems import OptimizationProblem, QuadraticExpression
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    dok_array,
    dok_matrix,  # legacy _matrix input acceptance
)


def MAKE_INSTANCE(coeffs, nvars=4):
    """Helper to create LinearExpression instance with a fake OptimizationProblem."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return QuadraticExpression(optimization_problem=op, coefficients=coeffs)


# ---------------- happy path ----------------
def test_result_is_dok_matrix_and_square():
    n = 2
    dense = [[1.0, 2.0], [2.0, 5.0]]
    expr = MAKE_INSTANCE(dense, nvars=n)
    assert isinstance(expr.coefficients, dok_array)
    assert expr.coefficients.shape == (n, n)


def test_accepts_list_of_lists_n_by_n_and_normalizes_to_dok():
    n = 4
    dense = [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.5, 0.0],
        [2.0, 0.0, 0.0, 5.0],
    ]
    expr = MAKE_INSTANCE(dense, nvars=n)
    assert isinstance(expr.coefficients, dok_array)
    assert expr.coefficients.shape == (n, n)
    # Coefficients are stored in a upper-triangular manner
    assert expr.coefficients[0, 0] == 1.0
    assert expr.coefficients[0, 3] == 4.0
    assert expr.coefficients[2, 2] == 4.5
    assert expr.coefficients[1, 2] == 0


def test_accepts_ndarray_n_by_n_and_normalizes_to_dok():
    n = 3
    arr = np.array([[1, 2, 0], [0, 3, 4], [0, 0, 5]], dtype=float)
    expr = MAKE_INSTANCE(arr, nvars=n)
    assert isinstance(expr.coefficients, dok_array)
    assert expr.coefficients.shape == (n, n)
    assert expr.coefficients[0, 1] == 2
    assert expr.coefficients[1, 2] == 4
    assert expr.coefficients[2, 2] == 5


@pytest.mark.parametrize("sparse_fmt", [dok_matrix, csr_matrix, csc_matrix])
def test_accepts_sparse_n_by_n_and_normalizes_to_dok(sparse_fmt):
    n = 3
    sp = sparse_fmt((n, n))
    sp[0, 0] = 1.0
    sp[1, 1] = 2.0
    sp[2, 2] = 3.0

    expr = MAKE_INSTANCE(sp, nvars=n)
    assert isinstance(expr.coefficients, dok_array)
    assert expr.coefficients.shape == (n, n)
    assert expr.coefficients[1, 1] == 2.0


# ---------------- error/exception cases ----------------


def test_rejects_list_outer_length_mismatch():
    n = 4
    # outer list length wrong
    bad = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_list_inner_length_mismatch():
    n = 4
    bad = [
        [0, 0, 0, 0],
        [0, 0, 0],  # short
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_ndarray_wrong_shape_rectangular():
    n = 4
    bad = np.zeros((n, n + 1))
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_ndarray_not_2d():
    n = 3
    bad = np.array([1, 2, 3])  # 1D
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_unsupported_format():
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE("not a valid format", nvars=2)


def test_accepts_empty_dict():
    n = 3
    expr = MAKE_INSTANCE({}, nvars=n)
    assert isinstance(expr.coefficients, dok_array)
    assert expr.coefficients.shape == (n, n)
    assert expr.coefficients.nnz == 0


def test_accepts_dict_with_index_keys_and_sums_symmetric_entries():
    n = 3
    coeffs = {(0, 1): 1.0, (1, 0): 2.0, (2, 2): 5.0}
    expr = MAKE_INSTANCE(coeffs, nvars=n)
    assert expr.coefficients[0, 1] == 3.0
    assert expr.coefficients[2, 2] == 5.0


def test_accepts_dict_with_name_keys():
    op = OptimizationProblem()
    op.binary_var_list(range(3))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={(names[0], names[1]): 4.0})
    assert expr.coefficients[0, 1] == 4.0


# ---------------- getitem / setitem ----------------


def test_getitem_setitem_with_indices():
    expr = MAKE_INSTANCE({}, nvars=3)
    expr[0, 1] = 7.0
    assert expr[0, 1] == 7.0
    # symmetric access returns the upper-triangle value
    assert expr[1, 0] == 7.0


def test_getitem_setitem_with_names():
    op = OptimizationProblem()
    op.binary_var_list(range(3))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={})
    expr[names[0], names[1]] = 9.0
    assert expr[names[0], names[1]] == 9.0
    assert expr[names[1], names[0]] == 9.0


# ---------------- to_array / to_dict ----------------


def test_to_array_default_and_symmetric():
    expr = MAKE_INSTANCE({(0, 1): 2.0}, nvars=2)
    arr = expr.to_array()
    assert arr[0, 1] == 2.0
    assert arr[1, 0] == 0.0

    sym = expr.to_array(symmetric=True)
    assert sym[0, 1] == 1.0
    assert sym[1, 0] == 1.0


def test_to_dict_indices_and_names_symmetric():
    op = OptimizationProblem()
    op.binary_var_list(range(2))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 1): 2.0})

    d_idx = expr.to_dict()
    assert d_idx == {(0, 1): 2.0}

    d_sym = expr.to_dict(symmetric=True)
    assert d_sym[(0, 1)] == 1.0
    assert d_sym[(1, 0)] == 1.0

    d_name = expr.to_dict(use_name=True)
    assert d_name == {(names[0], names[1]): 2.0}


# ---------------- evaluate / evaluate_gradient ----------------


def test_evaluate_with_ndarray_list_and_dict():
    expr = MAKE_INSTANCE({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2)
    x_arr = np.array([1.0, 1.0])
    x_list = [1.0, 1.0]
    x_dict = {0: 1.0, 1: 1.0}

    expected = 1.0 + 2.0 + 3.0
    assert expr.evaluate(x_arr) == expected
    assert expr.evaluate(x_list) == expected
    assert expr.evaluate(x_dict) == expected


def test_evaluate_with_name_keyed_dict():
    op = OptimizationProblem()
    op.binary_var_list(range(2))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 1): 2.0})
    assert expr.evaluate({names[0]: 1.0, names[1]: 1.0}) == 2.0


def test_evaluate_gradient():
    expr = MAKE_INSTANCE({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2)
    grad = expr.evaluate_gradient([1.0, 1.0])
    # Q + Q^T applied to [1, 1]
    assert list(grad) == [4.0, 8.0]


# ---------------- bounds ----------------


def test_bounds_diagonal_term_same_sign():
    op = OptimizationProblem()
    op.continuous_var_list(2, lowerbound=1.0, upperbound=3.0)
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 0): 2.0})
    bounds = expr.bounds
    assert bounds.lowerbound == 2.0
    assert bounds.upperbound == 18.0


def test_bounds_diagonal_term_sign_change_includes_zero():
    op = OptimizationProblem()
    op.continuous_var_list(2, lowerbound=-2.0, upperbound=3.0)
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 0): 1.0})
    bounds = expr.bounds
    assert bounds.lowerbound == 0.0
    assert bounds.upperbound == 9.0


def test_bounds_off_diagonal_term():
    op = OptimizationProblem()
    op.continuous_var_list(2, lowerbound=0.0, upperbound=2.0)
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 1): 1.0})
    bounds = expr.bounds
    assert bounds.lowerbound == 0.0
    assert bounds.upperbound == 4.0


def test_bounds_raises_on_unbounded_variable():
    op = OptimizationProblem()
    op.continuous_var_list(2)  # default upperbound is INFINITY
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 1): 1.0})
    with pytest.raises(OptimizationError):
        _ = expr.bounds


def test_bounds_raises_on_unbounded_second_variable():
    op = OptimizationProblem()
    op.continuous_var(lowerbound=0.0, upperbound=1.0)
    op.continuous_var(lowerbound=0.0, upperbound=INFINITY)
    expr = QuadraticExpression(optimization_problem=op, coefficients={(0, 1): 1.0})
    with pytest.raises(OptimizationError):
        _ = expr.bounds


# ---------------- repr / str ----------------


def test_repr_and_str():
    expr = MAKE_INSTANCE({(0, 1): 2.0}, nvars=2)
    assert "QuadraticExpression" in repr(expr)
    assert str(expr) != ""
