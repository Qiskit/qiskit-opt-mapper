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

import warnings

import numpy as np
import pytest
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.infinity import INFINITY
from qiskit_addon_opt_mapper.problems import (
    OptimizationProblem,
    QuadraticExpression,
    Representation,
)
from scipy.sparse import (
    coo_array,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dok_array,
    dok_matrix,
)

REPRESENTATION_TO_TYPE = {
    Representation.DENSE: np.ndarray,
    Representation.COO: coo_array,
    Representation.DOK: dok_array,
    Representation.CSR: csr_array,
    Representation.CSC: csc_array,
}


def MAKE_INSTANCE(coeffs, nvars=4, representation=Representation.DENSE):
    """Helper to create QuadraticExpression instance with a fake OptimizationProblem."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return QuadraticExpression(
        optimization_problem=op, coefficients=coeffs, representation=representation
    )


# ---------------- representation property ----------------
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_representation_property(representation):
    dense = [[1.0, 2.0], [2.0, 5.0]]
    expr = MAKE_INSTANCE(dense, nvars=2, representation=representation)
    assert expr.representation is representation


# ---------------- default representation selection ----------------
def MAKE_DEFAULT_INSTANCE(coeffs, nvars=4):
    """Helper to create a QuadraticExpression without specifying `representation`."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return QuadraticExpression(optimization_problem=op, coefficients=coeffs)


def test_default_representation_is_dense_when_nonempty_and_small():
    expr = MAKE_DEFAULT_INSTANCE({(0, 1): 2.0}, nvars=4)
    assert expr.representation is Representation.DENSE
    assert isinstance(expr.coefficients, np.ndarray)


@pytest.mark.parametrize("nvars", [4, 600])
def test_default_representation_is_sparse_when_empty_regardless_of_size(nvars):
    expr = MAKE_DEFAULT_INSTANCE({}, nvars=nvars)
    assert expr.representation is Representation.CSR
    assert expr.coefficients.shape == (nvars, nvars)
    assert expr.to_dict() == {}


def test_default_representation_empty_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MAKE_DEFAULT_INSTANCE({}, nvars=600)


def test_default_representation_large_n_warns_and_stays_dense():
    with pytest.warns(UserWarning, match="explicit `representation`"):
        expr = MAKE_DEFAULT_INSTANCE({(0, 1): 2.0}, nvars=600)
    assert expr.representation is Representation.DENSE


def test_default_representation_small_n_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MAKE_DEFAULT_INSTANCE({(0, 1): 2.0}, nvars=500)


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_explicit_representation_overrides_default_even_when_empty(representation):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expr = MAKE_INSTANCE({}, nvars=600, representation=representation)
    assert expr.representation is representation
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])


# ---------------- happy path ----------------
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_result_matches_representation_and_square(representation):
    n = 2
    dense = [[1.0, 2.0], [2.0, 5.0]]
    expr = MAKE_INSTANCE(dense, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n, n)


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_list_of_lists_n_by_n_and_normalizes(representation):
    n = 4
    dense = [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.5, 0.0],
        [2.0, 0.0, 0.0, 5.0],
    ]
    expr = MAKE_INSTANCE(dense, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n, n)
    # Coefficients are stored in a upper-triangular manner
    assert expr.to_dict() == {(0, 0): 1.0, (0, 3): 4.0, (1, 1): 3.0, (2, 2): 4.5, (3, 3): 5.0}


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_ndarray_n_by_n_and_normalizes(representation):
    n = 3
    arr = np.array([[1, 2, 0], [0, 3, 4], [0, 0, 5]], dtype=float)
    expr = MAKE_INSTANCE(arr, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n, n)
    assert expr.to_dict() == {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0, (1, 2): 4.0, (2, 2): 5.0}


@pytest.mark.parametrize("sparse_fmt", [dok_matrix, csr_matrix, csc_matrix])
@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_sparse_n_by_n_and_normalizes(sparse_fmt, representation):
    n = 3
    sp = sparse_fmt((n, n))
    sp[0, 0] = 1.0
    sp[1, 1] = 2.0
    sp[2, 2] = 3.0

    expr = MAKE_INSTANCE(sp, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n, n)
    assert expr.to_dict() == {(0, 0): 1.0, (1, 1): 2.0, (2, 2): 3.0}


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_coo_with_duplicate_and_mirrored_entries(representation):
    n = 3
    # duplicate (0, 1) entries and a mirrored (1, 0) entry must all be summed into [0, 1]
    coo = coo_array(([2.0, 3.0, 4.0], ([0, 0, 1], [1, 1, 0])), shape=(n, n))

    expr = MAKE_INSTANCE(coo, nvars=n, representation=representation)
    assert expr.to_dict() == {(0, 1): 9.0}


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


def test_rejects_sparse_wrong_shape():
    n = 4
    bad = dok_array((n, n + 1))
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_unsupported_format():
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE("not a valid format", nvars=2)


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_empty_dict(representation):
    n = 3
    expr = MAKE_INSTANCE({}, nvars=n, representation=representation)
    assert isinstance(expr.coefficients, REPRESENTATION_TO_TYPE[representation])
    assert expr.coefficients.shape == (n, n)
    assert expr.to_dict() == {}


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_accepts_dict_with_index_keys_and_sums_symmetric_entries(representation):
    n = 3
    coeffs = {(0, 1): 1.0, (1, 0): 2.0, (2, 2): 5.0}
    expr = MAKE_INSTANCE(coeffs, nvars=n, representation=representation)
    assert expr.to_dict() == {(0, 1): 3.0, (2, 2): 5.0}
    if representation is Representation.COO:
        # Scalar access on COO is unsafe; whether it raises depends on the scipy
        # version installed, so it is not asserted here. See `Representation`.
        return
    assert expr[0, 1] == 3.0
    assert expr[2, 2] == 5.0


def test_triangle_fold_handles_i_greater_than_j():
    # Regression test: a dict entry with i > j must fold into the upper triangle at (j, i),
    # not silently no-op and leave the value stranded at (i, j) in the lower triangle.
    n = 3
    expr = MAKE_INSTANCE({(2, 0): 7.0}, nvars=n)
    assert expr[0, 2] == 7.0
    assert expr.to_dict() == {(0, 2): 7.0}


def test_accepts_dict_with_name_keys():
    op = OptimizationProblem()
    op.binary_var_list(range(3))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={(names[0], names[1]): 4.0})
    assert expr[0, 1] == 4.0


# ---------------- getitem / setitem ----------------


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
@pytest.mark.parametrize(
    "representation", [r for r in REPRESENTATION_TO_TYPE if r is not Representation.COO]
)
def test_getitem_setitem_with_indices(representation):
    expr = MAKE_INSTANCE({}, nvars=3, representation=representation)
    expr[0, 1] = 7.0
    assert expr[0, 1] == 7.0
    # symmetric access returns the upper-triangle value
    assert expr[1, 0] == 7.0




@pytest.mark.parametrize("representation", [Representation.CSR, Representation.CSC])
def test_setitem_zero_drops_explicit_entry_on_csr_csc(representation):
    expr = MAKE_INSTANCE({(0, 1): 5.0}, nvars=3, representation=representation)
    expr[0, 1] = 0.0
    assert expr.coefficients.nnz == 0
    assert expr.to_dict() == {}


def test_getitem_setitem_with_names():
    op = OptimizationProblem()
    op.binary_var_list(range(3))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(optimization_problem=op, coefficients={})
    expr[names[0], names[1]] = 9.0
    assert expr[names[0], names[1]] == 9.0
    assert expr[names[1], names[0]] == 9.0


# ---------------- to_array / to_dict ----------------


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_to_array_default_and_symmetric(representation):
    expr = MAKE_INSTANCE({(0, 1): 2.0}, nvars=2, representation=representation)
    arr = expr.to_array()
    assert arr[0, 1] == 2.0
    assert arr[1, 0] == 0.0

    sym = expr.to_array(symmetric=True)
    assert sym[0, 1] == 1.0
    assert sym[1, 0] == 1.0


def test_dense_to_array_returns_copy():
    expr = MAKE_INSTANCE({(0, 1): 2.0}, nvars=2)
    arr = expr.to_array()
    arr[0, 1] = 999.0
    assert expr[0, 1] == 2.0


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_to_dict_indices_and_names_symmetric(representation):
    op = OptimizationProblem()
    op.binary_var_list(range(2))
    names = [v.name for v in op.variables]
    expr = QuadraticExpression(
        optimization_problem=op, coefficients={(0, 1): 2.0}, representation=representation
    )

    d_idx = expr.to_dict()
    assert d_idx == {(0, 1): 2.0}

    d_sym = expr.to_dict(symmetric=True)
    assert d_sym[(0, 1)] == 1.0
    assert d_sym[(1, 0)] == 1.0

    d_name = expr.to_dict(use_name=True)
    assert d_name == {(names[0], names[1]): 2.0}


# ---------------- evaluate / evaluate_gradient ----------------


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_evaluate_with_ndarray_list_and_dict(representation):
    expr = MAKE_INSTANCE(
        {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2, representation=representation
    )
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


@pytest.mark.parametrize("representation", list(REPRESENTATION_TO_TYPE))
def test_evaluate_gradient(representation):
    expr = MAKE_INSTANCE(
        {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2, representation=representation
    )
    grad = expr.evaluate_gradient([1.0, 1.0])
    # Q + Q^T applied to [1, 1]
    assert list(grad) == [4.0, 8.0]


def test_evaluate_gradient_reflects_mutation():
    expr = MAKE_INSTANCE({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2)
    first = expr.evaluate_gradient([1.0, 1.0])
    assert list(first) == [4.0, 8.0]

    expr[0, 0] = 10.0
    second = expr.evaluate_gradient([1.0, 1.0])
    assert list(second) == [22.0, 8.0]


def test_evaluate_gradient_reflects_coefficients_mutated_in_place():
    expr = MAKE_INSTANCE({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, nvars=2)
    first = expr.evaluate_gradient([1.0, 1.0])
    assert list(first) == [4.0, 8.0]

    expr.coefficients[0, 0] = 10.0
    second = expr.evaluate_gradient([1.0, 1.0])
    assert list(second) == [22.0, 8.0]


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
