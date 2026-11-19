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

"""Quadratic expression interface."""

import warnings
from collections import defaultdict
from typing import Any, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_array, csc_array, csr_array, dok_array, sparray, spmatrix, tril, triu

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .linear_expression import ExpressionBounds, Representation
from .optimization_problem_element import OptimizationProblemElement

_SPARSE_2D_CONSTRUCTORS: dict[Representation, type[sparray]] = {
    Representation.COO: coo_array,
    Representation.DOK: dok_array,
    Representation.CSR: csr_array,
    Representation.CSC: csc_array,
}


class QuadraticExpression(OptimizationProblemElement):
    """Representation of a quadratic expression by its coefficients."""

    def __init__(
        self,
        optimization_problem: Any,
        coefficients: (
            ndarray
            | sparray
            | spmatrix
            | list[list[float]]
            | dict[tuple[str | int, str | int], float]
        ),
        representation: Representation = Representation.DENSE,
    ) -> None:
        """Creates a new quadratic expression.

        The quadratic expression can be defined via an array, a list, a sparse matrix, or a
        dictionary that uses variable names or indices as keys and stores the values internally
        according to ``representation``. Values are stored in a compressed way, i.e., values at
        symmetric positions are summed up in the upper triangle. For example,
        {(0, 1): 1, (1, 0): 2} -> {(0, 1): 3}.

        Args:
            optimization_problem: The parent OptimizationProblem.
            coefficients: The (sparse) representation of the coefficients.
            representation: The internal storage representation. Fixed for the life of the object.

        """
        super().__init__(optimization_problem)
        self._representation = representation
        self._coefficients: ndarray | sparray
        self.coefficients = coefficients

    @property
    def representation(self) -> Representation:
        """Returns the internal storage representation of the quadratic expression."""
        return self._representation

    def __getitem__(self, key: tuple[str | int, str | int]) -> float:
        """Returns the coefficient where i, j can be a variable names or indices.

        On a ``Representation.COO`` expression, every call converts the whole matrix
        to CSR, making repeated element access O(nnz) per call. Prefer ``DOK``, ``CSR``,
        or ``CSC`` for workloads that do many scalar reads.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.


        Returns:
            The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        if isinstance(j, str):
            j = self.optimization_problem.variables_index[j]
        coefficients = self.coefficients
        if isinstance(coefficients, coo_array):
            coefficients = coefficients.tocsr()
        return float(coefficients[min(i, j), max(i, j)])

    def __setitem__(self, key: tuple[str | int, str | int], value: float) -> None:
        """Sets the coefficient where i, j can be a variable names or indices.

        On a ``Representation.COO`` expression, every call converts the whole matrix to DOK
        and back. On ``Representation.CSR``/``Representation.CSC``, inserting a coefficient at
        a previously-unset position raises scipy's ``SparseEfficiencyWarning``. Prefer ``DOK``
        for workloads that do many scalar writes.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.
            value: The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        if isinstance(j, str):
            j = self.optimization_problem.variables_index[j]
        if isinstance(self._coefficients, coo_array):
            warnings.warn(
                "Setting a single coefficient on a COO-represented QuadraticExpression "
                "converts the whole matrix to DOK and back, which is inefficient for "
                "repeated assignment; construct the expression with Representation.DOK "
                "instead if you need to set many individual coefficients.",
                stacklevel=2,
            )
            dok = self._coefficients.todok()
            dok[min(i, j), max(i, j)] = value
            self._coefficients = coo_array(dok)
            return
        self._coefficients[min(i, j), max(i, j)] = value
        if isinstance(self._coefficients, csr_array | csc_array):
            # Unlike DOK, CSR/CSC keep an explicit stored zero after this assignment;
            # drop it so to_dict()/nnz/etc. agree with the other representations.
            self._coefficients.eliminate_zeros()

    def _coeffs_to_representation(
        self,
        coefficients: (
            ndarray
            | sparray
            | spmatrix
            | list[list[float]]
            | dict[tuple[str | int, str | int], float]
        ),
    ) -> ndarray | sparray:
        """Maps given coefficients to this expression's storage representation.

        Args:
            coefficients: The coefficients to be mapped.


        Returns:
            The given coefficients cast to this expression's representation.

        Raises:
            OptimizationError: if coefficients are given in unsupported format.
        """
        n = self.optimization_problem.get_num_vars()

        def _dict_to_dok(coeffs: dict[tuple[int, int], float]) -> dok_array:
            if coeffs:
                rows, cols, values = zip(*((i, j, v) for (i, j), v in coeffs.items()), strict=True)
                ret = dok_array(coo_array((values, (rows, cols)), shape=(n, n)).todok())
            else:
                ret = dok_array((n, n))
            return ret

        if isinstance(coefficients, list):
            coeffs = defaultdict(float)
            if len(coefficients) != n:
                raise OptimizationError(
                    f"The coefficient list for the quadratic expression must be a list of {n} "
                    f"lists, each of length {n}, matching the number of variables."
                )
            for i, row in enumerate(coefficients):
                if len(row) != n:
                    raise OptimizationError(
                        f"Each inner list for the quadratic expression must be of length {n}"
                        f"matching the number of variables."
                    )
                for j, value in enumerate(row):
                    coeffs[i, j] = value
            folded = self._fold_sparse(_dict_to_dok(coeffs))
        elif isinstance(coefficients, ndarray):
            if coefficients.ndim != 2 or coefficients.shape != (n, n):
                raise OptimizationError(
                    f"The coefficient numpy array for the quadratic expression must be a (n, n) "
                    f"matrix with shape ({n}, {n}) matching the number of variables."
                )
            folded = self._fold_dense(coefficients)
        elif isinstance(coefficients, sparray | spmatrix):
            if coefficients.ndim != 2 or coefficients.shape != (n, n):
                raise OptimizationError(
                    f"The coefficient sparse matrix for the quadratic expression must be a (n, n) "
                    f"matrix with shape ({n}, {n}) matching the number of variables."
                )
            folded = self._fold_sparse(coefficients)
        elif isinstance(coefficients, dict):
            coeffs = defaultdict(float)
            for (i, j), value in coefficients.items():  # type: ignore
                i_idx = self.optimization_problem.variables_index[i] if isinstance(i, str) else i
                j_idx = self.optimization_problem.variables_index[j] if isinstance(j, str) else j
                if i_idx > j_idx:
                    i_idx, j_idx = j_idx, i_idx
                coeffs[i_idx, j_idx] += value
            folded = self._fold_sparse(_dict_to_dok(coeffs))
        else:
            raise OptimizationError(f"Unsupported format for coefficients: {coefficients}")

        return self._to_representation(folded)

    @staticmethod
    def _fold_dense(mat: ndarray) -> ndarray:
        """Folds a dense (n, n) array into upper-triangle-compressed form."""
        folded = np.triu(mat) + np.tril(mat, -1).T
        return np.ascontiguousarray(folded, dtype=np.float64)

    @staticmethod
    def _fold_sparse(mat: sparray | spmatrix) -> sparray:
        """Folds a sparse (n, n) matrix into upper-triangle-compressed form.

        Works directly on COO, avoiding a DOK round-trip for formats (e.g. CSR/CSC, or a
        dense array converted upstream) where that round-trip would dominate the cost.
        """
        coo = coo_array(mat)
        lower = tril(coo, -1, format="coo")
        return cast(sparray, triu(coo, format="coo") + lower.transpose())

    def _to_representation(self, folded: ndarray | sparray) -> ndarray | sparray:
        """Casts an already upper-triangle-folded matrix to this expression's representation."""
        if self._representation == Representation.DENSE:
            if isinstance(folded, ndarray):
                return folded
            return np.ascontiguousarray(folded.toarray(), dtype=np.float64)
        return _SPARSE_2D_CONSTRUCTORS[self._representation](folded)

    @staticmethod
    def _symmetric_matrix(mat: ndarray | sparray) -> ndarray | sparray:
        """Returns the upper-triangle-compressed ``mat`` expanded into symmetric form."""
        if isinstance(mat, ndarray):
            upper = np.triu(mat, 1) / 2
            return mat + upper.T - upper
        upper = triu(mat, 1, format="dok") / 2
        # `asformat` is necessary because subtraction results in other format
        return (mat + upper.transpose() - upper).asformat(mat.format)

    @property
    def coefficients(self) -> ndarray | sparray:
        """Returns the coefficients of the quadratic expression.

        Returns:
            The coefficients of the quadratic expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self,
        coefficients: (
            ndarray
            | sparray
            | spmatrix
            | list[list[float]]
            | dict[tuple[str | int, str | int], float]
        ),
    ) -> None:
        """Sets the coefficients of the quadratic expression.

        Args:
            coefficients: The coefficients of the quadratic expression.
        """
        self._coefficients = self._coeffs_to_representation(coefficients)

    def to_array(self, symmetric: bool = False) -> ndarray:
        """Returns the coefficients of the quadratic expression as array.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.


        Returns:
            An array with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        if isinstance(coeffs, ndarray):
            return cast(ndarray, coeffs.copy())
        return cast(ndarray, coeffs.toarray())

    def to_dict(
        self, symmetric: bool = False, use_name: bool = False
    ) -> dict[tuple[int, int] | tuple[str, str], float]:
        """Returns the coefficients of the quadratic expression as dictionary.

        Either using tuples of variable names or indices as keys.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.
            use_name: Determines whether to use index or names to refer to variables.


        Returns:
            An dictionary with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        if isinstance(coeffs, ndarray):
            rows, cols = np.nonzero(coeffs)
            items = ((int(i), int(j), float(coeffs[i, j])) for i, j in zip(rows, cols, strict=True))
        else:
            coo = coeffs.tocoo()
            items = (
                (int(i), int(j), float(v))
                for i, j, v in zip(coo.coords[0], coo.coords[1], coo.data, strict=True)
            )

        if use_name:
            variables = self.optimization_problem.variables
            return {(variables[i].name, variables[j].name): v for i, j, v in items}
        return {(i, j): v for i, j, v in items}

    def evaluate(self, x: ndarray | list | dict[str | int, float]) -> float:
        """Evaluate the quadratic expression for given variables: x * Q * x.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute x * Q * x for the quadratic expression
        val = x @ self.coefficients @ x

        # return the result
        return float(val)

    def evaluate_gradient(self, x: ndarray | list | dict[str | int, float]) -> ndarray:
        """Evaluate the gradient of the quadratic expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the gradient quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute (Q' + Q) * x as Q @ x + x @ Q, which never forms the (n, n) matrix Q + Q'
        mat = self._coefficients
        val = mat @ x + x @ mat

        # return the result
        return cast(ndarray, val)

    def _cast_as_array(self, x: ndarray | list | dict[str | int, float]) -> np.ndarray:
        """Converts input to an array if it is a dictionary or list."""
        if isinstance(x, dict):
            x_aux = np.zeros(self.optimization_problem.get_num_vars())
            for i, v in x.items():
                if isinstance(i, str):
                    i = self.optimization_problem.variables_index[i]
                x_aux[i] = v
            x = x_aux
        if isinstance(x, list):
            x = np.array(x)
        return x

    @property
    def bounds(self) -> ExpressionBounds:
        """Returns the lower bound and the upper bound of the quadratic expression.

        Returns:
            The lower bound and the upper bound of the quadratic expression

        Raises:
            OptimizationError: if the quadratic expression contains any unbounded variable
        """
        l_b = u_b = 0.0
        for (ind1, ind2), coeff in self.to_dict().items():
            x = self.optimization_problem.get_variable(ind1)
            if x.lowerbound == -INFINITY or x.upperbound == INFINITY:
                raise OptimizationError(
                    f"Quadratic expression contains an unbounded variable: {x.name}"
                )
            y = self.optimization_problem.get_variable(ind2)
            if y.lowerbound == -INFINITY or y.upperbound == INFINITY:
                raise OptimizationError(
                    f"Quadratic expression contains an unbounded variable: {y.name}"
                )
            lst = []
            if ind1 == ind2:
                if x.lowerbound * x.upperbound <= 0.0:
                    # lower bound and upper bound have different signs
                    lst.append(0.0)
                lst.extend([x.lowerbound**2, x.upperbound**2])
            else:
                lst.extend(
                    [
                        x.lowerbound * y.lowerbound,
                        x.lowerbound * y.upperbound,
                        x.upperbound * y.lowerbound,
                        x.upperbound * y.upperbound,
                    ]
                )
            lst2 = [coeff * val for val in lst]
            l_b += min(lst2)
            u_b += max(lst2)
        return ExpressionBounds(lowerbound=l_b, upperbound=u_b)

    def __repr__(self):
        """Repr. for QuadraticExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        return f"<{self.__class__.__name__}: {expr2str(quadratic=self, truncate=DEFAULT_TRUNCATE)}>"

    def __str__(self):
        """Repr. for QuadraticExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        return f"{expr2str(quadratic=self)}"
