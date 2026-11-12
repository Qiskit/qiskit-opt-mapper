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

"""Linear expression interface."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_array, csr_array, dok_array, sparray, spmatrix

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .optimization_problem_element import OptimizationProblemElement


@dataclass
class ExpressionBounds:
    """Lower bound and upper bound of a linear expression or a quadratic expression."""

    lowerbound: float
    """Lower bound"""

    upperbound: float
    """Upper bound"""


class Representation(Enum):
    """The internal storage representation of an expression's coefficients."""

    DENSE = "dense"
    COO = "coo"
    DOK = "dok"
    CSR = "csr"
    CSC = "csc"


_SPARSE_1D_CONSTRUCTORS: dict[Representation, type] = {
    Representation.COO: coo_array,
    Representation.DOK: dok_array,
    Representation.CSR: csr_array,
}


class LinearExpression(OptimizationProblemElement):
    """Representation of a linear expression by its coefficients."""

    def __init__(
        self,
        optimization_problem: Any,
        coefficients: ndarray | sparray | spmatrix | list[float] | dict[int | str, float],
        representation: Representation = Representation.DENSE,
    ) -> None:
        """Creates a new linear expression.

        The linear expression can be defined via an array, a list, a sparse matrix, or a dictionary
        that uses variable names or indices as keys and stores the values internally according to
        ``representation``.

        Args:
            optimization_problem: The parent OptimizationProblem.
            coefficients: The (sparse) representation of the coefficients.
            representation: The internal storage representation. Fixed for the life of the object.

        Raises:
            OptimizationError: if ``representation`` is ``Representation.CSC``, which has no 1-D
                form.
        """
        super().__init__(optimization_problem)
        if representation == Representation.CSC:
            raise OptimizationError(
                "Representation.CSC is not supported for LinearExpression: scipy's csc_array has "
                "no 1-D form."
            )
        self._representation = representation
        self.coefficients = coefficients

    @property
    def representation(self) -> Representation:
        """Returns the internal storage representation of the linear expression."""
        return self._representation

    def __getitem__(self, i: int | str) -> float:
        """Returns the i-th coefficient where i can be a variable name or index.

        Args:
            i: the index or name of the variable corresponding to the coefficient.


        Returns:
            The coefficient corresponding to the addressed variable.
        """
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        return float(self.coefficients[i])

    def __setitem__(self, i: int | str, value: float) -> None:
        """Set item for LinearExpression."""
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        self._coefficients[i] = value

    def _coeffs_to_representation(
        self, coefficients: ndarray | sparray | spmatrix | list | dict[int | str, float]
    ) -> ndarray | sparray:
        """Maps given 1d-coefficients to this expression's storage representation.

        Args:
            coefficients: The 1d-coefficients to be mapped.


        Returns:
            The given 1d-coefficients cast to this expression's representation.

        Raises:
            OptimizationError: if coefficients are given in unsupported format.
        """
        n = self.optimization_problem.get_num_vars()
        if isinstance(coefficients, list | ndarray):
            dense = np.ascontiguousarray(coefficients, dtype=np.float64)
            if dense.ndim != 1 or dense.shape[0] != n:
                raise OptimizationError(
                    "The coefficient array or list for the linear expression must be "
                    "one-dimensional and length must match the number of variables."
                )
        elif isinstance(coefficients, sparray | spmatrix):
            if coefficients.ndim != 1 or coefficients.shape[0] != n:
                raise OptimizationError(
                    "The sparse coefficient array for the linear expression must be "
                    "one-dimensional with length matching the number of variables."
                )
            dense = np.ascontiguousarray(coefficients.toarray().reshape(-1), dtype=np.float64)
        elif isinstance(coefficients, dict):
            dense = np.zeros(n, dtype=np.float64)
            for index, value in coefficients.items():
                if isinstance(index, str):
                    index = self.optimization_problem.variables_index[index]
                dense[index] = value
        else:
            raise OptimizationError(
                f"Unsupported format for coefficients: {type(coefficients)}. Supported formats are:"
                "list, numpy array, scipy sparse matrix, dictionary."
            )

        if self._representation == Representation.DENSE:
            return dense
        return _SPARSE_1D_CONSTRUCTORS[self._representation](dense)

    @property
    def coefficients(self) -> ndarray | sparray:
        """Returns the coefficients of the linear expression.

        Returns:
            The coefficients of the linear expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self,
        coefficients: ndarray | sparray | spmatrix | list[float] | dict[int | str, float],
    ) -> None:
        """Sets the coefficients of the linear expression.

        Args:
            coefficients: The coefficients of the linear expression.
        """
        self._coefficients = self._coeffs_to_representation(coefficients)

    def to_array(self) -> ndarray:
        """Returns the coefficients of the linear expression as array.

        Returns:
            An array with the coefficients corresponding to the linear expression.
        """
        if isinstance(self._coefficients, ndarray):
            return cast(ndarray, self._coefficients.copy())
        return cast(ndarray, self._coefficients.toarray())

    def to_dict(self, use_name: bool = False) -> dict[int | str, float]:
        """Returns the coefficients of the linear expression as dictionary.

        Either using variable names or indices as keys.

        Args:
            use_name: Determines whether to use index or names to refer to variables.


        Returns:
            An dictionary with the coefficients corresponding to the linear expression.
        """
        if isinstance(self._coefficients, ndarray):
            dense = self._coefficients
            indices = np.nonzero(dense)[0]
            items = ((int(k), float(dense[k])) for k in indices)
        elif isinstance(self._coefficients, dok_array):
            items = ((k, float(v)) for k, v in self._coefficients.items())
        else:
            coo = self._coefficients.tocoo()
            items = (
                (int(k), float(v)) for k, v in zip(coo.coords[0], coo.data, strict=True) if v != 0
            )

        if use_name:
            return {self.optimization_problem.variables[k].name: v for k, v in items}
        return dict(items)

    def evaluate(self, x: ndarray | list | dict[int | str, float]) -> float:
        """Evaluate the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the linear expression given the variable values.
        """
        x_arr: ndarray
        if isinstance(x, dict):
            n = self.optimization_problem.get_num_vars()
            x_arr = np.zeros(n, dtype=np.float64)
            for index, value in x.items():
                if isinstance(index, str):
                    index = self.optimization_problem.variables_index[index]
                x_arr[index] = value
        else:
            x_arr = np.asarray(x, dtype=np.float64)

        # compute the dot-product of the input and the linear coefficients
        val = self.coefficients @ x_arr

        # return the result
        return float(val)

    # pylint: disable=unused-argument
    def evaluate_gradient(self, x: ndarray | list | dict[int | str, float]) -> ndarray:
        """Evaluate the gradient of the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the gradient of the linear expression given the variable values.
        """
        # extract the coefficients as array and return it
        return self.to_array()

    @property
    def bounds(self) -> ExpressionBounds:
        """Returns the lower bound and the upper bound of the linear expression.

        Returns:
            The lower bound and the upper bound of the linear expression

        Raises:
            OptimizationError: if the linear expression contains any unbounded variable

        """
        l_b = u_b = 0.0
        for ind, coeff in self.to_dict().items():
            x = self.optimization_problem.get_variable(ind)
            if x.lowerbound == -INFINITY or x.upperbound == INFINITY:
                raise OptimizationError(
                    f"Linear expression contains an unbounded variable: {x.name}"
                )
            lst = [coeff * x.lowerbound, coeff * x.upperbound]
            l_b += min(lst)
            u_b += max(lst)
        return ExpressionBounds(lowerbound=l_b, upperbound=u_b)

    def __repr__(self):
        """Repr. for LinearExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        return f"<{self.__class__.__name__}: {expr2str(linear=self, truncate=DEFAULT_TRUNCATE)}>"

    def __str__(self):
        """Str. for LinearExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        return f"{expr2str(linear=self)}"
