# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Optimization Problem."""

import logging

from numpy import ndarray
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import spmatrix

from .base_optimization_problem import (  # noqa: F401
    BaseOptimizationProblem,
    CoeffLike,
    OptimizationProblemStatus,
)
from .constraint import ConstraintSense  # noqa: F401
from .optimization_objective import OptimizationObjective
from .variable import Variable, VarType  # noqa: F401

logger = logging.getLogger(__name__)


class OptimizationProblem(BaseOptimizationProblem):
    """Single-objective optimization problem representation.

    This representation supports a single objective (minimization or maximization),
    inequality and equality constraints, as well as continuous, binary, integer,
    and spin variables.
    """

    _objective: OptimizationObjective

    @property
    def objective(self) -> OptimizationObjective:
        """Returns the quadratic objective.

        Returns:
            The quadratic objective.
        """
        return self._objective

    def minimize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        higher_order: CoeffLike | dict[int, CoeffLike] | None = None,
    ) -> None:
        """Sets an objective to be minimized."""
        self._objective = OptimizationObjective(
            self,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            sense=OptimizationObjective.Sense.MINIMIZE,
            higher_order=higher_order,  # type: ignore
        )

    def maximize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        higher_order: CoeffLike | dict[int, CoeffLike] | None = None,
    ) -> None:
        """Sets an objective to be maximized."""
        self._objective = OptimizationObjective(
            self,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            sense=OptimizationObjective.Sense.MAXIMIZE,
            higher_order=higher_order,  # type: ignore
        )

    def to_ising(self) -> tuple[SparsePauliOp, float]:
        """Return the Ising Hamiltonian of this problem.

        Variables are mapped to qubits in qiskit order, i.e.,
        i-th variable is mapped to index n-i in the pauli string, where
        n is the total number of variables.


        Returns:
            qubit_op: The qubit operator for the problem
            offset: The constant value in the Ising Hamiltonian.

        Raises:
            OptimizationError: If a variable type is not binary.
            OptimizationError: If constraints exist in the problem.
        """
        # pylint: disable=cyclic-import
        from ..translators.ising import to_ising

        return to_ising(self)

    def substitute_variables(
        self,
        constants: dict[str | int, float] | None = None,
        variables: dict[str | int, tuple[str | int, float]] | None = None,
    ) -> "OptimizationProblem":
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                e.g., ``{'x': 2}`` means ``x`` is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., ``{'x': ('y', 2)}`` means ``x`` is substituted with ``y * 2``


        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, ``OptimizationProblem.status`` is still
            ``OptimizationProblem.Status.VALID``.
            Otherwise, it gets ``OptimizationProblem.Status.INFEASIBLE``.

        Raises:
            OptimizationError: if the substitution is invalid as follows.

                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """
        # pylint: disable=cyclic-import
        from .substitute_variables import substitute_variables

        return substitute_variables(self, constants, variables)

    def prettyprint(self, wrap: int = 80) -> str:
        """Returns a pretty printed string of this problem.

        Args:
            wrap: The text width to wrap the output strings. It is disabled by setting 0.
                Note that some strings might exceed this value, for example, a long variable
                name won't be wrapped. The default value is 80.


        Returns:
            A pretty printed string representing the problem.

        Raises:
            OptimizationError: if there is a non-printable name.
        """
        # pylint: disable=cyclic-import
        from qiskit_addon_opt_mapper.translators.prettyprint import prettyprint

        return prettyprint(self, wrap)

    def _reset_objectives(self) -> None:
        """Resets the single objective of this problem to an empty one."""
        self._objective = OptimizationObjective(self)

    def _objectives_repr(self) -> str:
        """Renders the objective fragment of ``__repr__``.

        Returns:
            The truncated rendering of the single objective.
        """
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        objective = self._objective
        expr_str = expr2str(
            constant=objective.constant,
            linear=objective.linear,
            quadratic=objective.quadratic,
            higher_order=objective.higher_order,
            truncate=DEFAULT_TRUNCATE,
        )
        return f"{objective.sense.name.lower()} {expr_str}"

    def _objectives_str(self) -> str:
        """Renders the objective fragment of ``__str__``.

        Returns:
            The rendering of the single objective.
        """
        return str(self._objective)
