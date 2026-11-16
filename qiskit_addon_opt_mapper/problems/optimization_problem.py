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


"""Optimization Problem."""

import logging

from qiskit.quantum_info import SparsePauliOp

from .base_optimization_problem import (  # noqa: F401
    BaseOptimizationProblem,
    HigherOrderCoeffLike,
    LinearCoeffLike,
    OptimizationProblemStatus,
    QuadraticCoeffLike,
)
from .constraint import ConstraintSense  # noqa: F401
from .optimization_objective import ObjSense, OptimizationObjective
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
        linear: LinearCoeffLike | None = None,
        quadratic: QuadraticCoeffLike | None = None,
        higher_order: HigherOrderCoeffLike | None = None,
        name: str = "",
    ) -> None:
        """Sets an objective to be minimized.

        Args:
            constant: The constant part of the objective function.
            linear: The coefficients for the linear part of the objective function.
            quadratic: The coefficients for the quadratic part of the objective function.
            higher_order: A dictionary of {order: coeffs} for multiple orders (k>=3).
            name: The name of the objective function. The default empty string leaves the
                objective unnamed.
        """
        if name:
            self._check_name(name, "Objective")
        self._append_or_set_objective_from_parts(
            OptimizationObjective.Sense.MINIMIZE,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            higher_order=higher_order,
            name=name,
        )

    def maximize(
        self,
        constant: float = 0.0,
        linear: LinearCoeffLike | None = None,
        quadratic: QuadraticCoeffLike | None = None,
        higher_order: HigherOrderCoeffLike | None = None,
        name: str = "",
    ) -> None:
        """Sets an objective to be maximized.

        Args:
            constant: The constant part of the objective function.
            linear: The coefficients for the linear part of the objective function.
            quadratic: The coefficients for the quadratic part of the objective function.
            higher_order: A dictionary of {order: coeffs} for multiple orders (k>=3).
            name: The name of the objective function. The default empty string leaves the
                objective unnamed.
        """
        if name:
            self._check_name(name, "Objective")
        self._append_or_set_objective_from_parts(
            OptimizationObjective.Sense.MAXIMIZE,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            higher_order=higher_order,
            name=name,
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
        from ..translators.prettyprint import DEFAULT_TRUNCATE

        return self._objective.render(DEFAULT_TRUNCATE)

    def _objectives_str(self) -> str:
        """Renders the objective fragment of ``__str__``.

        Returns:
            The rendering of the single objective.
        """
        return self._objective.render()

    def _append_or_set_objective_from_parts(
        self,
        sense: ObjSense,
        *,
        constant: float = 0.0,
        linear: LinearCoeffLike | None = None,
        quadratic: QuadraticCoeffLike | None = None,
        higher_order: HigherOrderCoeffLike | None = None,
        name: str = "",
    ) -> None:
        """Sets the single objective from its already-assembled parts.

        This method does not check ``name``. :meth:`minimize` and :meth:`maximize` check it
        before they call this method.

        Args:
            sense: The sense of the objective.
            constant: The constant part of the objective function.
            linear: The coefficients of the linear part.
            quadratic: The coefficients of the quadratic part.
            higher_order: A dictionary of ``{order: coeffs}`` for the higher-order part.
            name: The name of the objective. The default empty string leaves it unnamed.
        """
        self._objective = OptimizationObjective(
            self,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            sense=sense,
            higher_order=higher_order,
            name=name,
        )
