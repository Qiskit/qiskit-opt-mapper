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

"""Multi-Objective Optimization Problem."""

import logging
from collections.abc import Iterator

from qiskit.quantum_info import SparsePauliOp

from ..exceptions import OptimizationError
from .base_optimization_problem import (
    BaseOptimizationProblem,
    HigherOrderCoeffLike,
    LinearCoeffLike,
    QuadraticCoeffLike,
)
from .optimization_objective import ObjSense, OptimizationObjective

logger = logging.getLogger(__name__)


class MultiObjectiveProblem(BaseOptimizationProblem):
    """Multi-objective optimization problem representation.

    It holds an ordered collection of objective functions, added one at a time with
    :meth:`add_objective`, and otherwise behaves exactly like an
    :class:`~qiskit_addon_opt_mapper.OptimizationProblem`: it supports inequality and equality
    constraints, as well as continuous, binary, integer, and spin variables.

    A problem with zero objectives is legal, mirroring a problem with no constraints.
    """

    _objectives: list[OptimizationObjective]
    _objectives_index: dict[str, int]

    @property
    def objectives(self) -> list[OptimizationObjective]:
        """Returns the list of objectives of the optimization problem.

        Returns:
            List of objectives.
        """
        return self._objectives

    @property
    def objectives_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of an objective to its index.

        Returns:
            The objective index dictionary.
        """
        return self._objectives_index

    def add_objective(  # pylint: disable=too-many-positional-arguments
        self,
        constant: float = 0.0,
        linear: LinearCoeffLike | None = None,
        quadratic: QuadraticCoeffLike | None = None,
        higher_order: HigherOrderCoeffLike | None = None,
        sense: str | ObjSense = ObjSense.MINIMIZE,
        name: str | None = None,
    ) -> OptimizationObjective:
        """Adds an objective function to the optimization problem.

        The objective is of the form:
            ``constant + linear * x + x * quadratic * x + sum_{k>=3} H_k(x)``.

        Args:
            constant: The constant part of the objective function.
            linear: The coefficients for the linear part of the objective function.
            quadratic: The coefficients for the quadratic part of the objective function.
            higher_order: A dictionary of {order: coeffs} for multiple orders (k>=3).
            sense: The sense of the objective function,

              - ``min`` and ``minimize`` denote 'minimize'.
              - ``max`` and ``maximize`` denote 'maximize'.

              An :class:`~qiskit_addon_opt_mapper.problems.ObjSense` member is also accepted.
            name: The name of the objective function.
                If it's ``None`` or empty ``""``, the default name, e.g., ``f0``, is used.


        Returns:
            The added objective. It is the live object held by the problem, so mutating it is
            visible through :meth:`get_objective`.

        Raises:
            OptimizationError: if the objective name already exists or the sense is not valid.
        """
        if name:
            if name in self._objectives_index:
                raise OptimizationError(f"Objective name already exists: {name}")
            self._check_name(name, "Objective")
        else:
            k = self.get_num_objectives()
            while f"f{k}" in self._objectives_index:
                k += 1
            name = f"f{k}"
        objective = OptimizationObjective(
            self,
            constant,
            {} if linear is None else linear,
            {} if quadratic is None else quadratic,
            {} if higher_order is None else higher_order,
            ObjSense.convert(sense),
            name,
        )
        self._objectives_index[name] = len(self._objectives)
        self._objectives.append(objective)
        return objective

    def get_objective(self, i: int | str) -> OptimizationObjective:
        """Returns an objective for a given name or index.

        Args:
            i: The index or name of the objective. Negative indices follow standard Python
                sequence indexing (e.g., ``-1`` returns the last objective). Any integer type
                that supports ``__index__``, such as a NumPy integer, is accepted as an index.

        Returns:
            The corresponding objective.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If the name does not exist.
        """
        if isinstance(i, str):
            i = self._objectives_index[i]
        return self._objectives[i]

    def get_num_objectives(self) -> int:
        """Returns the number of objectives.

        Returns:
            The number of objectives.
        """
        return len(self._objectives)

    def remove_objective(self, i: str | int) -> None:
        """Remove an objective.

        Args:
            i: An index or name of an objective. Negative indices follow standard Python
                sequence indexing (e.g., ``-1`` removes the last objective).

        Raises:
            KeyError: If name does not exist.
            IndexError: If index is out of range.
        """
        if isinstance(i, str):
            i = self._objectives_index[i]
        del self._objectives[i]
        self._objectives_index = {obj.name: j for j, obj in enumerate(self._objectives)}

    def to_ising(self) -> list[tuple[SparsePauliOp, float]]:
        """Return one Ising Hamiltonian per objective of this problem.

        Variables are mapped to qubits in qiskit order, i.e.,
        i-th variable is mapped to index n-i in the pauli string, where
        n is the total number of variables.


        Returns:
            A list of tuples (qubit_op, offset), one per objective of the problem in objective
            order, where ``qubit_op`` is the qubit operator for the objective and ``offset`` is
            the constant value in its Ising Hamiltonian. It is empty if the problem has no
            objective.

        Raises:
            OptimizationError: If variables are neither only binary nor only spin.
            OptimizationError: If constraints exist in the problem.
        """
        # pylint: disable=cyclic-import
        from ..translators.ising import to_isings

        return to_isings(self)

    def _reset_objectives(self) -> None:
        """Resets the collection of objectives of this problem to an empty one."""
        self._objectives = []
        self._objectives_index = {}

    @property
    def is_multi_objective(self) -> bool:
        """Whether this problem holds a collection of objectives or a single one.

        Returns:
            ``True``, since a ``MultiObjectiveProblem`` always holds a collection of objectives.
        """
        return True

    def _iter_objectives(self) -> Iterator[OptimizationObjective]:
        """Iterates over the objectives of this problem in objective order.

        Returns:
            An iterator over the objectives of this problem, in index order.
        """
        return iter(self._objectives)

    def _objectives_repr(self) -> str:
        """Renders the objective fragment of ``__repr__``.

        Returns:
            The number of objectives, so that the representation stays short.
        """
        n = self.get_num_objectives()
        return f"{n} objective" if n == 1 else f"{n} objectives"

    def _objectives_str(self) -> str:
        """Renders the objective fragment of ``__str__``.

        Returns:
            The renderings of all objectives, separated by ``"; "``.
        """
        if not self._objectives:
            return "no objectives"
        return "; ".join(obj.render() for obj in self._objectives)

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
        """Appends an objective built from its already-assembled parts.

        Unlike ``OptimizationProblem``, this method does check ``name``, because it goes through
        :meth:`add_objective`. A non-printable name therefore warns again when an already-checked
        objective is copied, for example by :meth:`substitute_variables`.

        Args:
            sense: The sense of the objective.
            constant: The constant part of the objective function.
            linear: The coefficients of the linear part.
            quadratic: The coefficients of the quadratic part.
            higher_order: The coefficients of the higher-order part.
            name: The name of the objective. The default empty string selects the next default
                name, e.g., ``f0``.
        """
        self.add_objective(
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            higher_order=higher_order,
            sense=sense,
            name=name,
        )
