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

"""Converters to flip problem sense, e.g. maximization to minimization and vice versa."""

import copy
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from ..exceptions import OptimizationError
from ..problems.optimization_objective import ObjSense
from .optimization_problem_converter import OptimizationProblemConverter

if TYPE_CHECKING:
    from ..problems.base_optimization_problem import BaseOptimizationProblem

_T = TypeVar("_T", bound="BaseOptimizationProblem")


class _FlipProblemSense(OptimizationProblemConverter):
    """Flip the sense of a problem.

    e.g. converts from maximization to minimization and
    vice versa, regardless of the current sense.
    """

    def __init__(self) -> None:
        self._src_num_vars: int | None = None

    def convert(self, problem: _T) -> _T:
        """Flip the sense of a problem.

        Args:
            problem: The problem to be flipped.


        Returns:
            A converted problem, that has the flipped sense.
        """
        # copy original number of variables as reference.
        self._src_num_vars = problem.get_num_vars()

        needs_flip = any(
            obj.sense != self._get_desired_sense(obj.sense) for obj in problem.iter_objectives()
        )
        if not needs_flip:
            return problem

        desired_problem = copy.deepcopy(problem)
        for obj in desired_problem.iter_objectives():
            desired_sense = self._get_desired_sense(obj.sense)
            if obj.sense != desired_sense:
                obj.sense = desired_sense
                obj.constant = (-1) * obj.constant
                obj.linear = (-1) * obj.linear.coefficients
                obj.quadratic = (-1) * obj.quadratic.coefficients
                obj.higher_order = {
                    degree: (-1) * ho.to_array() for degree, ho in obj.higher_order.items()
                }

        return desired_problem

    def _get_desired_sense(self, current_sense: ObjSense) -> ObjSense:
        """Computes a desired sense of the objective. By default, flip the sense.

        Args:
            current_sense: The current sense of the objective.


        Returns:
            A desired sense, if the objective was minimization, then maximization and vice versa.
        """
        if current_sense == ObjSense.MAXIMIZE:
            return ObjSense.MINIMIZE
        return ObjSense.MAXIMIZE

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

        Note: This implementation does not modify the result, but the method is required because
        the base class defines `interpret` as an abstract method.


        Args:
            x: The result of the converted problem or the given result in case of FAILURE.


        Returns:
            The result of the original problem.

        Raises:
            OptimizationError: if the number of variables in the result differs from
                                     that of the original problem.
        """
        if len(x) != self._src_num_vars:
            raise OptimizationError(
                f"The number of variables in the passed result differs from "
                f"that of the original problem, should be {self._src_num_vars}, but got {len(x)}."
            )
        return np.asarray(x)


class MaximizeToMinimize(_FlipProblemSense):
    """Convert a maximization problem to a minimization problem only if it is a maximization problem.

    Otherwise problem's sense is unchanged.
    """

    def _get_desired_sense(self, current_sense: ObjSense) -> ObjSense:
        return ObjSense.MINIMIZE


class MinimizeToMaximize(_FlipProblemSense):
    """Convert a minimization problem to a maximization problem only if it is a minimization problem.

    Otherwise problem's sense is unchanged.
    """

    def _get_desired_sense(self, current_sense: ObjSense) -> ObjSense:
        return ObjSense.MAXIMIZE
