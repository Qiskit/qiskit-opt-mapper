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

"""An abstract class for optimization algorithms in Qiskit optimization module."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ..problems.base_optimization_problem import BaseOptimizationProblem

_T = TypeVar("_T", bound="BaseOptimizationProblem")


class OptimizationProblemConverter(ABC):
    """An abstract class for converters of optimization problem in Qiskit optimization module."""

    @abstractmethod
    def convert(self, problem: _T) -> _T:
        """Convert method.

        Convert an optimization problem into another form
        and keep the information required to interpret the result.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Interpret a result into another form using the information of conversion."""
        raise NotImplementedError
