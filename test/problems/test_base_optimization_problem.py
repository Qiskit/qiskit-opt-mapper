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
"""Tests specific to BaseOptimizationProblem (abstractness and hierarchy)."""

import pytest
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.problems import OptimizationProblemElement
from qiskit_addon_opt_mapper.problems.base_optimization_problem import BaseOptimizationProblem


class _MissingHooks(BaseOptimizationProblem):
    """Subclass that implements none of the abstract objective hooks."""


class _MissingStrHook(BaseOptimizationProblem):
    """Subclass that implements all abstract objective hooks except ``_objectives_str``."""

    def _reset_objectives(self) -> None:
        pass

    def _objectives_repr(self) -> str:
        return ""


def test_base_is_abstract():
    """BaseOptimizationProblem cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseOptimizationProblem("p")


@pytest.mark.parametrize("subclass", [_MissingHooks, _MissingStrHook])
def test_subclass_missing_hooks_is_abstract(subclass):
    """A subclass that does not implement every objective hook cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        subclass("p")


def test_is_subclass_of_base():
    """OptimizationProblem is a subclass of BaseOptimizationProblem."""
    assert issubclass(OptimizationProblem, BaseOptimizationProblem)


def test_repr_of_empty_problem():
    """The empty-problem repr is byte-identical to the one before the base class existed."""
    assert (
        repr(OptimizationProblem("p"))
        == "<OptimizationProblem: minimize 0, 0 variables, 0 constraints, 'p'>"
    )


def test_element_accepts_optimization_problem_parent():
    """An element can be created with, and re-parented to, an OptimizationProblem."""
    element = OptimizationProblemElement(OptimizationProblem("p"))
    new_parent = OptimizationProblem("q")
    element.optimization_problem = new_parent
    assert element.optimization_problem is new_parent


def test_element_rejects_non_problem_parent():
    """Elements reject parents that are not a BaseOptimizationProblem, in init and setter."""
    with pytest.raises(TypeError, match="BaseOptimizationProblem instance expected"):
        OptimizationProblemElement("not a problem")
    element = OptimizationProblemElement(OptimizationProblem("p"))
    with pytest.raises(TypeError, match="BaseOptimizationProblem instance expected"):
        element.optimization_problem = "not a problem"
