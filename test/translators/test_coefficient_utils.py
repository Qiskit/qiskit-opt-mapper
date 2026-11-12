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

"""Tests for the shared translator coefficient helpers."""

import numpy as np
import pytest
from qiskit_addon_opt_mapper.translators._coefficient_utils import is_zero_coefficients
from scipy.sparse import dok_array


def _sparse_with_nonzero() -> dok_array:
    array = dok_array((4,))
    array[1] = 1.0
    return array


@pytest.mark.parametrize(
    "coefficients",
    [
        np.zeros(4),
        dok_array((4,)),
    ],
    ids=["dense", "sparse"],
)
def test_is_zero_coefficients_true_for_all_zero(coefficients):
    assert is_zero_coefficients(coefficients) is True


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([0.0, 1.0, 0.0, 0.0]),
        _sparse_with_nonzero(),
    ],
    ids=["dense", "sparse"],
)
def test_is_zero_coefficients_false_for_nonzero(coefficients):
    assert is_zero_coefficients(coefficients) is False
