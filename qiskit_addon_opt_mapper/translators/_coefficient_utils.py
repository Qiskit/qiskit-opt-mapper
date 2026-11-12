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

"""Shared helpers for translators that inspect expression coefficients."""

import numpy as np
from numpy import ndarray
from scipy.sparse import sparray, spmatrix


def is_zero_coefficients(coefficients: ndarray | sparray | spmatrix) -> bool:
    """Checks whether all coefficients are zero, regardless of storage representation.

    Sparse ``sparray``/``spmatrix`` instances expose a cheap ``nnz`` count; plain ``ndarray``
    instances (the case once dense storage is supported) do not, so this dispatches on type
    instead of assuming ``.nnz`` is always present.

    Args:
        coefficients: The coefficients to check.

    Returns:
        True if every coefficient is zero, False otherwise.
    """
    if isinstance(coefficients, ndarray):
        return not bool(np.any(coefficients))
    return bool(coefficients.nnz == 0)
