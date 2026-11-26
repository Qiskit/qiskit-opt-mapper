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

"""Compiled hypervolume by slicing. Imported only when ``numba`` is installed and needed."""

from __future__ import annotations

import numpy as np
from numba import njit


def _compile(function):
    """Compile with the result cached on disk, or without caching if no folder is writable."""
    try:
        return njit(cache=True)(function)
    except RuntimeError:
        return njit(cache=False)(function)


@_compile
def _slicing_2d(points: np.ndarray, ref0: float, ref1: float) -> float:
    """Area dominated by points in two objectives. Minimization."""
    order = np.argsort(points[:, 0])
    total = 0.0
    best = ref1
    last = order.shape[0] - 1
    for k in range(order.shape[0]):
        best = min(best, points[order[k], 1])
        right = ref0 if k == last else points[order[k + 1], 0]
        total += (right - points[order[k], 0]) * (ref1 - best)
    return total


@_compile
def _slicing(points: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume by slicing the last objective. Minimization."""
    n_objectives = points.shape[1]
    if n_objectives == 1:
        return float(reference[0] - points[:, 0].min())
    if n_objectives == 2:
        return float(_slicing_2d(points, reference[0], reference[1]))

    order = np.argsort(points[:, n_objectives - 1])
    last = order.shape[0] - 1
    total = 0.0
    for k in range(order.shape[0]):
        upper = reference[n_objectives - 1] if k == last else points[order[k + 1], n_objectives - 1]
        height = upper - points[order[k], n_objectives - 1]
        if height > 0:
            sub = np.empty((k + 1, n_objectives - 1))
            for j in range(k + 1):
                sub[j, :] = points[order[j], : n_objectives - 1]
            total += height * _slicing(sub, reference[: n_objectives - 1])
    return total


def hypervolume_numba(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Hypervolume with compiled code. Minimization; every point is inside the reference point."""
    return float(
        _slicing(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(reference_point, dtype=np.float64),
        )
    )
