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

"""The hypervolume indicator of a set of objective values."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

import qiskit_addon_opt_mapper.optionals as _optionals

from ..problems.optimization_objective import ObjSense

_NUMBA_MIN_OBJECTIVES = 4


def hypervolume(
    values: np.ndarray | Sequence[Sequence[float]],
    reference_point: np.ndarray | Sequence[float] | None = None,
    *,
    sense: str | ObjSense | Sequence[str | ObjSense],
    exclude_outside_reference: bool = True,
) -> float:
    """Compute the hypervolume of a set of objective values.

    The hypervolume is the size of the region that is dominated by at least one point and lies
    between the points and the reference point (an area for two objectives, a volume for three,
    and so on). A larger value means a better set of solutions. It lets you compare, for example,
    the result of a quantum run against a classical benchmark using only arrays of numbers.

    Any set of points is accepted. A point that another point beats in every objective (a
    dominated point) does not change the value, so you can pass the Pareto front or a raw
    collection of samples and get the same result.

    For two or three objectives, the computation uses a built-in NumPy implementation that needs
    no extra package. For four or more objectives, it uses a compiled version of the same
    algorithm when ``numba`` is installed (``pip install 'qiskit-addon-opt-mapper[numba]'``).
    Without ``numba``, the NumPy implementation is used for any number of objectives, but it is
    slow for four or more (it can take minutes for five objectives and a few hundred points).
    Both give the same value.

    Args:
        values: Objective values of the points, shape ``(n_points, n_objectives)``. Every entry
            must be finite. A set without points, shape ``(0, n_objectives)``, has hypervolume 0.
        reference_point: The worst values you are willing to accept, shape ``(n_objectives,)``,
            in the same units and orientation as ``values``. Every entry must be finite. A point
            only contributes if it is better than the reference point in at least one objective
            and not worse in any. If ``None`` (default), the worst value of each objective among
            ``values`` is used (the nadir point). Then the reference point depends on the points
            you pass, so hypervolumes of different sets are not comparable, and a point that is
            worst in every objective contributes nothing. To compare sets, pass the same
            ``reference_point`` to every call.
        sense: Whether each objective is minimized or maximized. Either one value that applies to
            every objective, or a sequence with one value per objective. A value is an
            :class:`~qiskit_addon_opt_mapper.problems.ObjSense` or one of the strings ``"min"``,
            ``"minimize"``, ``"max"``, ``"maximize"`` (not case sensitive). This argument is
            required and must be passed by name. There is no default.
        exclude_outside_reference: Controls what happens to points that are worse than the
            reference point in some objective, or equal to it in every objective. Such points
            enclose no volume. If ``True`` (default), they are left out and a warning reports how
            many. If ``False``, a ``ValueError`` is raised instead.

    Returns:
        The hypervolume, in the product of the units of the objectives. It is ``0.0`` if no
        point remains after the exclusion.

    Raises:
        ValueError: If ``values`` does not have shape ``(n_points, n_objectives)`` with at least
            one objective, if ``reference_point`` does not have shape ``(n_objectives,)``, if
            ``sense`` is a sequence of the wrong length, if any entry is NaN or infinite, or if
            ``exclude_outside_reference`` is ``False`` and a point is outside the reference point.
        OptimizationError: If a ``sense`` value is not valid.

    Warns:
        UserWarning: If points are excluded because they are outside the reference point.

    Examples:
        Two points in two objectives, both maximized, measured from the origin. The first point
        covers an area of 3, the second covers an area of 4, and they overlap in an area of 2,
        so the total is 3 + 4 - 2:

        >>> hypervolume([[3.0, 1.0], [2.0, 2.0]], [0.0, 0.0], sense="max")
        5.0

        Without a reference point, the worst value of each objective is used. Here that is the
        point (1, 1), and only the middle point encloses any area:

        >>> hypervolume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], sense="min")
        0.25
    """
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError(
            f"values must have shape (n_points, n_objectives) with n_objectives >= 1, "
            f"got {points.shape}."
        )
    n_objectives = points.shape[1]
    reference = None if reference_point is None else np.asarray(reference_point, dtype=float)
    if reference is not None and reference.shape != (n_objectives,):
        raise ValueError(
            f"reference_point must have shape ({n_objectives},) to match values, "
            f"got {reference.shape}."
        )
    if not (np.isfinite(points).all() and (reference is None or np.isfinite(reference).all())):
        raise ValueError("values and reference_point must not contain NaN or infinity.")

    signs = _minimization_signs(sense, n_objectives)
    points = points * signs
    if points.shape[0] == 0:
        return 0.0
    reference = points.max(axis=0) if reference is None else reference * signs

    inside = np.all(points <= reference, axis=1) & np.any(points < reference, axis=1)
    n_outside = int((~inside).sum())
    if n_outside:
        if not exclude_outside_reference:
            raise ValueError(
                f"{n_outside} point(s) are outside the reference point (worse in some objective, "
                f"or equal in every objective)."
            )
        warnings.warn(
            f"{n_outside} point(s) are outside the reference point and are excluded from the "
            f"hypervolume. They enclose no volume. Move the reference point further away to "
            f"include them.",
            UserWarning,
            stacklevel=2,
        )
        points = points[inside]
    if points.shape[0] == 0:
        return 0.0

    if n_objectives >= _NUMBA_MIN_OBJECTIVES and _optionals.HAS_NUMBA:
        return _hypervolume_numba(points, reference)
    return _hypervolume_slicing(points, reference)


def _minimization_signs(
    sense: str | ObjSense | Sequence[str | ObjSense], n_objectives: int
) -> np.ndarray:
    """Return ``+1`` for each minimized objective and ``-1`` for each maximized one."""
    if isinstance(sense, Sequence) and not isinstance(sense, str):
        senses = [ObjSense.convert(item) for item in sense]
        if len(senses) != n_objectives:
            raise ValueError(
                f"sense must have one entry per objective ({n_objectives}), got {len(senses)}."
            )
    else:
        senses = [ObjSense.convert(sense)] * n_objectives
    return np.array([item.value for item in senses], dtype=float)


def _hypervolume_numba(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Hypervolume with compiled code. Minimization; every point is inside the reference point."""
    from ._hypervolume_numba import hypervolume_numba  # pylint: disable=import-outside-toplevel

    return hypervolume_numba(points, reference_point)


def _hypervolume_slicing(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Hypervolume by slicing the last objective. Minimization; every point is inside the reference point.

    Sort the points by the last objective. Between two consecutive values of that objective,
    the points at or below the lower value cover the same region in the other objectives, so the
    hypervolume is the sum over these slices of the slice height times the hypervolume of the
    remaining objectives. This is the "hypervolume by slicing objectives" method of
    While et al. (2006), IEEE Transactions on Evolutionary Computation 10(1).
    """
    n_objectives = points.shape[1]
    if n_objectives == 1:
        return float(reference_point[0] - points[:, 0].min())
    if n_objectives == 2:
        ordered = points[np.argsort(points[:, 0])]
        widths = np.diff(np.append(ordered[:, 0], reference_point[0]))
        heights = reference_point[1] - np.minimum.accumulate(ordered[:, 1])
        return float(np.sum(widths * heights))

    ordered = points[np.argsort(points[:, -1])]
    slice_heights = np.append(ordered[1:, -1], reference_point[-1]) - ordered[:, -1]
    return float(
        sum(
            slice_heights[i] * _hypervolume_slicing(ordered[: i + 1, :-1], reference_point[:-1])
            for i in np.flatnonzero(slice_heights > 0)
        )
    )
