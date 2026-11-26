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

"""Tests of the hypervolume metric."""

import importlib
import itertools
import subprocess
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import qiskit_addon_opt_mapper.optionals as _optionals
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.metrics import hypervolume
from qiskit_addon_opt_mapper.problems import MultiObjectiveProblem, ObjSense

from ..optimization_test_case import OptimizationTestCase

_MODULE = sys.modules[hypervolume.__module__]


def _brute_force(points, reference_point):
    """Inclusion-exclusion hypervolume for minimization. Exponential, for small sets only."""
    total = 0.0
    for size in range(1, len(points) + 1):
        for subset in itertools.combinations(range(len(points)), size):
            corner = points[list(subset)].max(axis=0)
            total += (-1) ** (size + 1) * np.prod(reference_point - corner)
    return total


def _random_front(rng, n_points, n_objectives):
    """Mutually non-dominated points on a sphere, all inside the unit reference point."""
    raw = np.abs(rng.standard_normal((n_points, n_objectives)))
    return 0.99 * raw / np.linalg.norm(raw, axis=1, keepdims=True)


class TestHypervolumeValues(OptimizationTestCase):
    """Hand-computed values. The built-in implementation is forced, so no backend is needed."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(_optionals, "HAS_NUMBA", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_point(self):
        """A single point covers the box up to the reference point."""
        self.assertAlmostEqual(hypervolume([[2.0, 2.0]], [0.0, 0.0], sense="max"), 4.0)
        self.assertAlmostEqual(hypervolume([[-2.0, -2.0]], [0.0, 0.0], sense="min"), 4.0)

    def test_two_objectives(self):
        """Two non-dominated points: boxes 3 and 4 overlap in 2."""
        self.assertAlmostEqual(hypervolume([[3.0, 1.0], [2.0, 2.0]], [0.0, 0.0], sense="max"), 5.0)

    def test_staircase_two_objectives(self):
        """Three points on a staircase: boxes 3, 4 and 3, pair overlaps 2, 2 and 1, triple 1."""
        values = [[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]]
        self.assertAlmostEqual(hypervolume(values, [0.0, 0.0], sense="max"), 6.0)

    def test_three_objectives(self):
        """Boxes 0.5 and 0.25 overlap in 0.125."""
        values = [[0.0, 0.0, 0.5], [0.5, 0.5, 0.0]]
        self.assertAlmostEqual(hypervolume(values, [1.0, 1.0, 1.0], sense="min"), 0.625)

    def test_four_objectives(self):
        """Boxes 0.5 and 0.125 overlap in 0.0625."""
        values = [[0.0, 0.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.0]]
        self.assertAlmostEqual(hypervolume(values, np.ones(4), sense="min"), 0.5625)

    def test_one_objective(self):
        """With one objective the hypervolume is the distance of the best point to the reference."""
        self.assertAlmostEqual(hypervolume([[0.25]], [1.0], sense="min"), 0.75)
        self.assertAlmostEqual(hypervolume([[0.25], [0.5]], [1.0], sense="min"), 0.75)

    def test_mirrored_data_gives_same_value(self):
        """Maximizing mirrored values equals minimizing the originals."""
        rng = np.random.default_rng(3)
        for n_objectives in (2, 3, 4):
            with self.subTest(n_objectives=n_objectives):
                values = _random_front(rng, 6, n_objectives)
                reference = np.ones(n_objectives)
                minimized = hypervolume(values, reference, sense="min")
                maximized = hypervolume(-values, -reference, sense="max")
                self.assertAlmostEqual(minimized, maximized)

    def test_mixed_senses(self):
        """One value per objective: minimize the first, maximize the second."""
        minimized = hypervolume([[0.2, -0.2], [0.8, -0.8]], [1.0, 0.0], sense="min")
        mixed = hypervolume([[0.2, 0.2], [0.8, 0.8]], [1.0, 0.0], sense=["min", "max"])
        self.assertAlmostEqual(mixed, minimized)
        self.assertAlmostEqual(mixed, 0.28)

    def test_sense_spellings(self):
        """ObjSense members and strings of any case are accepted, alone or in a list."""
        expected = hypervolume([[2.0, 2.0]], [0.0, 0.0], sense="max")
        for sense in (
            ObjSense.MAXIMIZE,
            "MAX",
            "Maximize",
            [ObjSense.MAXIMIZE, "max"],
            ("max", "max"),
        ):
            with self.subTest(sense=sense):
                self.assertAlmostEqual(hypervolume([[2.0, 2.0]], [0.0, 0.0], sense=sense), expected)

    def test_array_like_inputs(self):
        """Lists and tuples are accepted as well as arrays."""
        self.assertAlmostEqual(hypervolume(((0.5, 0.5),), (1.0, 1.0), sense="min"), 0.25)

    def test_dominated_points_do_not_change_value(self):
        """Dominated and duplicate points inside the reference point are harmless."""
        base = [[0.2, 0.8], [0.8, 0.2]]
        with_extra = [[0.2, 0.8], [0.8, 0.2], [0.9, 0.9], [0.2, 0.8], [0.2, 0.9]]
        self.assertAlmostEqual(
            hypervolume(with_extra, [1.0, 1.0], sense="min"),
            hypervolume(base, [1.0, 1.0], sense="min"),
        )
        self.assertAlmostEqual(hypervolume(base, [1.0, 1.0], sense="min"), 0.28)

    def test_dominated_points_three_objectives(self):
        """Dominated points in three objectives do not change the value."""
        front = [[0.0, 0.0, 0.5], [0.5, 0.5, 0.0]]
        with_extra = [*front, [0.6, 0.6, 0.6], [0.5, 0.5, 0.5]]
        self.assertAlmostEqual(hypervolume(with_extra, np.ones(3), sense="min"), 0.625)

    def test_matches_brute_force(self):
        """Random fronts with 2 to 5 objectives agree with inclusion-exclusion."""
        rng = np.random.default_rng(11)
        for n_objectives in (2, 3, 4, 5):
            with self.subTest(n_objectives=n_objectives):
                values = _random_front(rng, 7, n_objectives)
                reference = np.ones(n_objectives)
                self.assertAlmostEqual(
                    hypervolume(values, reference, sense="min"), _brute_force(values, reference)
                )

    def test_negative_and_shifted_values(self):
        """Values and reference point may be negative."""
        rng = np.random.default_rng(5)
        for n_objectives in (2, 3, 4):
            with self.subTest(n_objectives=n_objectives):
                values = _random_front(rng, 6, n_objectives) * 3.0 - 5.0
                reference = np.full(n_objectives, -1.5)
                self.assertAlmostEqual(
                    hypervolume(values, reference, sense="min"), _brute_force(values, reference)
                )

    def test_input_is_not_modified(self):
        """The arrays passed in are left unchanged by the sense conversion."""
        values = np.array([[3.0, 1.0], [2.0, 2.0]])
        reference = np.array([0.0, 0.0])
        hypervolume(values, reference, sense="max")
        np.testing.assert_array_equal(values, [[3.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_equal(reference, [0.0, 0.0])

    def test_returns_float(self):
        """The result is a Python float."""
        self.assertIsInstance(hypervolume([[0.5, 0.5]], [1.0, 1.0], sense="min"), float)
        self.assertIsInstance(hypervolume(np.empty((0, 2)), [1.0, 1.0], sense="min"), float)


class TestHypervolumeOutsideReference(OptimizationTestCase):
    """Points that are not inside the reference point."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(_optionals, "HAS_NUMBA", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_outside_point_is_excluded_with_warning(self):
        """A point worse than the reference in one objective is dropped, and a warning counts it."""
        values = [[0.2, 0.8], [1.2, 0.2]]
        with self.assertWarnsRegex(UserWarning, "1 point.*outside the reference point"):
            result = hypervolume(values, [1.0, 1.0], sense="min")
        self.assertAlmostEqual(result, 0.16)

    def test_all_outside_returns_zero(self):
        """If every point is outside, the result is 0.0 with a warning."""
        with self.assertWarnsRegex(UserWarning, "2 point.*outside the reference point"):
            result = hypervolume([[1.2, 0.5], [1.3, 0.4]], [1.0, 1.0], sense="min")
        self.assertEqual(result, 0.0)

    def test_point_equal_to_reference_is_excluded(self):
        """A point equal to the reference in every objective encloses nothing."""
        with self.assertWarnsRegex(UserWarning, "1 point"):
            result = hypervolume([[1.0, 1.0], [0.5, 0.5]], [1.0, 1.0], sense="min")
        self.assertAlmostEqual(result, 0.25)

    def test_outside_with_maximization(self):
        """The test is made after converting to minimization."""
        with self.assertWarnsRegex(UserWarning, "1 point"):
            result = hypervolume([[2.0, 2.0], [-1.0, 3.0]], [0.0, 0.0], sense="max")
        self.assertAlmostEqual(result, 4.0)

    def test_tie_in_one_objective_is_kept_silently(self):
        """A point that ties the reference in one objective but is better in another is kept."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = hypervolume([[0.5, 1.0], [0.2, 0.8]], [1.0, 1.0], sense="min")
        self.assertAlmostEqual(result, 0.16)

    def test_raise_instead_of_excluding(self):
        """With exclude_outside_reference=False, outside points raise ValueError."""
        with self.assertRaisesRegex(ValueError, "1 point.*outside the reference point"):
            hypervolume(
                [[0.2, 0.8], [1.2, 0.2]], [1.0, 1.0], sense="min", exclude_outside_reference=False
            )

    def test_no_error_when_nothing_is_outside(self):
        """exclude_outside_reference=False is silent when all points are inside."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = hypervolume(
                [[0.5, 0.5]], [1.0, 1.0], sense="min", exclude_outside_reference=False
            )
        self.assertAlmostEqual(result, 0.25)

    def test_empty_input_returns_zero_without_warning(self):
        """No points means zero hypervolume."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(hypervolume(np.empty((0, 3)), [1.0, 1.0, 1.0], sense="min"), 0.0)

    def test_warning_points_at_caller(self):
        """The warning is attributed to the calling file."""
        with self.assertWarns(UserWarning) as context:
            hypervolume([[1.2, 0.5]], [1.0, 1.0], sense="min")
        self.assertEqual(context.filename, __file__)


class TestHypervolumeDefaultReferencePoint(OptimizationTestCase):
    """Without a reference point, the worst value of each objective is used (the nadir point)."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(_optionals, "HAS_NUMBA", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_nadir_when_minimizing(self):
        """The default equals passing the largest value of each objective explicitly."""
        values = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        self.assertAlmostEqual(hypervolume(values, sense="min"), 0.25)
        self.assertAlmostEqual(
            hypervolume(values, sense="min"), hypervolume(values, [1.0, 1.0], sense="min")
        )

    def test_default_is_nadir_when_maximizing(self):
        """The default equals passing the smallest value of each objective explicitly."""
        values = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        self.assertAlmostEqual(hypervolume(values, sense="max"), 0.25)
        self.assertAlmostEqual(
            hypervolume(values, sense="max"), hypervolume(values, [0.0, 0.0], sense="max")
        )

    def test_default_with_mixed_senses(self):
        """The nadir is taken per objective, using that objective's sense.

        With the first objective minimized and the second maximized, the nadir is (2, 1). In
        minimization form the points are (0, -4), (1, -1) and (2, -3) against the reference
        (2, -1), which gives 1 * 3 + 1 * 3 + 0 = 6.
        """
        values = [[0.0, 4.0], [1.0, 1.0], [2.0, 3.0]]
        senses = ["min", "max"]
        self.assertAlmostEqual(hypervolume(values, sense=senses), 6.0)
        self.assertAlmostEqual(hypervolume(values, [2.0, 1.0], sense=senses), 6.0)

    def test_default_with_none_is_the_same_as_omitting(self):
        """Passing ``None`` explicitly is the same as leaving it out."""
        values = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        self.assertEqual(hypervolume(values, None, sense="min"), hypervolume(values, sense="min"))

    def test_default_does_not_warn_for_a_front(self):
        """Points that sit on the nadir in one objective are kept and no warning is raised."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hypervolume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], sense="min")

    def test_default_with_two_extreme_points_is_zero(self):
        """Two points leave no dominated area against their own nadir."""
        self.assertEqual(hypervolume([[0.0, 1.0], [1.0, 0.0]], sense="min"), 0.0)

    def test_default_with_single_point_is_zero_and_warns(self):
        """A single point is tied with its own nadir in every objective, so it is excluded."""
        with self.assertWarnsRegex(UserWarning, "1 point"):
            self.assertEqual(hypervolume([[0.5, 0.5]], sense="min"), 0.0)

    def test_default_depends_on_the_points(self):
        """Adding a dominated point that moves the nadir changes the value of the same front.

        The extra point moves the nadir from (1, 1) to (2, 1). Strips of the front against it:
        width 0.5 and height 0.5 from x = 0.5, then width 1 and height 1 from x = 1, so 1.25.
        """
        front = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        with_dominated_point = [*front, [2.0, 0.9]]
        self.assertAlmostEqual(hypervolume(front, sense="min"), 0.25)
        self.assertAlmostEqual(hypervolume(with_dominated_point, sense="min"), 1.25)

    def test_default_with_empty_input_is_zero(self):
        """No points give 0.0, without a warning and without a reference point."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(hypervolume(np.empty((0, 2)), sense="min"), 0.0)

    def test_default_rejects_nan(self):
        """NaN is rejected when the reference point is derived."""
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            hypervolume([[np.nan, 0.5]], sense="min")

    def test_default_runs_through_numba_for_four_objectives(self):
        """The derived reference point reaches the compiled backend in minimization sense."""
        with (
            mock.patch.object(_optionals, "HAS_NUMBA", True),
            mock.patch.object(_MODULE, "_hypervolume_numba", return_value=7.0) as backend,
        ):
            result = hypervolume([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]], sense="min")
        self.assertEqual(result, 7.0)
        _, reference = backend.call_args.args
        np.testing.assert_array_equal(reference, [2.0, 2.0, 4.0, 4.0])


class TestHypervolumeValidation(OptimizationTestCase):
    """Input validation. None of these needs a backend."""

    def test_values_must_be_two_dimensional(self):
        """A flat array is rejected."""
        with self.assertRaisesRegex(ValueError, "shape"):
            hypervolume([0.5, 0.5], [1.0, 1.0], sense="min")
        with self.assertRaisesRegex(ValueError, "shape"):
            hypervolume(np.ones((2, 2, 2)), [1.0, 1.0], sense="min")

    def test_values_need_an_objective(self):
        """Shape (n, 0) is rejected."""
        with self.assertRaisesRegex(ValueError, "n_objectives >= 1"):
            hypervolume(np.empty((3, 0)), [], sense="min")

    def test_reference_point_dimension_mismatch(self):
        """The reference point must have one entry per objective."""
        with self.assertRaisesRegex(ValueError, r"reference_point must have shape \(2,\)"):
            hypervolume([[0.5, 0.5]], [1.0, 1.0, 1.0], sense="min")
        with self.assertRaisesRegex(ValueError, "reference_point"):
            hypervolume([[0.5, 0.5]], [[1.0, 1.0]], sense="min")

    def test_nan_and_infinity(self):
        """NaN and infinity are rejected in values and in the reference point."""
        for values, reference in (
            ([[np.nan, 0.5]], [1.0, 1.0]),
            ([[np.inf, 0.5]], [1.0, 1.0]),
            ([[0.5, 0.5]], [np.nan, 1.0]),
            ([[0.5, 0.5]], [1.0, np.inf]),
            ([[0.5, 0.5]], [1.0, -np.inf]),
        ):
            with (
                self.subTest(values=values, reference=reference),
                self.assertRaisesRegex(ValueError, "NaN or infinity"),
            ):
                hypervolume(values, reference, sense="min")

    def test_non_numeric_values(self):
        """Values that are not numbers are rejected by NumPy."""
        with self.assertRaises(ValueError):
            hypervolume([["a", "b"]], [1.0, 1.0], sense="min")

    def test_sense_sequence_length(self):
        """A sequence of senses needs one entry per objective."""
        with self.assertRaisesRegex(ValueError, "one entry per objective"):
            hypervolume([[0.5, 0.5]], [1.0, 1.0], sense=["min"])
        with self.assertRaisesRegex(ValueError, "one entry per objective"):
            hypervolume([[0.5, 0.5]], [1.0, 1.0], sense=["min", "min", "min"])

    def test_invalid_sense(self):
        """Unknown strings and non-sense objects raise OptimizationError."""
        for sense in ("smallest", "", True, 1, None, ["min", "up"]):
            with self.subTest(sense=sense), self.assertRaises(OptimizationError):
                hypervolume([[0.5, 0.5]], [1.0, 1.0], sense=sense)

    def test_sense_is_required(self):
        """There is no default sense, with or without a reference point."""
        with self.assertRaises(TypeError):
            hypervolume([[0.5, 0.5]], [1.0, 1.0])  # pylint: disable=missing-kwoa
        with self.assertRaises(TypeError):
            hypervolume([[0.5, 0.5]])  # pylint: disable=missing-kwoa

    def test_sense_is_keyword_only(self):
        """The sense cannot be passed by position."""
        with self.assertRaises(TypeError):
            hypervolume([[0.5, 0.5]], [1.0, 1.0], "min")  # pylint: disable=too-many-function-args

    def test_exclude_outside_reference_is_keyword_only(self):
        """The filter option cannot be passed by position."""
        with self.assertRaises(TypeError):
            hypervolume([[0.5, 0.5]], [1.0, 1.0], "min", False)  # pylint: disable=too-many-function-args

    def test_validation_runs_before_backend_use(self):
        """Bad input is rejected even when the backend is selected."""
        with (
            mock.patch.object(_optionals, "HAS_NUMBA", True),
            self.assertRaisesRegex(ValueError, "NaN or infinity"),
        ):
            hypervolume([[np.nan, 0.5, 0.5]], np.ones(3), sense="min")


def _enumerate_objective_values(problem):
    """Objective values of every assignment of a problem with binary variables only."""
    n_variables = problem.get_num_vars()
    assignments = itertools.product([0.0, 1.0], repeat=n_variables)
    return np.array(
        [[objective.evaluate(list(x)) for objective in problem.objectives] for x in assignments]
    )


def _non_dominated(values, senses):
    """Rows of ``values`` that no other row beats in every objective, for the given senses."""
    signs = np.array([sense.value for sense in senses], dtype=float)
    minimized = values * signs
    keep = [
        not any(
            np.all(other <= row) and np.any(other < row)
            for j, other in enumerate(minimized)
            if j != i
        )
        for i, row in enumerate(minimized)
    ]
    return values[keep]


class TestHypervolumeWithMultiObjectiveProblem(OptimizationTestCase):
    """Objective values and senses taken from a ``MultiObjectiveProblem``."""

    def setUp(self):
        super().setUp()
        self.problem = MultiObjectiveProblem("two_objectives")
        self.problem.binary_var("x")
        self.problem.binary_var("y")
        self.problem.add_objective(linear={"x": 1, "y": 2}, sense="min", name="cost")
        self.problem.add_objective(linear={"x": 2, "y": 1}, sense="max", name="gain")
        self.senses = [objective.sense for objective in self.problem.objectives]

    def test_hand_computed_value_with_mixed_senses(self):
        """Cost is minimized and gain is maximized; the staircase below has area 7.

        The assignments give (cost, gain) = (0, 0), (2, 1), (1, 2) and (3, 3). The point (2, 1)
        is dominated by (1, 2). Measured from cost 4 and gain 0, the front covers a strip of width
        1 and height 0, one of width 2 and height 2, and one of width 1 and height 3: 0 + 4 + 3.
        """
        values = _enumerate_objective_values(self.problem)
        front = _non_dominated(values, self.senses)
        self.assertEqual(len(front), 3)
        self.assertAlmostEqual(hypervolume(front, [4.0, 0.0], sense=self.senses), 7.0)

    def test_senses_read_from_problem_match_negated_problem(self):
        """Turning the maximized objective into a minimized one with negated sign changes nothing."""
        flipped = MultiObjectiveProblem("flipped")
        flipped.binary_var("x")
        flipped.binary_var("y")
        flipped.add_objective(linear={"x": 1, "y": 2}, sense="min", name="cost")
        flipped.add_objective(linear={"x": -2, "y": -1}, sense="min", name="negative_gain")
        original_senses = [objective.sense for objective in self.problem.objectives]
        flipped_senses = [objective.sense for objective in flipped.objectives]

        original = hypervolume(
            _non_dominated(_enumerate_objective_values(self.problem), original_senses),
            [4.0, 0.0],
            sense=original_senses,
        )
        negated = hypervolume(
            _non_dominated(_enumerate_objective_values(flipped), flipped_senses),
            [4.0, 0.0],
            sense=flipped_senses,
        )
        self.assertAlmostEqual(original, negated)

    def test_dominated_assignments_do_not_change_value(self):
        """Passing every assignment, dominated ones included, gives the value of the front."""
        values = _enumerate_objective_values(self.problem)
        front = _non_dominated(values, self.senses)
        self.assertAlmostEqual(
            hypervolume(values, [4.0, 0.0], sense=self.senses),
            hypervolume(front, [4.0, 0.0], sense=self.senses),
        )


class TestHypervolumeBackendSelection(OptimizationTestCase):
    """Which implementation is used. Runs without numba by replacing the backend function."""

    def test_numba_used_for_four_or_more_objectives(self):
        """With numba available, 4 or more objectives are sent to it, in minimization sense."""
        with (
            mock.patch.object(_optionals, "HAS_NUMBA", True),
            mock.patch.object(_MODULE, "_hypervolume_numba", return_value=42.0) as backend,
        ):
            result = hypervolume([[2.0, 2.0, 2.0, 2.0]], [0.0, 0.0, 0.0, 0.0], sense="max")
        self.assertEqual(result, 42.0)
        points, reference = backend.call_args.args
        np.testing.assert_array_equal(points, [[-2.0, -2.0, -2.0, -2.0]])
        np.testing.assert_array_equal(reference, [-0.0, -0.0, -0.0, -0.0])

    def test_numpy_used_for_two_and_three_objectives_even_with_numba(self):
        """Up to three objectives never go to numba, because the NumPy code is as fast."""
        with (
            mock.patch.object(_optionals, "HAS_NUMBA", True),
            mock.patch.object(_MODULE, "_hypervolume_numba") as backend,
        ):
            two = hypervolume([[2.0, 2.0]], [0.0, 0.0], sense="max")
            three = hypervolume([[2.0, 2.0, 2.0]], [0.0, 0.0, 0.0], sense="max")
        backend.assert_not_called()
        self.assertAlmostEqual(two, 4.0)
        self.assertAlmostEqual(three, 8.0)

    def test_numpy_used_without_numba(self):
        """Without numba, four objectives use the NumPy code."""
        with (
            mock.patch.object(_optionals, "HAS_NUMBA", False),
            mock.patch.object(_MODULE, "_hypervolume_numba") as backend,
        ):
            result = hypervolume(
                [[0.0, 0.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.0]], np.ones(4), sense="min"
            )
        backend.assert_not_called()
        self.assertAlmostEqual(result, 0.5625)

    def test_import_does_not_load_numba(self):
        """Importing the package and calling hypervolume on two objectives never imports numba."""
        code = (
            "import sys\n"
            "import qiskit_addon_opt_mapper\n"
            "from qiskit_addon_opt_mapper.metrics import hypervolume\n"
            "hypervolume([[0.5, 0.5]], [1.0, 1.0], sense='min')\n"
            "assert 'numba' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


@unittest.skipIf(not _optionals.HAS_NUMBA, "numba not available.")
class TestHypervolumeNumba(OptimizationTestCase):
    """Tests that need numba. They agree with the NumPy code and with inclusion-exclusion."""

    def test_hand_computed_values(self):
        """Hand-computed values through the numba path."""
        self.assertAlmostEqual(
            hypervolume([[0.0, 0.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.0]], np.ones(4), sense="min"),
            0.5625,
        )
        self.assertAlmostEqual(
            hypervolume(
                [[0.0, 0.0, 0.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.5, 0.0]], np.ones(5), sense="min"
            ),
            0.53125,
        )

    def test_agrees_with_numpy_implementation(self):
        """The compiled code and the NumPy code agree for 1 to 6 objectives."""
        rng = np.random.default_rng(21)
        for n_objectives in (1, 2, 3, 4, 5, 6):
            with self.subTest(n_objectives=n_objectives):
                values = _random_front(rng, 1 if n_objectives == 1 else 25, n_objectives)
                reference = np.ones(n_objectives)
                self.assertAlmostEqual(
                    _MODULE._hypervolume_numba(values, reference),  # pylint: disable=protected-access
                    _MODULE._hypervolume_slicing(values, reference),  # pylint: disable=protected-access
                )

    def test_agrees_with_brute_force_on_dominated_input(self):
        """Dominated points in the input do not change the numba result."""
        rng = np.random.default_rng(8)
        for n_objectives in (4, 5, 6):
            with self.subTest(n_objectives=n_objectives):
                front = _random_front(rng, 5, n_objectives)
                extra = front[rng.integers(0, 5, 4)] + rng.uniform(0.0, 0.004, (4, n_objectives))
                values = np.vstack([front, extra])
                reference = np.ones(n_objectives)
                self.assertAlmostEqual(
                    hypervolume(values, reference, sense="min"), _brute_force(front, reference)
                )

    def test_tie_in_last_objective(self):
        """Points that share the last objective value are handled."""
        values = np.array([[0.1, 0.4, 0.2, 0.5], [0.4, 0.1, 0.3, 0.5], [0.3, 0.3, 0.1, 0.6]])
        reference = np.ones(4)
        self.assertAlmostEqual(
            hypervolume(values, reference, sense="min"), _brute_force(values, reference)
        )

    def test_maximization_through_numba(self):
        """Mirrored data gives the same value through numba."""
        rng = np.random.default_rng(2)
        values = _random_front(rng, 10, 4)
        reference = np.ones(4)
        self.assertAlmostEqual(
            hypervolume(values, reference, sense="min"),
            hypervolume(-values, -reference, sense="max"),
        )

    def test_outside_points_are_filtered_before_numba(self):
        """Points outside the reference point are excluded before the compiled code runs."""
        values = [[0.0, 0.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.0], [1.2, 0.1, 0.1, 0.1]]
        with self.assertWarnsRegex(UserWarning, "1 point"):
            result = hypervolume(values, np.ones(4), sense="min")
        self.assertAlmostEqual(result, 0.5625)


@unittest.skipIf(not _optionals.HAS_NUMBA, "numba not available.")
class TestHypervolumeNumbaSource(OptimizationTestCase):
    """Run the numba module as plain Python, so that its lines are traced by coverage."""

    def setUp(self):
        super().setUp()
        import numba  # pylint: disable=import-outside-toplevel
        from qiskit_addon_opt_mapper.metrics import (  # pylint: disable=import-outside-toplevel
            _hypervolume_numba,
        )

        self._module = _hypervolume_numba
        self.addCleanup(importlib.reload, _hypervolume_numba)
        self.addCleanup(setattr, numba.config, "DISABLE_JIT", numba.config.DISABLE_JIT)
        numba.config.DISABLE_JIT = True
        importlib.reload(_hypervolume_numba)

    def test_interpreted_code_agrees_with_numpy_implementation(self):
        """With the JIT disabled, the same code gives the NumPy result for 1 to 5 objectives."""
        rng = np.random.default_rng(5)
        for n_objectives in (1, 2, 3, 4, 5):
            with self.subTest(n_objectives=n_objectives):
                values = _random_front(rng, 1 if n_objectives == 1 else 12, n_objectives)
                reference = np.ones(n_objectives)
                self.assertAlmostEqual(
                    self._module.hypervolume_numba(values, reference),
                    _MODULE._hypervolume_slicing(values, reference),  # pylint: disable=protected-access
                )

    def test_interpreted_code_handles_ties_in_last_objective(self):
        """Points that share the last objective value add no slice."""
        values = np.array([[0.1, 0.4, 0.2, 0.5], [0.4, 0.1, 0.3, 0.5], [0.3, 0.3, 0.1, 0.6]])
        reference = np.ones(4)
        self.assertAlmostEqual(
            self._module.hypervolume_numba(values, reference),
            _brute_force(values, reference),
        )


@unittest.skipIf(not _optionals.HAS_NUMBA, "numba not available.")
class TestHypervolumeNumbaCache(OptimizationTestCase):
    """Compiling works when numba cannot write its cache."""

    def test_falls_back_to_no_cache_when_no_folder_is_writable(self):
        """If numba raises ``RuntimeError`` for ``cache=True``, the function is compiled uncached."""
        import numba  # pylint: disable=import-outside-toplevel
        from qiskit_addon_opt_mapper.metrics import (  # pylint: disable=import-outside-toplevel
            _hypervolume_numba,
        )

        real_njit = numba.njit
        calls = []

        def njit_without_cache_folder(*args, **kwargs):
            calls.append(kwargs)
            if kwargs.get("cache"):
                raise RuntimeError("cannot cache function: no locator available")
            return real_njit(*args, **kwargs)

        with mock.patch.object(_hypervolume_numba, "njit", njit_without_cache_folder):
            compiled = _hypervolume_numba._compile(  # pylint: disable=protected-access
                lambda x: x + 1.0
            )

        self.assertEqual([call["cache"] for call in calls], [True, False])
        self.assertEqual(compiled(1.0), 2.0)

    def test_tries_to_cache_first(self):
        """With a writable folder, the function is compiled with ``cache=True`` only once."""
        import numba  # pylint: disable=import-outside-toplevel
        from qiskit_addon_opt_mapper.metrics import (  # pylint: disable=import-outside-toplevel
            _hypervolume_numba,
        )

        with mock.patch.object(_hypervolume_numba, "njit", wraps=numba.njit) as recorder:
            _hypervolume_numba._compile(lambda x: x)  # pylint: disable=protected-access
        recorder.assert_called_once_with(cache=True)


if __name__ == "__main__":
    unittest.main()
