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

"""Test to_ising"""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper import MultiObjectiveProblem, OptimizationError, OptimizationProblem
from qiskit_addon_opt_mapper.converters import EqualityToPenalty
from qiskit_addon_opt_mapper.problems import Constraint, ObjSense, OptimizationObjective
from qiskit_addon_opt_mapper.translators import to_ising, to_isings
from qiskit_addon_opt_mapper.translators.ising import _objective_to_ising

from ..optimization_test_case import OptimizationTestCase

QUBIT_OP_MAXIMIZE_SAMPLE = SparsePauliOp.from_list(
    [
        ("IIIZ", -199999.5),
        ("IIZI", -399999.5),
        ("IZII", -599999.5),
        ("ZIII", -799999.5),
        ("IIZZ", 100000),
        ("IZIZ", 150000),
        ("IZZI", 300000),
        ("ZIIZ", 200000),
        ("ZIZI", 400000),
        ("ZZII", 600000),
    ]
)
OFFSET_MAXIMIZE_SAMPLE = 1149998


class TestIsingTranslator(OptimizationTestCase):
    """Test to_ising"""

    def test_to_ising(self):
        """test to_ising"""

        with self.subTest("minimize"):
            # minimize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(linear={"x": 1}, quadratic={("x", "y"): 1})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZI", -0.25), ("IZ", -0.75), ("ZZ", 0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.75)

        with self.subTest("maximize"):
            # maximize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(linear={"x": 1}, quadratic={("x", "y"): 1})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZI", 0.25), ("IZ", 0.75), ("ZZ", -0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, -0.75)

    def test_to_ising2(self):
        """test to_ising 2"""

        with self.subTest("minimize"):
            # minimize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZZ", 1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

        with self.subTest("maximize"):
            # maximize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZZ", -1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

    def test_to_ising_wo_variable(self):
        """test to_ising with problems without variables"""
        with self.subTest("empty problem"):
            q_p = OptimizationProblem()
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 0)

        with self.subTest("min 3"):
            q_p = OptimizationProblem()
            q_p.minimize(constant=3)
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 3)

        with self.subTest("max -1"):
            q_p = OptimizationProblem()
            q_p.maximize(constant=-1)
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 1)

    def test_optimizationproblem_to_ising(self):
        """Test optimization problem to operators"""
        op = OptimizationProblem()
        for i in range(4):
            op.binary_var(name=f"x{i}")
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.maximize(0, linear, {})
        linear = {}
        for i, x in enumerate(op.variables):
            linear[x.name] = i + 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 3, "sum1")
        penalize = EqualityToPenalty(penalty=1e5)
        op2 = penalize.convert(op)
        qubitop, offset = op2.to_ising()
        self.assertTrue(qubitop.equiv(QUBIT_OP_MAXIMIZE_SAMPLE))
        self.assertEqual(offset, OFFSET_MAXIMIZE_SAMPLE)

    def test_valid_variable_type(self):
        """Validate the types of the variables for QuadraticProgram.to_ising."""
        # Integer variable
        with self.assertRaises(OptimizationError):
            op = OptimizationProblem()
            op.integer_var(0, 10, "int_var")
            _ = op.to_ising()
        # Continuous variable
        with self.assertRaises(OptimizationError):
            op = OptimizationProblem()
            op.continuous_var(0, 10, "continuous_var")
            _ = op.to_ising()

    def test_to_ising_higher_order(self):
        """test to_ising with higher order terms"""

        with self.subTest("cubic term"):
            # minimize: x0 * x1 * x2
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.minimize(
                higher_order={3: {(0, 1, 2): 2}},
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIZ", -0.25),
                    ("IZI", -0.25),
                    ("ZII", -0.25),
                    ("IZZ", 0.25),
                    ("ZIZ", 0.25),
                    ("ZZI", 0.25),
                    ("ZZZ", -0.25),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.25)

        with self.subTest("cubic term 2"):
            # minimize: 2 * x0^2 * x1 + 3 * x1 * x2^2
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.minimize(
                higher_order={
                    3: {
                        (0, 0, 1): 2.0,
                        (1, 2, 2): 3.0,
                    }
                },
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIZ", -0.5),
                    ("IZI", -1.25),
                    ("ZII", -0.75),
                    ("IZZ", 0.5),
                    ("ZZI", 0.75),
                    ("III", 0.625),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.625)

        with self.subTest("cubic and quartic terms"):
            # minimize: x0 * x1 * x2 + x0 * x1 * x2 * x3
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.binary_var(name="x3")
            o_p.minimize(
                higher_order={
                    3: {(0, 1, 2): 1},
                    4: {(0, 1, 2, 3): 1},
                }
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIIZ", -0.1875),
                    ("IIZI", -0.1875),
                    ("IZII", -0.1875),
                    ("ZIII", -0.0625),
                    ("IIZZ", 0.1875),
                    ("IZIZ", 0.1875),
                    ("IZZI", 0.1875),
                    ("ZIIZ", 0.0625),
                    ("ZIZI", 0.0625),
                    ("ZZII", 0.0625),
                    ("IZZZ", -0.1875),
                    ("ZIZZ", -0.0625),
                    ("ZZIZ", -0.0625),
                    ("ZZZI", -0.0625),
                    ("ZZZZ", 0.0625),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.1875)

    def test_objective_to_ising_translates_the_given_objective(self):
        """`_objective_to_ising` translates the objective it is given, not the problem's own."""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.minimize(linear={"x": 1})
        other = OptimizationObjective(op, linear={"y": 2}, constant=1, sense=ObjSense.MAXIMIZE)
        qubit_op, offset = _objective_to_ising(op, other)
        self.assertTrue(qubit_op.equiv(SparsePauliOp.from_list([("ZI", 1.0)])))
        self.assertEqual(offset, -2.0)
        self.assertEqual(to_ising(op), _objective_to_ising(op, op.objective))


def _make_problem(sense: str = "minimize", name: str = "") -> OptimizationProblem:
    problem = OptimizationProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    getattr(problem, sense)(constant=2, linear={"x": 1}, quadratic={("x", "y"): 1}, name=name)
    return problem


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_to_ising_ignores_objective_name(sense):
    """Naming the objective does not change the Hamiltonian or the offset."""
    assert _make_problem(sense, "").to_ising() == _make_problem(sense, "f0").to_ising()


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_objective_to_ising_matches_to_ising(sense):
    """``_objective_to_ising`` on the problem objective equals ``to_ising``."""
    problem = _make_problem(sense, "f0")
    assert _objective_to_ising(problem, problem.objective) == to_ising(problem)


def test_objective_to_ising_uses_given_objective_sense():
    """The sign of the Hamiltonian follows the sense of the objective that is passed in."""
    minimized = _make_problem("minimize", "f0")
    maximized = _make_problem("maximize", "f0")

    min_op, min_offset = _objective_to_ising(minimized, minimized.objective)
    max_op, max_offset = _objective_to_ising(minimized, maximized.objective)

    assert max_op == -min_op
    assert max_offset == -min_offset


def test_objective_to_ising_rejects_constraints():
    """The constraint check of ``to_ising`` is applied by ``_objective_to_ising``."""
    problem = _make_problem()
    problem.linear_constraint({"x": 1}, "==", 1)
    with pytest.raises(OptimizationError):
        _objective_to_ising(problem, problem.objective)


def test_to_ising_spin_variables():
    """``to_ising`` translates spin variables with constant, linear, quadratic, and higher-order terms."""
    problem = OptimizationProblem("p")
    problem.spin_var("s0")
    problem.spin_var("s1")
    problem.spin_var("s2")
    problem.minimize(
        constant=1.0,
        linear={"s0": 2.0},
        quadratic={("s0", "s1"): 3.0, ("s1", "s1"): 0.5},
        higher_order={3: {("s0", "s1", "s2"): 4.0}},
    )
    op, offset = problem.to_ising()
    assert isinstance(op, SparsePauliOp)
    assert offset == 1.5


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_to_isings_single_objective_is_one_element_list(sense):
    """``to_isings`` of an ``OptimizationProblem`` is a one-element list equal to ``to_ising``."""
    problem = _make_problem(sense, "f0")
    assert to_isings(problem) == [to_ising(problem)]


def test_to_isings_multi_objective_order_and_sense():
    """``to_isings`` returns one pair per objective, in objective order, with sense applied."""
    problem = MultiObjectiveProblem("p")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.add_objective(constant=2, linear={"x": 1}, quadratic={("x", "y"): 1})
    problem.add_objective(constant=2, linear={"x": 1}, quadratic={("x", "y"): 1}, sense="max")
    (min_op, min_offset), (max_op, max_offset) = to_isings(problem)
    assert max_op == -min_op
    assert max_offset == -min_offset
    assert (min_op, min_offset) == to_ising(_make_problem("minimize"))


def test_to_isings_empty_multi_objective_problem():
    """A problem without objectives gives an empty list."""
    assert to_isings(MultiObjectiveProblem("p")) == []


def test_to_isings_rejects_constraints_per_objective():
    """The constraint guard is applied for every objective."""
    problem = MultiObjectiveProblem("p")
    problem.binary_var("x")
    problem.add_objective(linear={"x": 1})
    problem.linear_constraint({"x": 1}, "==", 1)
    with pytest.raises(OptimizationError):
        to_isings(problem)


def test_to_isings_without_objectives_rejects_constraints():
    """Zero-objective problems with constraints raise OptimizationError."""
    problem = MultiObjectiveProblem("p")
    problem.binary_var("x")
    problem.linear_constraint({"x": 1}, "==", 1)
    with pytest.raises(OptimizationError, match="There must be no constraint"):
        to_isings(problem)


def test_to_isings_rejects_higher_order_constraints():
    """A higher-order constraint is rejected like a linear or quadratic one."""
    problem = MultiObjectiveProblem("p")
    for name in "xyz":
        problem.binary_var(name)
    problem.add_objective(linear={"x": 1})
    problem.higher_order_constraint(higher_order={3: {(0, 1, 2): 1}}, sense="==", rhs=1)
    with pytest.raises(OptimizationError, match="There must be no constraint"):
        to_isings(problem)


def test_to_ising_rejects_higher_order_constraints():
    """``to_ising`` rejects higher-order constraints for a single-objective problem."""
    problem = OptimizationProblem("p")
    for name in "xyz":
        problem.binary_var(name)
    problem.minimize(linear={"x": 1})
    problem.higher_order_constraint(higher_order={3: {(0, 1, 2): 1}}, sense="==", rhs=1)
    with pytest.raises(OptimizationError, match="There must be no constraint"):
        to_ising(problem)


def test_to_isings_continuous_variable_message_does_not_suggest_qubo():
    """The error says continuous variables are unsupported instead of pointing to the QUBO converter."""
    problem = MultiObjectiveProblem("p")
    problem.continuous_var(name="c")
    with pytest.raises(OptimizationError, match="Continuous variables are not supported") as err:
        to_isings(problem)
    assert "OptimizationProblemToQubo" not in str(err.value)


def test_to_isings_without_objectives_rejects_continuous_or_integer_variables():
    """Zero-objective problems with continuous/integer variables raise OptimizationError."""
    problem_cont = MultiObjectiveProblem("p")
    problem_cont.continuous_var(name="c")
    with pytest.raises(OptimizationError, match="must contain either only binary variables"):
        to_isings(problem_cont)

    problem_int = MultiObjectiveProblem("p")
    problem_int.integer_var(name="i")
    with pytest.raises(OptimizationError, match="must contain either only binary variables"):
        to_isings(problem_int)
