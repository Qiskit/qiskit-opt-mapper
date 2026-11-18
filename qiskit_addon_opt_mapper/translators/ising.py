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

"""Translator between an Ising Hamiltonian and a optimization problem."""

import itertools

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp

from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems.base_optimization_problem import BaseOptimizationProblem
from qiskit_addon_opt_mapper.problems.optimization_objective import OptimizationObjective
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


def to_ising(optimization_problem: OptimizationProblem) -> tuple[SparsePauliOp, float]:
    """Return the Ising Hamiltonian of this problem.

    Variables are mapped to qubits in qiskit order, i.e.,
    i-th variable is mapped to index n-i in the pauli string, where
    n is the total number of variables.

    Args:
        optimization_problem: The problem to be translated.

    Returns:
        A tuple (qubit_op, offset) comprising the qubit operator for the problem
        and offset for the constant value in the Ising Hamiltonian.

    Raises:
        OptimizationError: If variables are neither only binary nor only spin.
        OptimizationError: If constraints exist in the problem.
    """
    return _objective_to_ising(optimization_problem, optimization_problem.objective)


def _validate_problem_for_ising(problem: BaseOptimizationProblem) -> None:
    """Validate that problem has no constraints and contains only binary or spin variables."""
    if problem.linear_constraints or problem.quadratic_constraints:
        raise OptimizationError(
            "There must be no constraint in the problem. "
            "You can use `OptimizationProblemToQubo` converter "
            "to convert constraints to penalty terms of the objective function."
        )

    num_vars = problem.get_num_vars()
    if not (num_vars == problem.get_num_binary_vars() or num_vars == problem.get_num_spin_vars()):
        raise OptimizationError(
            "The problem must contain either only binary variables or only spin variables. "
            "You can use `OptimizationProblemToHubo` converter "
            "to convert integer variables to binary variables. "
            "And you can use `OptimizationProblemToQubo` converter "
            "to convert continuous variables to binary variables."
        )


def to_isings(optimization_problem: BaseOptimizationProblem) -> list[tuple[SparsePauliOp, float]]:
    """Return one Ising Hamiltonian per objective of this problem.

    Variables are mapped to qubits in qiskit order, i.e.,
    i-th variable is mapped to index n-i in the pauli string, where
    n is the total number of variables.

    Args:
        optimization_problem: The problem to be translated.

    Returns:
        A list of tuples (qubit_op, offset), one per objective of the problem in objective
        order. An :class:`~qiskit_addon_opt_mapper.OptimizationProblem` yields a list of one
        tuple.

    Raises:
        OptimizationError: If variables are neither only binary nor only spin.
        OptimizationError: If constraints exist in the problem.
    """
    _validate_problem_for_ising(optimization_problem)
    return [
        _objective_to_ising(optimization_problem, objective, validate=False)
        for objective in optimization_problem._iter_objectives()  # pylint: disable=protected-access
    ]


def _objective_to_ising(
    problem: BaseOptimizationProblem, objective: OptimizationObjective, validate: bool = True
) -> tuple[SparsePauliOp, float]:
    """Return the Ising Hamiltonian of a single objective of a problem.

    Args:
        problem: The problem providing the variables and constraints.
        objective: The objective to be translated.
        validate: Whether to validate the problem. Pass ``False`` only if the caller
            has already called :func:`_validate_problem_for_ising` on ``problem``.

    Returns:
        A tuple (qubit_op, offset) comprising the qubit operator for the objective
        and offset for the constant value in the Ising Hamiltonian.

    Raises:
        OptimizationError: If ``validate`` is true and variables are neither only binary
            nor only spin.
        OptimizationError: If ``validate`` is true and constraints exist in the problem.
    """
    if validate:
        _validate_problem_for_ising(problem)

    num_vars = problem.get_num_vars()

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = objective.sense.value

    # convert a constant part of the objective function into Hamiltonian.
    offset = objective.constant * sense

    linear = objective.linear.coefficients.tocoo()
    lin_idx, lin_coef = linear.col, linear.data * sense
    quadratic = objective.quadratic.coefficients.tocoo()
    diag = quadratic.row == quadratic.col
    quad_idx = np.column_stack((quadratic.row, quadratic.col))[~diag]
    quad_coef = quadratic.data[~diag] * sense
    diag_idx, diag_coef = quadratic.row[diag], quadratic.data[diag] * sense

    # Each block holds the variable indices of a set of Pauli Z strings, one row per
    # string, and their coefficients. A variable appearing twice in a row cancels out.
    blocks: list[tuple[np.ndarray, np.ndarray]] = []

    if num_vars == problem.get_num_binary_vars():
        # if all variables are binary variables, substitute x = (1 - z) / 2.
        # Linear terms x_i = x_i^2 come from both the linear and the diagonal quadratic
        # part. Each off-diagonal quadratic term x_i x_j contributes to Z_i and Z_j.
        z_coef = np.zeros(num_vars)
        np.add.at(z_coef, lin_idx, -lin_coef / 2)
        np.add.at(z_coef, diag_idx, -diag_coef / 2)
        np.add.at(z_coef, quad_idx.ravel(), -np.repeat(quad_coef, 2) / 4)
        offset += lin_coef.sum() / 2 + diag_coef.sum() / 2 + quad_coef.sum() / 4

        blocks.append((np.arange(num_vars)[:, None], z_coef))
        blocks.append((quad_idx, quad_coef / 4))

        # convert higher order terms of the objective function into Hamiltonian.
        # For each binary term of order k we get Pauli spins with orders ranging from 0 to k due
        # to the expansion (1-z0)(1-z1)(1-z2) ... / 2**k for a term x0x1x2..., for example.
        for degree, high_order_exp in objective.higher_order.items():
            coeffs = high_order_exp.to_dict()
            if not coeffs:
                continue
            indices = np.array(list(coeffs), dtype=int).reshape(-1, degree)
            values = np.fromiter(coeffs.values(), dtype=float, count=len(coeffs)) * sense
            for i in range(1, degree + 1):
                # enumerate all combinations of i positions in the terms. For example,
                # for the term (0, 1, 2) and i=1 this adds IIZ, IZI, and ZII.
                weight = values * (-1) ** i / 2**degree
                for comb in itertools.combinations(range(degree), i):
                    blocks.append((indices[:, comb], weight))
            offset += values.sum() / 2**degree

    elif num_vars == problem.get_num_spin_vars():
        # if all variables are spin variables, z_i^2 = 1 goes into the offset.
        blocks.append((lin_idx[:, None], lin_coef))
        blocks.append((quad_idx, quad_coef))
        offset += diag_coef.sum()

        for degree, high_order_exp in objective.higher_order.items():
            coeffs = high_order_exp.to_dict()
            if not coeffs:
                continue
            indices = np.array(list(coeffs), dtype=int).reshape(-1, degree)
            values = np.fromiter(coeffs.values(), dtype=float, count=len(coeffs)) * sense
            blocks.append((indices, values))

    else:
        raise OptimizationError(
            "The problem must contain either only binary variables or only spin variables. "
            "You can use `OptimizationProblemToHubo` converter "
            "to convert integer variables to binary variables. "
        )

    coeffs = np.concatenate([coef for _, coef in blocks])
    keep = coeffs != 0
    if not np.any(keep):
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        return SparsePauliOp("I" * max(1, num_vars), 0), float(offset)

    z = np.concatenate([_z_mask(num_vars, idx) for idx, _ in blocks])[keep]

    # The operator has only Z strings, so there is no Pauli phase for qiskit to work out.
    qubit_op = SparsePauliOp(
        PauliList.from_symplectic(z, np.zeros_like(z)),
        coeffs[keep].astype(complex),
        ignore_pauli_phase=True,
        copy=False,
    )
    if objective.higher_order:
        # Higher order terms may repeat a Pauli string, so combine them and drop zeros.
        qubit_op = qubit_op.simplify(atol=0)

    return qubit_op, float(offset)


def _z_mask(num_vars: int, indices: np.ndarray) -> np.ndarray:
    """Return the Z part of the Pauli strings given by rows of variable indices.

    Args:
        num_vars: The number of qubits.
        indices: An ``(m, k)`` array of the variable indices in each of the ``m`` strings.

    Returns:
        An ``(m, num_vars)`` boolean array. A variable that appears an even number of
        times in a row is not set since ``Z^2 = I``.
    """
    z = np.zeros((len(indices), num_vars), dtype=bool)
    rows = np.arange(len(indices))
    for col in indices.T:
        z[rows, col] ^= True
    return z
