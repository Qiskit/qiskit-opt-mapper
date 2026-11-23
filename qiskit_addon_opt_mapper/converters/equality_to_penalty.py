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

"""Converter to convert a problem with equality constraints to unconstrained with penalty terms."""

import copy
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from ..exceptions import OptimizationError
from ..problems.constraint import Constraint
from ..problems.optimization_objective import OptimizationObjective
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter

if TYPE_CHECKING:
    from ..problems.base_optimization_problem import BaseOptimizationProblem

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound="BaseOptimizationProblem")

Monomial = tuple[str, ...]  # ()=const, ('x',)=linear, ('x','y')=quadratic, etc.
Poly = dict[Monomial, float]


class EqualityToPenalty(OptimizationProblemConverter):
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self, penalty: float | None = None) -> None:
        """Init method.

        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                     If None is passed, a penalty factor will be automatically calculated on
                     every conversion.
                     The penalty factor is calculated as follows:

                         1 + (upperbound - lowerbound) of objective function.

        """
        self._src_num_vars: int | None = None
        self._penalty: float | None = penalty
        self._should_define_penalty: bool = penalty is None
        self._src: BaseOptimizationProblem | None = None
        self._dst: BaseOptimizationProblem | None = None

    def convert(self, problem: _T) -> _T:
        """Convert a problem with equality constraints into an unconstrained problem.

        Args:
            problem: The problem to be solved, that does not contain inequality constraints.


        Returns:
            The converted problem, that is an unconstrained problem.

        Raises:
            OptimizationError: If an inequality constraint exists.
        """
        if problem.is_multi_objective:
            raise OptimizationError(
                "MultiObjectiveProblem is not supported for EqualityToPenalty. "
                "In multi-objective optimization, constraints should not be folded as "
                "penalties into existing objectives."
            )

        self._src = copy.deepcopy(problem)
        self._src_num_vars = problem.get_num_vars()
        dst = problem._new_empty()
        self._dst = dst

        for var in problem.variables:
            if var.vartype == Variable.Type.BINARY:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.BINARY,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.INTEGER:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.CONTINUOUS:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.SPIN:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.SPIN,
                    lowerbound=-1,
                    upperbound=1,
                    internal=True,
                )
            else:
                raise OptimizationError(f"Unknown variable type: {var.vartype}")

        if self._should_define_penalty:
            penalty = self._auto_define_penalty(problem)
            logger.info("Automatically defined penalty factor: %f", penalty)
        else:
            assert self._penalty is not None
            penalty = self._penalty
            if penalty <= 0:
                raise OptimizationError("Penalty factor must be positive.")
            logger.info("Using user-defined penalty factor: %f", penalty)

        all_constraints = (
            list(problem.linear_constraints)
            + list(problem.quadratic_constraints)
            + list(problem.higher_order_constraints)
        )
        for con in all_constraints:
            if con.sense != Constraint.Sense.EQ:
                raise OptimizationError(
                    "An inequality constraint exists. Only equality constraints are supported."
                )

        def _add_poly_to_objective(
            poly: Poly,
            scale: float,
            obj_sense: int,
            lin_dict: dict[int | str, float],
            quad_dict: dict[tuple[int, int] | tuple[str, str], float],
            ho_dict: dict[int, dict[tuple[int, ...] | tuple[str, ...], float]],
        ) -> float:
            const, linc, quadc, hoc = _poly_split_by_degree(poly)
            for x, c in linc.items():
                if c:
                    lin_dict[x] = lin_dict.get(x, 0.0) + obj_sense * scale * c
            for (i, j), c in quadc.items():
                if c:
                    tup = (i, j)
                    quad_dict[tup] = quad_dict.get(tup, 0.0) + obj_sense * scale * c
            for d, terms in hoc.items():
                tgt = ho_dict.setdefault(d, {})
                for names, c in terms.items():
                    if c:
                        tgt[names] = tgt.get(names, 0.0) + obj_sense * scale * c
            return float(obj_sense * scale * const)

        for obj in problem.iter_objectives():
            offset = obj.constant
            linear = obj.linear.to_dict(use_name=True)
            quadratic = obj.quadratic.to_dict(use_name=True)
            ho = {d: expr.to_dict(use_name=True) for d, expr in obj.higher_order.items()}
            sense = 1 if obj.sense == OptimizationObjective.Sense.MINIMIZE else -1

            for con in all_constraints:
                c = float(con.rhs)
                f: Poly = {}
                _poly_add(f, _poly_from_constraint_linear(con))
                if getattr(con, "quadratic", None):
                    _poly_add(f, _poly_from_constraint_quadratic(con))
                if getattr(con, "higher_order", None):
                    _poly_add(f, _poly_from_constraint_higher(con))

                offset += sense * penalty * (c * c)
                offset += _add_poly_to_objective(
                    f, -2.0 * penalty * c, sense, linear, quadratic, ho
                )
                f2 = _poly_mul(f, f)
                offset += _add_poly_to_objective(f2, penalty, sense, linear, quadratic, ho)

            self._dst._append_or_set_objective_from_parts(
                obj.sense,
                constant=offset,
                linear=linear,
                quadratic=quadratic,
                higher_order=ho,  # type: ignore[arg-type]
                name=obj.name,
            )

        # Update the penalty to the one just used
        self._penalty = penalty

        return dst

    @staticmethod
    def _auto_define_penalty(problem: "BaseOptimizationProblem") -> float:
        """Automatically define the penalty coefficient.

        Returns:
            Return the minimum valid penalty factor calculated
            from the upper bound and the lower bound of the objective function.
            If a constraint has a float coefficient,
            return the default value for the penalty factor.
        """
        default_penalty = 1e5

        # Check coefficients of constraints.
        # If a constraint has a float coefficient, return the default value for the penalty factor.
        terms = []
        for constraint in problem.linear_constraints:
            terms.append(constraint.rhs)
            terms.extend(constraint.linear.to_array().tolist())
        if any(isinstance(term, float) and not term.is_integer() for term in terms):
            logger.warning(
                "Warning: Using %f for the penalty coefficient because "
                "a float coefficient exists in constraints. \n"
                "The value could be too small. "
                "If so, set the penalty coefficient manually.",
                default_penalty,
            )
            return default_penalty

        max_bound = 0.0
        for obj in problem.iter_objectives():
            lin_b = obj.linear.bounds
            quad_b = obj.quadratic.bounds
            ho_b = [ho_exp.bounds for ho_exp in obj.higher_order.values()]
            span = (
                (lin_b.upperbound - lin_b.lowerbound)
                + (quad_b.upperbound - quad_b.lowerbound)
                + sum((b.upperbound - b.lowerbound) for b in ho_b)
            )
            if span > max_bound:
                max_bound = span

        return float(1.0 + max_bound)

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

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
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)

    @property
    def penalty(self) -> float | None:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: float | None) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, a penalty factor will be automatically calculated
                     on every conversion.
        """
        self._penalty = penalty
        self._should_define_penalty = penalty is None


def _normalize_monomial(m: Monomial) -> Monomial:
    """Normalize monomial key for symmetric terms.

    - Sort variable names for a canonical key: ('y','x')->('x','y')
    - Keep duplicates as-is; we do NOT enforce z^2=z (no multilinear reduction here).
    """
    return tuple(sorted(m))


def _poly_add(dst: Poly, src: Mapping[Monomial, float], scale: float = 1.0) -> None:
    for m, c in src.items():
        if c:
            mm = _normalize_monomial(m)
            dst[mm] = dst.get(mm, 0.0) + scale * c


def _poly_mul(a: Poly, b: Poly) -> Poly:
    """Convolution of two sparse polynomials represented as {monomial: coeff}."""
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = _normalize_monomial(m1 + m2)
            out[m] = out.get(m, 0.0) + c1 * c2
    return out


def _poly_from_constraint_linear(constraint) -> Poly:
    """Build f(x) from a *linear* constraint part: sum_j a_j x_j."""
    row = constraint.linear.to_dict(use_name=True)
    return {(name,): coef for name, coef in row.items() if coef != 0.0}


def _poly_from_constraint_quadratic(constraint) -> Poly:
    """Build f(x) from *quadratic* part: sum_{ij} q_ij x_i x_j."""
    q = constraint.quadratic.to_dict(use_name=True)
    poly: Poly = {}
    for (i, j), coef in q.items():
        if coef != 0.0:
            m = _normalize_monomial((i, j))
            poly[m] = poly.get(m, 0.0) + coef
    return poly


def _poly_from_constraint_higher(constraint) -> Poly:
    """Build f(x) from higher-order parts: sum_{deg>=3} sum_{S} h_S prod_{v in S} v."""
    poly: Poly = {}
    # constraint.higher_order: dict[int, Expr]; each Expr has to_dict(use_name=True)
    for _deg, expr in getattr(constraint, "higher_order", {}).items():
        terms = expr.to_dict(use_name=True)  # dict[tuple[str,...], float]
        for names, coef in terms.items():
            if coef != 0.0:
                m = _normalize_monomial(tuple(names))
                poly[m] = poly.get(m, 0.0) + coef
    return poly


def _poly_split_by_degree(poly: Poly):
    """Split poly into constant/linear/quadratic/higher dicts for objective accumulation."""
    const = poly.get((), 0.0)
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    higher: dict[int, dict[tuple[str, ...], float]] = defaultdict(dict)
    for m, c in poly.items():
        if not m:  # ()
            continue
        d = len(m)
        if d == 1:
            (x,) = m
            linear[x] = linear.get(x, 0.0) + c
        elif d == 2:
            quadratic[(m[0], m[1])] = quadratic.get((m[0], m[1]), 0.0) + c
        else:
            hd = higher[d]
            hd[m] = hd.get(m, 0.0) + c
    return const, linear, quadratic, dict(higher)
