"""General L2 projection of empirical marginals onto coherent-marginal polytope.

Dispatches to fast closed-form routines when the clique structure permits;
falls back to a general convex QP solved with cvxpy + OSQP for arbitrary
clique structure.
"""

from __future__ import annotations

import numpy as np

from ..types import Clique
from . import closed_form as cf
from .constraints import compile_constraints, compile_equalities


def project(clique: Clique, p_hat: np.ndarray | None = None) -> np.ndarray:
    """Project ``p_hat`` (or ``clique.p_hat``) onto M_C in squared error.

    Uses the closed-form fast path when the clique structure is one of:
        - size-2 with a single 'neg' relation,
        - size-2 with a single 'implies' relation,
        - size-2 with a single 'mutex' relation,
        - size-3 with a single 'and' relation,
        - size-3 with a single 'or' relation.

    Otherwise solves the general QP via cvxpy + OSQP.

    Parameters
    ----------
    clique : Clique
        The logical clique. ``clique.p_hat`` is used if ``p_hat`` is None.
    p_hat : np.ndarray of shape (m,), optional
        Empirical marginal vector. If None, uses ``clique.p_hat``.

    Returns
    -------
    np.ndarray of shape (m,)
        Projected marginals in M_C.
    """
    if p_hat is None:
        if clique.p_hat is None:
            raise ValueError("Either p_hat must be passed or clique.p_hat must be set.")
        p_hat = clique.p_hat
    p_hat = np.asarray(p_hat, dtype=float).copy()
    if p_hat.shape != (clique.m,):
        raise ValueError(f"p_hat shape {p_hat.shape} does not match clique m={clique.m}")

    fast = _try_closed_form(clique, p_hat)
    if fast is not None:
        return fast
    return _solve_general_qp(clique, p_hat)


def _try_closed_form(clique: Clique, p_hat: np.ndarray) -> np.ndarray | None:
    """Return closed-form projection when applicable, else None."""
    if len(clique.relations) != 1:
        return None
    rel = clique.relations[0]

    if clique.m == 2 and rel.type == "neg" and rel.indices == (0, 1):
        return cf.project_neg(p_hat)
    if clique.m == 2 and rel.type == "implies" and rel.indices == (0, 1):
        return cf.project_implies(p_hat)
    if clique.m == 2 and rel.type == "mutex" and rel.indices == (0, 1):
        return cf.project_mutex(p_hat)
    if clique.m == 2 and rel.type == "equal" and rel.indices == (0, 1):
        return cf.project_equal(p_hat)
    if (
        rel.type == "partition"
        and len(rel.indices) == clique.m
        and rel.indices == tuple(range(clique.m))
    ):
        return cf.project_partition(p_hat)
    if clique.m == 3 and rel.type == "and" and rel.indices == (0, 1, 2):
        return cf.project_and(p_hat)
    if clique.m == 3 and rel.type == "or" and rel.indices == (0, 1, 2):
        return cf.project_or(p_hat)
    return None


def _solve_general_qp(clique: Clique, p_hat: np.ndarray) -> np.ndarray:
    """Solve  min ||r - p_hat||^2  s.t.  A r ≤ b,  A_eq r = b_eq."""
    import cvxpy as cp

    A_ineq, b_ineq = compile_constraints(clique)
    A_eq, b_eq = compile_equalities(clique)

    r = cp.Variable(clique.m)
    objective = cp.Minimize(cp.sum_squares(r - p_hat))
    constraints = [A_ineq @ r <= b_ineq]
    if A_eq.shape[0] > 0:
        constraints.append(A_eq @ r == b_eq)

    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.OSQP,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=50_000,
        polishing=True,
        verbose=False,
        warm_start=True,
    )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"QP did not converge: status={prob.status} "
            f"(clique m={clique.m}, relations={clique.relations})"
        )

    out = np.asarray(r.value, dtype=float).flatten()
    # Defensive numerical clip: tiny KKT residuals can push us 1e-10 outside [0,1].
    return np.clip(out, 0.0, 1.0)


def kkt_residual(clique: Clique, projected: np.ndarray) -> float:
    """Maximum constraint violation of the projected vector against M_C.

    Useful as the empirical bound ε in Corollary 1 (no-arbitrage PnL bound).
    """
    A_ineq, b_ineq = compile_constraints(clique)
    A_eq, b_eq = compile_equalities(clique)
    ineq_violation = np.max(np.maximum(A_ineq @ projected - b_ineq, 0.0)) if A_ineq.size else 0.0
    eq_violation = np.max(np.abs(A_eq @ projected - b_eq)) if A_eq.size else 0.0
    return float(max(ineq_violation, eq_violation))
