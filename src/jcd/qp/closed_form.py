"""Closed-form L2 projections onto small coherent-marginal polytopes.

Each function takes an empirical marginal vector ``p_hat`` and returns its
Euclidean projection onto the polytope ``M_C`` of feasible coherent marginals
for a specific small clique structure. These are used both as fast paths in
the dispatcher and as gold-standard unit tests against the general QP solver.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Size-2: negation Q_2 = ¬ Q_1
# ---------------------------------------------------------------------------

def project_neg(p_hat: np.ndarray) -> np.ndarray:
    """L2 projection onto {(p, 1-p) : p ∈ [0, 1]}.

    Closed form (eq. 1 in the paper):
        p_1* = (1 + p̂_1 - p̂_2) / 2
        p_2* = 1 - p_1*
    The unconstrained projection always lies in [0, 1] when p_hat ∈ [0, 1]^2,
    but we clip defensively in case of numerical noise.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (2,):
        raise ValueError(f"project_neg requires shape (2,), got {p_hat.shape}")
    p1_star = 0.5 * (1.0 + p_hat[0] - p_hat[1])
    p1_star = float(np.clip(p1_star, 0.0, 1.0))
    return np.array([p1_star, 1.0 - p1_star])


# ---------------------------------------------------------------------------
# Size-2: implication Q_1 ⇒ Q_2  (i.e. p_1 ≤ p_2)
# ---------------------------------------------------------------------------

def project_implies(p_hat: np.ndarray) -> np.ndarray:
    """L2 projection onto {(p_1, p_2) ∈ [0, 1]^2 : p_1 ≤ p_2}.

    If the input already satisfies p_1 ≤ p_2, return it (clipped to the box).
    Otherwise project onto the diagonal hyperplane p_1 = p_2 (which gives
    the arithmetic mean for both coordinates), then clip.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (2,):
        raise ValueError(f"project_implies requires shape (2,), got {p_hat.shape}")
    p1, p2 = p_hat
    if p1 <= p2:
        out = np.clip(p_hat, 0.0, 1.0)
        return out.copy()
    # Project onto diagonal {p_1 = p_2}
    mid = 0.5 * (p1 + p2)
    mid = float(np.clip(mid, 0.0, 1.0))
    return np.array([mid, mid])


# ---------------------------------------------------------------------------
# Partition: Σ p_i = 1, all p_i ≥ 0  (multi-candidate Polymarket events)
# ---------------------------------------------------------------------------

def project_partition(p_hat: np.ndarray) -> np.ndarray:
    """L2 projection onto the probability simplex {p ∈ [0,1]^n : Σ p_i = 1}.

    Implements the O(n log n) sort-based algorithm of Wang & Carreira-Perpiñán
    (2013) "Projection onto the Probability Simplex". For an N-candidate
    Polymarket event this is dramatically cheaper than encoding N(N-1)/2
    pairwise mutex constraints in a general QP.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.ndim != 1:
        raise ValueError(f"project_partition requires 1-D input; got {p_hat.shape}")
    n = p_hat.shape[0]
    if n < 2:
        raise ValueError(f"partition needs n >= 2; got {n}")
    u = np.sort(p_hat)[::-1]                    # descending sort
    cssv = np.cumsum(u) - 1.0
    rho_candidates = u - cssv / np.arange(1, n + 1)
    rho_idx = np.where(rho_candidates > 0)[0]
    if rho_idx.size == 0:
        # All-zero edge case: project to uniform.
        return np.full(n, 1.0 / n)
    rho = int(rho_idx[-1])                      # last positive index
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(p_hat - theta, 0.0)


# ---------------------------------------------------------------------------
# Size-2: equality / paraphrase  Q_1 ≡ Q_2  →  p_1 = p_2
# ---------------------------------------------------------------------------

def project_equal(p_hat: np.ndarray) -> np.ndarray:
    """L2 projection onto {(p, p) : p ∈ [0, 1]}.

    Closed form: project to the diagonal, p_1* = p_2* = (p̂_1 + p̂_2) / 2,
    then clip to [0, 1]. Used for paraphrase consistency checks.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (2,):
        raise ValueError(f"project_equal requires shape (2,), got {p_hat.shape}")
    mid = float(np.clip(0.5 * (p_hat[0] + p_hat[1]), 0.0, 1.0))
    return np.array([mid, mid])


# ---------------------------------------------------------------------------
# Size-2: mutual exclusion (Q_1 ∧ Q_2 impossible)  —  p_1 + p_2 ≤ 1
# ---------------------------------------------------------------------------

def project_mutex(p_hat: np.ndarray) -> np.ndarray:
    """L2 projection onto {(p_1, p_2) ∈ [0, 1]^2 : p_1 + p_2 ≤ 1}.

    If p_1 + p_2 ≤ 1 already, clip to box and return. Otherwise project onto
    the half-space boundary p_1 + p_2 = 1, then clip.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (2,):
        raise ValueError(f"project_mutex requires shape (2,), got {p_hat.shape}")
    s = p_hat[0] + p_hat[1]
    if s <= 1.0:
        return np.clip(p_hat, 0.0, 1.0)
    excess = (s - 1.0) / 2.0
    out = p_hat - excess
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Size-3: conjunction Q_3 = Q_1 ∧ Q_2  (Fréchet-clipping projection)
# ---------------------------------------------------------------------------

def project_and(p_hat: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """L2 projection onto the Fréchet polytope for Q_3 = Q_1 ∧ Q_2.

    M_C = { (p_1, p_2, p_3) ∈ [0,1]^3
            : max(0, p_1 + p_2 - 1) ≤ p_3 ≤ min(p_1, p_2) }.

    The polytope is defined by 6 box constraints + 4 linear inequalities:
        p_3 ≤ p_1,   p_3 ≤ p_2,   p_3 ≥ p_1 + p_2 - 1,   p_3 ≥ 0.
    We solve the projection via Dykstra's cyclic projection algorithm onto the
    individual halfspaces; this converges to the L2 projection onto the
    intersection.

    Parameters
    ----------
    p_hat : np.ndarray, shape (3,)
        Empirical marginal vector.
    max_iter : int
        Maximum Dykstra iterations.
    tol : float
        Convergence tolerance on consecutive iterate change.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (3,):
        raise ValueError(f"project_and requires shape (3,), got {p_hat.shape}")

    # Halfspace constraints in the form a^T x ≤ b
    A = np.array(
        [
            [1.0, 0.0, 0.0],   # p_1 ≤ 1
            [-1.0, 0.0, 0.0],  # p_1 ≥ 0
            [0.0, 1.0, 0.0],   # p_2 ≤ 1
            [0.0, -1.0, 0.0],  # p_2 ≥ 0
            [0.0, 0.0, 1.0],   # p_3 ≤ 1  (redundant given p_3 ≤ min(p_1,p_2) but safe)
            [0.0, 0.0, -1.0],  # p_3 ≥ 0
            [-1.0, 0.0, 1.0],  # p_3 ≤ p_1
            [0.0, -1.0, 1.0],  # p_3 ≤ p_2
            [1.0, 1.0, -1.0],  # p_3 ≥ p_1 + p_2 - 1  ⇔  p_1 + p_2 - p_3 ≤ 1
        ]
    )
    b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    return _dykstra_halfspaces(p_hat, A, b, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
# Size-3: disjunction Q_3 = Q_1 ∨ Q_2  (Fréchet-clipping projection)
# ---------------------------------------------------------------------------

def project_or(p_hat: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """L2 projection onto the Fréchet polytope for Q_3 = Q_1 ∨ Q_2.

    Using P(A ∨ B) = P(A) + P(B) - P(A ∧ B) and the conjunction Fréchet bounds
    on P(A ∧ B), the disjunction polytope is:
        max(p_1, p_2) ≤ p_3 ≤ min(1, p_1 + p_2),  p ∈ [0, 1]^3.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.shape != (3,):
        raise ValueError(f"project_or requires shape (3,), got {p_hat.shape}")

    A = np.array(
        [
            [1.0, 0.0, 0.0],   # p_1 ≤ 1
            [-1.0, 0.0, 0.0],  # p_1 ≥ 0
            [0.0, 1.0, 0.0],   # p_2 ≤ 1
            [0.0, -1.0, 0.0],  # p_2 ≥ 0
            [0.0, 0.0, 1.0],   # p_3 ≤ 1
            [0.0, 0.0, -1.0],  # p_3 ≥ 0
            [1.0, 0.0, -1.0],  # p_3 ≥ p_1
            [0.0, 1.0, -1.0],  # p_3 ≥ p_2
            [-1.0, -1.0, 1.0], # p_3 ≤ p_1 + p_2
        ]
    )
    b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    return _dykstra_halfspaces(p_hat, A, b, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
# Internal: Dykstra's algorithm for projection onto an intersection of halfspaces
# ---------------------------------------------------------------------------

def _project_halfspace(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Project x onto the closed halfspace {y : a·y ≤ b}."""
    excess = float(a @ x) - b
    if excess <= 0.0:
        return x
    return x - (excess / float(a @ a)) * a


def _dykstra_halfspaces(
    x0: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int = 200, tol: float = 1e-12
) -> np.ndarray:
    """Dykstra's cyclic projection algorithm onto the intersection ∩_i {y : A[i]·y ≤ b[i]}.

    Standard reference: Boyle & Dykstra (1986). For polyhedra (each constraint
    a closed halfspace) Dykstra converges to the unique L2 projection. For
    small problems (n ≤ 10 constraints) it converges in << 100 iterations to
    machine precision.
    """
    x = x0.astype(float).copy()
    n_constraints = A.shape[0]
    increments = [np.zeros_like(x) for _ in range(n_constraints)]

    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n_constraints):
            y = x + increments[i]
            x_new = _project_halfspace(y, A[i], float(b[i]))
            increments[i] = y - x_new
            x = x_new
        if np.max(np.abs(x - x_prev)) < tol:
            break
    return x
