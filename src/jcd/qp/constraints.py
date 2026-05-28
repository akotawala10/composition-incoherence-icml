"""Compile lists of logical Relations into linear-inequality constraints
defining the polytope ``M_C`` of feasible coherent marginal vectors.

Output convention: pair (A, b) such that the polytope is {r ∈ R^m : A r ≤ b}.
The standard box constraints r ∈ [0, 1]^m are always included.
"""

from __future__ import annotations

import numpy as np

from ..types import Clique, Relation


def compile_constraints(clique: Clique) -> tuple[np.ndarray, np.ndarray]:
    """Compile a Clique's relations into halfspace inequalities A r ≤ b.

    Returns
    -------
    (A, b) : (np.ndarray of shape (n_constraints, m), np.ndarray of shape (n_constraints,))
        Each row encodes one halfspace constraint a_i^T r ≤ b_i.
        Box constraints r_i ∈ [0, 1] are always included.
    """
    m = clique.m
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # Box constraints: 0 ≤ r_i ≤ 1
    for i in range(m):
        e_i = np.zeros(m)
        e_i[i] = 1.0
        rows.append(e_i.copy())
        rhs.append(1.0)
        rows.append(-e_i.copy())
        rhs.append(0.0)

    # Logical-relation constraints
    for rel in clique.relations:
        rows_b = _relation_to_rows(rel, m)
        for row, val in rows_b:
            rows.append(row)
            rhs.append(val)

    return np.asarray(rows), np.asarray(rhs)


def compile_equalities(clique: Clique) -> tuple[np.ndarray, np.ndarray]:
    """Compile equality constraints A_eq r = b_eq from negation relations.

    Negation Q_j = ¬ Q_i is a hard equality p_i + p_j = 1, which we extract
    here so general-QP solvers can leverage it as an exact equality rather
    than two opposing inequalities.
    """
    m = clique.m
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for rel in clique.relations:
        if rel.type == "neg":
            i, j = rel.indices
            row = np.zeros(m)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(1.0)
        elif rel.type == "equal":
            i, j = rel.indices
            row = np.zeros(m)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
            rhs.append(0.0)
        elif rel.type == "partition":
            row = np.zeros(m)
            for i in rel.indices:
                row[i] = 1.0
            rows.append(row)
            rhs.append(1.0)

    if not rows:
        return np.zeros((0, m)), np.zeros((0,))
    return np.asarray(rows), np.asarray(rhs)


def _relation_to_rows(rel: Relation, m: int) -> list[tuple[np.ndarray, float]]:
    """Translate a single Relation into halfspace rows (a, b) with a^T r ≤ b."""
    if rel.type == "neg":
        # p_i + p_j = 1 is also encoded as two inequalities for solvers that
        # cannot consume equalities; compile_equalities() returns the equality.
        i, j = rel.indices
        row_pos = np.zeros(m)
        row_pos[i] = 1.0
        row_pos[j] = 1.0
        # p_i + p_j ≤ 1
        a1 = (row_pos.copy(), 1.0)
        # -(p_i + p_j) ≤ -1
        a2 = (-row_pos.copy(), -1.0)
        return [a1, a2]

    if rel.type == "equal":
        # p_i = p_j  encoded as two opposing inequalities
        i, j = rel.indices
        row_pos = np.zeros(m); row_pos[i] = 1.0; row_pos[j] = -1.0
        return [(row_pos.copy(), 0.0), (-row_pos.copy(), 0.0)]

    if rel.type == "implies":
        # Q_i ⇒ Q_j  ⇔  p_i ≤ p_j  ⇔  p_i - p_j ≤ 0
        i, j = rel.indices
        row = np.zeros(m)
        row[i] = 1.0
        row[j] = -1.0
        return [(row, 0.0)]

    if rel.type == "mutex":
        # p_i + p_j ≤ 1
        i, j = rel.indices
        row = np.zeros(m)
        row[i] = 1.0
        row[j] = 1.0
        return [(row, 1.0)]

    if rel.type == "and":
        # Q_k = Q_i ∧ Q_j: Fréchet bounds on p_k.
        #   p_k ≤ p_i      ⇔   -p_i + p_k ≤ 0
        #   p_k ≤ p_j      ⇔   -p_j + p_k ≤ 0
        #   p_k ≥ p_i+p_j-1 ⇔   p_i + p_j - p_k ≤ 1
        i, j, k = rel.indices
        rows = []
        r1 = np.zeros(m); r1[i] = -1.0; r1[k] = 1.0
        rows.append((r1, 0.0))
        r2 = np.zeros(m); r2[j] = -1.0; r2[k] = 1.0
        rows.append((r2, 0.0))
        r3 = np.zeros(m); r3[i] = 1.0; r3[j] = 1.0; r3[k] = -1.0
        rows.append((r3, 1.0))
        return rows

    if rel.type == "partition":
        # Σ p_{i_k} ≤ 1   AND   Σ p_{i_k} ≥ 1   (encoded as two inequalities;
        # callers that want the equality form should use compile_equalities).
        row = np.zeros(m)
        for i in rel.indices:
            row[i] = 1.0
        return [(row.copy(), 1.0), (-row.copy(), -1.0)]

    if rel.type == "or":
        # Q_k = Q_i ∨ Q_j  with  P(Q_i ∨ Q_j) = p_i + p_j - P(Q_i ∧ Q_j).
        # Fréchet bounds give:
        #   p_k ≥ p_i           ⇔  p_i - p_k ≤ 0
        #   p_k ≥ p_j           ⇔  p_j - p_k ≤ 0
        #   p_k ≤ p_i + p_j     ⇔  -p_i - p_j + p_k ≤ 0
        i, j, k = rel.indices
        rows = []
        r1 = np.zeros(m); r1[i] = 1.0; r1[k] = -1.0
        rows.append((r1, 0.0))
        r2 = np.zeros(m); r2[j] = 1.0; r2[k] = -1.0
        rows.append((r2, 0.0))
        r3 = np.zeros(m); r3[i] = -1.0; r3[j] = -1.0; r3[k] = 1.0
        rows.append((r3, 0.0))
        return rows

    raise ValueError(f"Unknown relation type: {rel.type}")
