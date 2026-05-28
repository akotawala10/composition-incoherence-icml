"""Data types for JCD: cliques and logical relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

RelationType = Literal[
    "neg", "implies", "and", "or", "mutex", "equal", "partition"
]


@dataclass(frozen=True)
class Relation:
    """A logical relation among questions in a clique.

    Index conventions:
      - 'neg':       indices=(i, j) means Q_j = ¬ Q_i (so p_i + p_j = 1).
      - 'implies':   indices=(i, j) means Q_i ⇒ Q_j (so p_i ≤ p_j).
      - 'mutex':     indices=(i, j) means Q_i and Q_j are mutually exclusive
                     (p_i + p_j ≤ 1).
      - 'equal':     indices=(i, j) means Q_i ≡ Q_j (so p_i = p_j); used for
                     paraphrase checks.
      - 'and':       indices=(i, j, k) means Q_k = Q_i ∧ Q_j.
      - 'or':        indices=(i, j, k) means Q_k = Q_i ∨ Q_j.
      - 'partition': indices=(i_1, ..., i_n) means Q_{i_1} ⊕ ... ⊕ Q_{i_n} is
                     a partition of the outcome space (Σ p_{i_k} = 1, all
                     non-negative). Used for multi-candidate Polymarket events
                     where exactly one market resolves YES and the rest NO.
    """

    type: RelationType
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.type == "partition":
            if len(self.indices) < 2:
                raise ValueError(
                    f"partition expects >= 2 indices, got {len(self.indices)}"
                )
        else:
            expected = {
                "neg": 2, "implies": 2, "mutex": 2, "equal": 2, "and": 3, "or": 3,
            }[self.type]
            if len(self.indices) != expected:
                raise ValueError(
                    f"Relation '{self.type}' expects {expected} indices, got {len(self.indices)}"
                )
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"Relation indices must be distinct: {self.indices}")


@dataclass
class Clique:
    """A logically-related group of binary questions plus optional empirical marginals.

    Attributes:
        m: number of questions in the clique.
        relations: list of logical relations among the questions.
        question_ids: optional human-readable IDs (length m).
        p_hat: optional K-sample empirical marginal vector (shape (m,)).
    """

    m: int
    relations: list[Relation] = field(default_factory=list)
    question_ids: list[str] | None = None
    p_hat: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError(f"Clique must have m >= 2, got {self.m}")
        if self.question_ids is not None and len(self.question_ids) != self.m:
            raise ValueError("question_ids length must equal m")
        if self.p_hat is not None:
            self.p_hat = np.asarray(self.p_hat, dtype=float)
            if self.p_hat.shape != (self.m,):
                raise ValueError(f"p_hat must have shape ({self.m},), got {self.p_hat.shape}")
        for rel in self.relations:
            if max(rel.indices) >= self.m:
                raise ValueError(f"Relation {rel} references index out of range for m={self.m}")
