"""Loader for the Paleka et al. 2024 consistency-forecasting tuples.

Source: https://github.com/dpaleka/consistency-forecasting/tree/main/src/data/tuples
Three subsets: ``scraped`` (resolved Manifold/Metaculus, May-Aug 2024),
``newsapi`` (synthetic, 2024), ``2028`` (long-horizon synthetic, unresolved).

Each tuple file is a JSONL with one record per line. The schema differs by
checker but always includes question slots (e.g. ``P``, ``not_P`` for
NegChecker; ``P``, ``Q``, ``P_and_Q`` for AndChecker) and a ``metadata`` dict.
Each question dict carries ``id``, ``title``, ``body``, ``resolution_date``,
``question_type``, ``data_source``, ``url``, ``resolution`` (bool, may be
absent for unresolved).

This loader maps a tuple into a JCD :class:`Clique` with appropriate
:class:`Relation` plus a parallel ground-truth resolution vector for scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import Clique, Relation

log = logging.getLogger(__name__)

# Workshop scope: the 5 checkers that map cleanly to linear-constraint cliques.
# CondChecker uses a bilinear ratio constraint p(P)·p(Q|P) = p(P∧Q); we exclude
# from v1 (TODO: SOCP / projected-gradient handler).
SUPPORTED_CHECKERS = (
    "NegChecker",
    "AndChecker",
    "OrChecker",
    "ConsequenceChecker",
    "ParaphraseChecker",
)

# Question-slot keys for each supported checker (order matters: defines the
# Clique's question index ordering, which the relation's `indices` field
# refers to).
_SLOT_KEYS: dict[str, tuple[str, ...]] = {
    "NegChecker":         ("P", "not_P"),
    "AndChecker":         ("P", "Q", "P_and_Q"),
    "OrChecker":          ("P", "Q", "P_or_Q"),
    "ConsequenceChecker": ("P", "cons_P"),
    "ParaphraseChecker":  ("P", "para_P"),
}

# Logical relation each checker enforces, with indices into the slot ordering.
_CHECKER_RELATION: dict[str, Relation] = {
    "NegChecker":         Relation(type="neg",     indices=(0, 1)),
    "AndChecker":         Relation(type="and",     indices=(0, 1, 2)),
    "OrChecker":          Relation(type="or",      indices=(0, 1, 2)),
    "ConsequenceChecker": Relation(type="implies", indices=(0, 1)),
    "ParaphraseChecker":  Relation(type="equal",   indices=(0, 1)),
}


@dataclass(frozen=True)
class PalekaQuestion:
    """One forecasting question as it appears inside a Paleka tuple."""

    id: str
    title: str
    body: str
    resolution_date: str | None
    question_type: str | None
    data_source: str | None
    url: str | None
    resolution: bool | None  # may be None for unresolved (e.g. 2028 subset)
    raw: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> PalekaQuestion:
        return cls(
            id=str(d.get("id", "")),
            title=str(d.get("title", "")),
            body=str(d.get("body", "")),
            resolution_date=d.get("resolution_date"),
            question_type=d.get("question_type"),
            data_source=d.get("data_source"),
            url=d.get("url"),
            resolution=_coerce_bool(d.get("resolution")),
            raw=d,
        )


@dataclass(frozen=True)
class PalekaTuple:
    """One Paleka consistency tuple: a clique structure + resolved questions."""

    checker: str            # e.g. "NegChecker"
    subset: str             # "scraped" | "newsapi" | "2028"
    questions: tuple[PalekaQuestion, ...]
    relation: Relation
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def m(self) -> int:
        return len(self.questions)

    @property
    def resolutions(self) -> np.ndarray | None:
        """Return ground-truth outcome vector y ∈ {0,1}^m, or None if unresolved."""
        ys = [q.resolution for q in self.questions]
        if any(y is None for y in ys):
            return None
        return np.asarray([float(y) for y in ys], dtype=float)

    @property
    def is_resolved(self) -> bool:
        return self.resolutions is not None


def load_paleka_tuples(
    path: str | Path,
    checker: str,
    *,
    subset: str = "scraped",
    require_resolved: bool = True,
    max_records: int | None = None,
) -> list[PalekaTuple]:
    """Load tuples for a single checker from a Paleka JSONL file.

    Parameters
    ----------
    path : str or Path
        Either the path to the JSONL file directly OR a directory that
        contains ``{checker}.jsonl`` (e.g. ``data/paleka/scraped/``).
    checker : str
        One of :data:`SUPPORTED_CHECKERS`.
    subset : str
        Subset label (stored on each tuple; default ``"scraped"``).
    require_resolved : bool
        If True (default), drop tuples where any question lacks ``resolution``.
        Note: ConsequenceChecker's ``cons_P`` slot is typically a synthetic
        consequence without its own market resolution. With the default
        ``require_resolved=True``, ConsequenceChecker on the ``scraped``
        subset returns zero tuples; pass ``require_resolved=False`` and
        score against the resolved subset of questions only.
    max_records : int, optional
        If given, return at most this many tuples (after the filter).

    Returns
    -------
    list of PalekaTuple
    """
    if checker not in SUPPORTED_CHECKERS:
        raise ValueError(
            f"checker={checker!r} not supported (yet). Supported: {SUPPORTED_CHECKERS}"
        )

    p = Path(path)
    if p.is_dir():
        p = p / f"{checker}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"Paleka JSONL file not found: {p}")

    slots = _SLOT_KEYS[checker]
    relation = _CHECKER_RELATION[checker]

    out: list[PalekaTuple] = []
    n_skipped_unresolved = 0
    with p.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping invalid JSON at %s:%d: %s", p, lineno, e)
                continue
            try:
                qs = tuple(PalekaQuestion.from_dict(rec[k]) for k in slots)
            except KeyError as e:
                log.warning("Missing slot %s in %s:%d; skipping", e, p, lineno)
                continue

            tup = PalekaTuple(
                checker=checker,
                subset=subset,
                questions=qs,
                relation=relation,
                metadata=rec.get("metadata", {}),
            )

            if require_resolved and not tup.is_resolved:
                n_skipped_unresolved += 1
                continue

            out.append(tup)
            if max_records is not None and len(out) >= max_records:
                break

    if n_skipped_unresolved:
        log.info(
            "Loaded %d %s tuples from %s; skipped %d unresolved",
            len(out), checker, p, n_skipped_unresolved,
        )
    return out


def paleka_tuple_to_clique(
    tup: PalekaTuple, p_hat: np.ndarray | None = None
) -> Clique:
    """Convert a Paleka tuple into a JCD Clique, optionally with empirical marginals."""
    return Clique(
        m=tup.m,
        relations=[tup.relation],
        question_ids=[q.id for q in tup.questions],
        p_hat=p_hat,
    )


def _coerce_bool(x) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
        return None
    return None
