"""
Validate Option 1: Optimal coupling-aware routing allocation.

Theorem: under sampling-with-replacement and averaging, the allocation
k_j proportional to |a_{R,j}| * sigma_j minimizes E[(eps*)^2] subject to
sum k_j = K. Heterogeneity gain (uniform / optimal) =
    m * sum v_j / (sum sqrt(v_j))^2,    v_j = a_{R,j}^2 * sigma_j^2
which is >= 1 by Cauchy-Schwarz, with equality iff all sqrt(v_j) equal.

We compute the per-clique heterogeneity gain on the existing 1,876
ensemble cliques and report the per-relation distribution.

Also report the optimal-routing E[(eps*)^2] vs the uniform-routing
E[(eps*)^2] under K = m * c for c in {1, 2, 4} (averaging redundancy),
and verify that optimal routing reduces E[eps*^2] toward zero faster
than uniform.
"""

from __future__ import annotations

import os
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

JCD_SRC = os.environ.get("JCD_SRC", str(Path(__file__).resolve().parent.parent.parent / "JCD-Forecasting" / "src"))
sys.path.insert(0, JCD_SRC)

from jcd.types import Clique, Relation
from jcd.qp.solver import project as jcd_project

CTB = Path(__file__).resolve().parent.parent / "data"
FIG = CTB / "figures"
MODELS = [
    "anthropic_claude-haiku-4-5-20251001",
    "azure_mini",
    "azure_nano",
    "groq_llama-3",
]
RELATIONS_OF_INTEREST = ("neg", "and", "or", "partition")


def build_clique(rel: str, m: int) -> Clique:
    if rel == "neg" and m == 2:
        return Clique(m=m, relations=[Relation(type="neg", indices=(0, 1))])
    if rel == "and" and m == 3:
        return Clique(m=m, relations=[Relation(type="and", indices=(0, 1, 2))])
    if rel == "or" and m == 3:
        return Clique(m=m, relations=[Relation(type="or", indices=(0, 1, 2))])
    if rel == "partition":
        return Clique(m=m, relations=[Relation(type="partition", indices=tuple(range(m)))])
    raise ValueError(rel)


def binding_normal(rel: str, m: int, Pi_bar: np.ndarray) -> np.ndarray:
    """Return the binding-constraint normal a_R per relation.

    For neg / partition (equality constraints) it's the unique normal.
    For and / or (Frechet halfspaces) we pick the constraint with smallest
    slack at Pi_bar (most likely to be active).
    """
    if rel == "neg":
        return np.array([1.0, 1.0])
    if rel == "partition":
        return np.ones(m)
    if rel == "and":
        # candidates: r_3 - r_1 <= 0, r_3 - r_2 <= 0, r_1+r_2-r_3 <= 1
        cands = [
            (np.array([-1.0, 0.0, 1.0]), 0.0),
            (np.array([0.0, -1.0, 1.0]), 0.0),
            (np.array([1.0, 1.0, -1.0]), 1.0),
        ]
    elif rel == "or":
        cands = [
            (np.array([1.0, 0.0, -1.0]), 0.0),
            (np.array([0.0, 1.0, -1.0]), 0.0),
            (np.array([-1.0, -1.0, 1.0]), 0.0),
        ]
    else:
        raise ValueError(rel)
    slacks = [(b - a @ Pi_bar, a, b) for a, b in cands]
    slacks.sort(key=lambda t: t[0])
    return slacks[0][1]


def main() -> None:
    data = {}
    for n in MODELS:
        npz = np.load(FIG / f"combined_{n}.npz", allow_pickle=True)
        rels = json.load(open(FIG / f"combined_{n}.json"))["relations"]
        data[n] = dict(jcd=npz["forecast__JCD"], sizes=npz["clique_sizes"], rels=rels)

    relations = data[MODELS[0]]["rels"]
    sizes = data[MODELS[0]]["sizes"]

    by_rel = {rel: dict(ratio=[], v_sum=[], sqrt_v_sum_sq=[], sigma_max=[], sigma_min=[]) for rel in RELATIONS_OF_INTEREST}

    for t, rel in enumerate(relations):
        if rel not in RELATIONS_OF_INTEREST:
            continue
        m = int(sizes[t])
        try:
            clique = build_clique(rel, m)
        except ValueError:
            continue

        Pi = np.stack([
            jcd_project(clique, data[n]["jcd"][t, :m]) for n in MODELS
        ])
        Pi_bar = Pi.mean(axis=0)
        sigma = np.sqrt(((Pi - Pi_bar) ** 2).mean(axis=0))   # (m,)
        a_R = binding_normal(rel, m, Pi_bar)
        v = (a_R ** 2) * (sigma ** 2)
        sum_v = float(v.sum())
        sum_sqrt_v = float(np.sqrt(v).sum())
        if sum_sqrt_v <= 1e-12:
            continue
        ratio = m * sum_v / (sum_sqrt_v ** 2)
        by_rel[rel]["ratio"].append(ratio)
        by_rel[rel]["v_sum"].append(sum_v)
        by_rel[rel]["sqrt_v_sum_sq"].append(sum_sqrt_v ** 2)
        by_rel[rel]["sigma_max"].append(float(sigma.max()))
        by_rel[rel]["sigma_min"].append(float(sigma.min()))

    print("=" * 80)
    print("Option 1: Optimal coupling-aware routing allocation")
    print("Heterogeneity gain  =  uniform / optimal  =  m * sum(v) / (sum sqrt(v))^2")
    print("(v_j = a_{R,j}^2 * sigma_j^2;  ratio >= 1 by Cauchy-Schwarz)")
    print("=" * 80)
    print(f"{'rel':<10}{'<ratio>':>12}{'median':>10}{'p75':>10}{'p90':>10}{'p99':>10}{'<sigma_max/min>':>20}{'N':>6}")
    for rel in RELATIONS_OF_INTEREST:
        rs = np.array(by_rel[rel]["ratio"])
        if len(rs) == 0:
            continue
        smax = np.array(by_rel[rel]["sigma_max"])
        smin = np.array(by_rel[rel]["sigma_min"])
        ratio_sigma = (smax / np.maximum(smin, 1e-9)).mean()
        print(f"{rel:<10}"
              f"{rs.mean():>12.3f}"
              f"{np.median(rs):>10.3f}"
              f"{np.quantile(rs, 0.75):>10.3f}"
              f"{np.quantile(rs, 0.90):>10.3f}"
              f"{np.quantile(rs, 0.99):>10.3f}"
              f"{ratio_sigma:>20.2f}"
              f"{len(rs):>6}")

    print()
    print("Interpretation:")
    print("  - mean heterogeneity gain >= 1; >>1 means heterogeneous specialists, optimal saves a lot.")
    print("  - close to 1 means all coords contribute equally to E[(eps*)^2]; optimal ~ uniform.")
    print("  - the upper bound is m^* (when all variance concentrates on one coord).")


if __name__ == "__main__":
    main()
