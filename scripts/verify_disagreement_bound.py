"""
Verify Proposition 3.6 (disagreement upper bound on epsilon^star).

For each ensemble clique, compute:
  eps_star  = ||x - Pi*(x)||_2 where x is the composed quote
  bound     = min_beta ||x - Pi_beta(p_hat^(beta))||_2

By Prop 3.6, bound >= eps_star deterministically. We report the
ratio bound / eps_star aggregated by relation type.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from jcd.types import Clique, Relation
from jcd.qp.solver import project as jcd_project

K = 8
SEED = 0
RELATIONS_OF_INTEREST = ("neg", "and", "or", "partition")
MODELS = [
    "anthropic_claude-haiku-4-5-20251001",
    "azure_mini",
    "azure_nano",
    "groq_llama-3",
]


def build_clique(rel: str, m: int) -> Clique:
    if rel == "neg" and m == 2:
        rels = [Relation(type="neg", indices=(0, 1))]
    elif rel == "and" and m == 3:
        rels = [Relation(type="and", indices=(0, 1, 2))]
    elif rel == "or" and m == 3:
        rels = [Relation(type="or", indices=(0, 1, 2))]
    elif rel == "partition":
        rels = [Relation(type="partition", indices=tuple(range(m)))]
    else:
        raise ValueError(f"unsupported {rel} m={m}")
    return Clique(m=m, relations=rels)


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = {}
    for n in MODELS:
        npz = np.load(ROOT.parent / f"figures/combined_{n}.npz", allow_pickle=True)
        with open(ROOT.parent / f"figures/combined_{n}.json") as f:
            j = json.load(f)
        data[n] = dict(jcd=npz["forecast__JCD"], sizes=npz["clique_sizes"], rels=j["relations"])

    relations = data[MODELS[0]]["rels"]
    sizes = data[MODELS[0]]["sizes"]
    jcd_stack = np.stack([data[n]["jcd"] for n in MODELS], axis=0)

    by_rel = {r: dict(eps=[], bound=[]) for r in RELATIONS_OF_INTEREST}

    SHUFFLES = 4
    for s in range(SHUFFLES):
        for t, rel in enumerate(relations):
            if rel not in RELATIONS_OF_INTEREST:
                continue
            m = int(sizes[t])
            try:
                clique = build_clique(rel, m)
            except ValueError:
                continue
            jcd_full = jcd_stack[:, t, :m]  # (M, m)
            # Re-project each stored JCD output onto M_C to remove OSQP
            # slack (the stored values were OSQP-feasible but not exactly
            # in M_C; we need exact feasibility to use them as references).
            jcd_full_clean = np.stack([jcd_project(clique, jcd_full[b]) for b in range(len(MODELS))])
            assign = rng.integers(0, len(MODELS), size=m)
            x = jcd_full_clean[assign, np.arange(m)]
            proj = jcd_project(clique, x)
            eps = float(np.linalg.norm(x - proj))
            bound = float(min(np.linalg.norm(x - jcd_full_clean[b]) for b in range(len(MODELS))))
            by_rel[rel]["eps"].append(eps)
            by_rel[rel]["bound"].append(bound)

    print(f"{'relation':<12}{'N':>6}{'<eps>':>10}{'<bound>':>10}{'ratio':>10}{'min ratio':>12}{'frac bound>=eps':>18}")
    for rel in RELATIONS_OF_INTEREST:
        eps = np.array(by_rel[rel]["eps"])
        bound = np.array(by_rel[rel]["bound"])
        if not len(eps):
            continue
        # restrict to cliques with positive eps (where the bound is meaningful)
        mask = eps > 1e-6
        ratio = (bound[mask] / np.maximum(eps[mask], 1e-12)).mean() if mask.any() else float('nan')
        minr = (bound[mask] / np.maximum(eps[mask], 1e-12)).min() if mask.any() else float('nan')
        valid = float(np.mean(bound + 1e-9 >= eps))
        print(f"{rel:<12}{int(mask.sum()):>6d}{eps[mask].mean():>10.4f}{bound[mask].mean():>10.4f}"
              f"{ratio:>10.3f}{minr:>12.3f}{valid:>18.4f}")


if __name__ == "__main__":
    main()
