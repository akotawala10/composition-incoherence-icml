"""
Validate Alternative E: quantitative magnitude bound on E[(eps*)^2].

Claim: under owner-selection with i.i.d. uniform random assignment of each
joint coordinate to one of k specialists whose JCD outputs all lie in
M*, the expectation of (eps*)^2 over the random assignment satisfies

    E_sigma[(eps*)^2]   =   c_R * tr(Sigma_Pi)              (negation, partition)
    E_sigma[(eps*)^2]  <=   c_R * tr(Sigma_Pi)              (conj, disj, generic)

where:
    Sigma_Pi  =  (1/k) sum_a (Pi_a - Pi_bar) (Pi_a - Pi_bar)^T  is the empirical
                 covariance of the k specialists' projected forecasts;
    Pi_bar    =  (1/k) sum_a Pi_a (lies in M* by convexity);
    c_neg = 1/2,  c_partition = 1/m*,  c_and = c_or = 1.

We compare per-clique observed mean (eps*)^2 over 4 seeds versus the
theoretical c_R * tr(Sigma_Pi).
"""

from __future__ import annotations

import os
import json
import sys
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
SEEDS = 16  # more seeds for tighter expectation estimate
MASTER_SEED = 0


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
    data = {}
    for n in MODELS:
        npz = np.load(FIG / f"combined_{n}.npz", allow_pickle=True)
        rels = json.load(open(FIG / f"combined_{n}.json"))["relations"]
        data[n] = dict(jcd=npz["forecast__JCD"], sizes=npz["clique_sizes"], rels=rels)

    relations = data[MODELS[0]]["rels"]
    sizes = data[MODELS[0]]["sizes"]
    rng_master = np.random.default_rng(MASTER_SEED)

    by_rel = {rel: dict(observed=[], predicted=[], tr=[]) for rel in RELATIONS_OF_INTEREST}

    for t, rel in enumerate(relations):
        if rel not in RELATIONS_OF_INTEREST:
            continue
        m = int(sizes[t])
        try:
            clique = build_clique(rel, m)
        except ValueError:
            continue

        # Each specialist's JCD output, re-projected to canonical feasibility.
        Pi = np.stack([
            jcd_project(clique, data[n]["jcd"][t, :m]) for n in MODELS
        ])  # (k, m)
        k = Pi.shape[0]
        Pi_bar = Pi.mean(axis=0)            # (m,) in M* by convexity
        # tr(Sigma_Pi) = (1/k) sum_{a, j} (Pi_a,j - Pi_bar,j)^2
        tr_sigma = float(np.mean(np.sum((Pi - Pi_bar) ** 2, axis=1)))

        c_R = {"neg": 0.5, "partition": 1.0 / m, "and": 1.0, "or": 1.0}[rel]
        predicted_eps2 = c_R * tr_sigma

        # Empirical: average (eps*)^2 over SEEDS uniform random assignments.
        eps2_samples = []
        for s in range(SEEDS):
            rng = np.random.default_rng(MASTER_SEED + 17 * t + s)
            assign = rng.integers(0, k, size=m)
            x = Pi[assign, np.arange(m)]
            x_proj = jcd_project(clique, x)
            eps2_samples.append(float(np.linalg.norm(x - x_proj) ** 2))
        observed_eps2 = float(np.mean(eps2_samples))

        by_rel[rel]["observed"].append(observed_eps2)
        by_rel[rel]["predicted"].append(predicted_eps2)
        by_rel[rel]["tr"].append(tr_sigma)

    print("=" * 76)
    print("Alternative-E magnitude-bound validation")
    print(f"  E_sigma[(eps*)^2]  ~  c_R * tr(Sigma_Pi),  {SEEDS} uniform random seeds/clique")
    print("=" * 76)
    print(f"{'rel':<10}{'<obs eps^2>':>14}{'<pred eps^2>':>14}{'ratio obs/pred':>18}{'corr':>10}{'N':>6}")
    for rel in RELATIONS_OF_INTEREST:
        obs = np.array(by_rel[rel]["observed"])
        pred = np.array(by_rel[rel]["predicted"])
        if len(obs) < 3:
            continue
        m_obs = obs.mean()
        m_pred = pred.mean()
        ratio = m_obs / m_pred if m_pred > 0 else float("nan")
        # Pearson correlation between per-clique observed and predicted
        if np.std(obs) > 0 and np.std(pred) > 0:
            corr = float(np.corrcoef(obs, pred)[0, 1])
        else:
            corr = float("nan")
        print(f"{rel:<10}{m_obs:>14.5f}{m_pred:>14.5f}{ratio:>18.4f}{corr:>10.4f}{len(obs):>6}")

    print()
    print("Interpretation:")
    print("  - For negation and partition the bound is an EQUALITY in expectation;")
    print("    ratio should be ~1.0 and corr should be high.")
    print("  - For conjunction/disjunction the bound is an UPPER bound (1-Lipschitz")
    print("    contraction of L2 projection onto the Frechet box); ratio < 1.")
    print()
    print("If the negation ratio deviates from 1, it reflects finite-seed Monte-Carlo error")
    print("and the small bias from the joint feasibility constraint p in [0,1]^m.")


if __name__ == "__main__":
    main()
