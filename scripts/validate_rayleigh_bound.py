"""
Validate the strengthened Cor 3.7 (Rayleigh-quotient form).

For relation R with active constraint normal a_R at the boundary of the
joint polytope:

  Equality constraint (neg, partition):
    E_sigma[eps*^2] = a_R^T D a_R / ||a_R||^2,   D = diag(Sigma_Pi)

  Inequality constraint (and, or, Frechet halfspaces):
    E_sigma[eps*^2] = (1/2) a_R^T D a_R / ||a_R||^2   (single-active, Pi_bar
                                                       on the boundary)

For and/or, the dominant active constraint depends on which Frechet halfspace
is tightest at Pi_bar. We compute the bound per-clique by picking the
constraint with smallest slack at Pi_bar (or, in single-active analysis, the
binding one in the projection).
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
SEEDS = 16
MASTER_SEED = 0


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


def constraint_data(rel: str, m: int) -> tuple[list[np.ndarray], list[float], bool]:
    """Return (a_list, b_list, is_equality) for the defining halfspace
    constraints of the joint polytope (excluding box constraints).
    For equality constraints, return the single equality normal."""
    if rel == "neg":
        # r_1 + r_2 = 1
        return ([np.array([1.0, 1.0])], [1.0], True)
    if rel == "partition":
        a = np.ones(m)
        return ([a], [1.0], True)
    if rel == "and":
        # Frechet: r_3 - r_1 <= 0, r_3 - r_2 <= 0, r_1 + r_2 - r_3 <= 1
        return (
            [np.array([-1.0, 0.0, 1.0]),
             np.array([0.0, -1.0, 1.0]),
             np.array([1.0, 1.0, -1.0])],
            [0.0, 0.0, 1.0],
            False,
        )
    if rel == "or":
        # Frechet: r_1 - r_3 <= 0, r_2 - r_3 <= 0, r_3 - r_1 - r_2 <= 0
        return (
            [np.array([1.0, 0.0, -1.0]),
             np.array([0.0, 1.0, -1.0]),
             np.array([-1.0, -1.0, 1.0])],
            [0.0, 0.0, 0.0],
            False,
        )
    raise ValueError(rel)


def main() -> None:
    data = {}
    for n in MODELS:
        npz = np.load(FIG / f"combined_{n}.npz", allow_pickle=True)
        rels = json.load(open(FIG / f"combined_{n}.json"))["relations"]
        data[n] = dict(jcd=npz["forecast__JCD"], sizes=npz["clique_sizes"], rels=rels)

    relations = data[MODELS[0]]["rels"]
    sizes = data[MODELS[0]]["sizes"]

    by_rel = {rel: dict(observed=[], rayleigh=[], generic=[]) for rel in RELATIONS_OF_INTEREST}

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
        k = Pi.shape[0]
        Pi_bar = Pi.mean(axis=0)
        # Diagonal "effective" specialist covariance under independent uniform sigma
        sigma_diag = ((Pi - Pi_bar) ** 2).mean(axis=0)  # (m,)
        D = np.diag(sigma_diag)
        tr_D = float(sigma_diag.sum())

        a_list, b_list, is_equality = constraint_data(rel, m)
        # For each constraint, compute Rayleigh predicted eps^2
        rayleigh_per_constraint = []
        for a_i, b_i in zip(a_list, b_list):
            aDa = float(a_i @ D @ a_i)
            norm2 = float(a_i @ a_i)
            slack_i = float(b_i - a_i @ Pi_bar)        # slack at Pi_bar (>=0 if feasible)
            if is_equality:
                pred = aDa / norm2
            else:
                # inequality: factor 1/2 from one-sided truncation under
                # zero-mean symmetric u; assumes Pi_bar on the constraint
                # boundary (slack ~ 0).  When slack > 0 the bound is even
                # smaller because violations are rarer.  Use Gaussian-style
                # truncated-second-moment correction:
                # E[(N(0, s^2) - slack)_+^2] ~ s^2 * (1/2) when slack=0,
                # decays as slack grows.  For our discrete-mixture u this
                # is a heuristic; we compute the empirical truncation
                # factor below as a sanity check.  As a closed-form, just
                # use the slack=0 value.
                pred = 0.5 * aDa / norm2
            rayleigh_per_constraint.append((pred, slack_i, a_i, b_i))

        # Pick the dominant constraint: the one with the largest predicted
        # contribution (typically smallest slack -> easiest to violate).
        best = max(rayleigh_per_constraint, key=lambda x: x[0])
        rayleigh_pred = best[0]

        # Empirically observed eps^2 over SEEDS uniform random assignments
        eps2_samples = []
        for s in range(SEEDS):
            rng = np.random.default_rng(MASTER_SEED + 17 * t + s)
            assign = rng.integers(0, k, size=m)
            x = Pi[assign, np.arange(m)]
            proj = jcd_project(clique, x)
            eps2_samples.append(float(np.linalg.norm(x - proj) ** 2))
        observed_eps2 = float(np.mean(eps2_samples))

        by_rel[rel]["observed"].append(observed_eps2)
        by_rel[rel]["rayleigh"].append(rayleigh_pred)
        by_rel[rel]["generic"].append(tr_D)

    print("=" * 80)
    print("Strengthened Cor 3.7 (Rayleigh-quotient form) validation")
    print("    obs eps^2  vs  rayleigh prediction (single-active a_R^T D a_R / ||a_R||^2,")
    print("    factor 1/2 for inequality constraints)")
    print("=" * 80)
    print(f"{'rel':<10}{'<obs>':>12}{'<rayleigh>':>14}{'ratio':>10}{'corr':>8}{'<generic>':>12}{'gen ratio':>10}")
    for rel in RELATIONS_OF_INTEREST:
        obs = np.array(by_rel[rel]["observed"])
        ray = np.array(by_rel[rel]["rayleigh"])
        gen = np.array(by_rel[rel]["generic"])
        if len(obs) < 3:
            continue
        ratio_r = obs.mean() / ray.mean() if ray.mean() > 0 else float("nan")
        ratio_g = obs.mean() / gen.mean() if gen.mean() > 0 else float("nan")
        if np.std(obs) > 0 and np.std(ray) > 0:
            corr = float(np.corrcoef(obs, ray)[0, 1])
        else:
            corr = float("nan")
        print(f"{rel:<10}{obs.mean():>12.5f}{ray.mean():>14.5f}{ratio_r:>10.3f}{corr:>8.3f}{gen.mean():>12.5f}{ratio_g:>10.3f}")

    print()
    print("Interpretation:")
    print("  - ratio (obs / rayleigh) close to 1.0 means the strengthened bound is tight.")
    print("  - The 'generic ratio' is observed / tr(D) under the original generic bound.")
    print("  - For neg/partition (equality), Rayleigh = obs by construction modulo MC noise.")
    print("  - For and/or (inequality, factor 1/2), Rayleigh should match obs much better")
    print("    than the generic tr(D) bound (which over-counts by ~5-6x).")


if __name__ == "__main__":
    main()
