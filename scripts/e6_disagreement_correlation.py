"""
Disagreement-residual correlation.

Proposition 3.5 in the paper gives an inequality:
    eps_star(p) <= || A(Pi_1, ..., Pi_k) - r ||_2  for any r in M*
and uses the JCD-projected forecast of any single component as the
reference r. The paper reports mean bound-to-residual ratios per
relation but does not report whether disagreement *predicts* eps_star
across cliques.

This script measures Spearman correlation between inter-LLM
disagreement and the empirical eps_star, per relation, on the
1,876-bet ensemble.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT = REPO_ROOT / "results" / "e6_disagreement_correlation.json"
SEED = 0
SEEDS = 4
N_MODELS = 4
KEEP = ("neg", "and", "or", "partition")


def make_clique(p: np.ndarray, relation: str) -> Clique:
    m = p.size
    if relation == "partition":
        rels = [Relation(type="partition", indices=tuple(range(m)))]
    elif relation == "neg":
        rels = [Relation(type="neg", indices=(0, 1))]
    elif relation == "and":
        rels = [Relation(type="and", indices=(0, 1, 2))]
    elif relation == "or":
        rels = [Relation(type="or", indices=(0, 1, 2))]
    else:
        raise ValueError(relation)
    return Clique(m=m, relations=rels, p_hat=p)


def main() -> None:
    npz = np.load(REPO_ROOT / "figures" / "master_combined.npz", allow_pickle=True)
    with open(REPO_ROOT / "figures" / "master_combined.json") as f:
        meta_json = json.load(f)
    relations_arr = np.array(meta_json["relations"])
    n_total = len(relations_arr)
    n_cliques = n_total // N_MODELS
    forecast_jcd = npz["forecast__JCD"].reshape(N_MODELS, n_cliques, -1)
    sizes_full = npz["clique_sizes"].reshape(N_MODELS, n_cliques)
    rels_full = relations_arr.reshape(N_MODELS, n_cliques)
    rels = rels_full[0]
    sizes = sizes_full[0]

    rng = np.random.default_rng(SEED)
    keep_idx = np.where(np.isin(rels, KEEP))[0]

    by_rel = {r: dict(disagree=[], eps=[], bound=[]) for r in KEEP}

    for seed in range(SEEDS):
        for ci in keep_idx:
            relation = str(rels[ci])
            m = int(sizes[ci])
            assignment = rng.integers(0, N_MODELS, size=m)
            if relation == "partition":
                if any(int(sizes_full[a, ci]) < m for a in assignment):
                    continue

            # Composed quote x.
            x = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            clique = make_clique(x, relation)
            proj = jcd_project(clique)
            eps_star = float(np.linalg.norm(x - proj))

            # Disagreement upper bound: min over components beta of ||x - r_beta||
            # where r_beta = beta's full-clique JCD-projected forecast (already
            # in M_C; we re-project to be safe against OSQP slack as in
            # verify_disagreement_bound.py).
            cands = []
            for b in range(N_MODELS):
                if relation == "partition" and int(sizes_full[b, ci]) < m:
                    continue
                r_b = forecast_jcd[b, ci, :m]
                # Re-tighten via jcd_project to remove any numerical slack.
                r_b_clean = jcd_project(make_clique(r_b, relation))
                cands.append(float(np.linalg.norm(x - r_b_clean)))
            if not cands:
                continue
            bound = min(cands)

            # Model-pair disagreement: max_{a,b} ||r_a - r_b||_2.
            pair_disagree = 0.0
            for a in range(N_MODELS):
                if relation == "partition" and int(sizes_full[a, ci]) < m:
                    continue
                for b in range(a + 1, N_MODELS):
                    if relation == "partition" and int(sizes_full[b, ci]) < m:
                        continue
                    d = float(np.linalg.norm(
                        forecast_jcd[a, ci, :m] - forecast_jcd[b, ci, :m]
                    ))
                    pair_disagree = max(pair_disagree, d)

            by_rel[relation]["eps"].append(eps_star)
            by_rel[relation]["bound"].append(bound)
            by_rel[relation]["disagree"].append(pair_disagree)

    # Aggregate.
    summary = {}
    for r in KEEP:
        eps = np.array(by_rel[r]["eps"])
        bnd = np.array(by_rel[r]["bound"])
        dis = np.array(by_rel[r]["disagree"])
        if eps.size == 0:
            summary[r] = dict(n=0)
            continue
        # Restrict to positive-eps cliques (where the relationship is meaningful).
        pos = eps > 1e-9
        n_pos = int(pos.sum())
        rho_b, p_b = stats.spearmanr(bnd[pos], eps[pos]) if n_pos >= 3 else (np.nan, np.nan)
        rho_d, p_d = stats.spearmanr(dis[pos], eps[pos]) if n_pos >= 3 else (np.nan, np.nan)
        # Pearson R^2 of eps on disagreement (linear).
        if n_pos >= 3:
            slope, intercept, r_pearson, _, _ = stats.linregress(dis[pos], eps[pos])
            r2 = float(r_pearson ** 2)
        else:
            r2 = float("nan")
        summary[r] = dict(
            n=int(eps.size),
            n_pos=n_pos,
            spearman_bound_eps=float(rho_b),
            p_spearman_bound=float(p_b),
            spearman_pairmax_eps=float(rho_d),
            p_spearman_pairmax=float(p_d),
            pearson_R2_pairmax=r2,
            mean_bound=float(bnd.mean()),
            mean_eps=float(eps.mean()),
            mean_pairmax_disagree=float(dis.mean()),
        )

    out_doc = dict(
        meta=dict(seed=SEED, seeds=SEEDS, n_models=N_MODELS),
        per_relation=summary,
    )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_doc, f, indent=2)

    # Console.
    print("\n=== Disagreement -> residual correlation ===")
    print(
        f"{'relation':<10}{'n':>5}{'n_pos':>7}"
        f"{'rho(bound,eps)':>17}{'rho(pair,eps)':>16}{'R^2(pair,eps)':>16}"
    )
    for r in KEEP:
        s = summary[r]
        if s.get("n", 0) == 0:
            continue
        print(
            f"  {r:<10s}{s['n']:>3d}{s['n_pos']:>6d}"
            f"{s['spearman_bound_eps']:>17.3f}{s['spearman_pairmax_eps']:>16.3f}"
            f"{s['pearson_R2_pairmax']:>16.3f}"
        )
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
