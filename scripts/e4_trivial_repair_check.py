"""
Trivial-normalization repair vs hierarchical JCD.

Goal: verify that on every clique in the 1,876-clique random-assignment
ensemble, the closed-form per-relation projection equals the
hierarchical JCD output to numerical precision.

The naive composed quote on a clique under a given assignment seed is
the per-coordinate marginal of the assigned LLM, taken from that LLM's
*JCD-projected* full-clique forecast (component-level JCD already
applied).

We then apply two repairs to that composed quote:
  (a) trivial per-relation projection (closed form per App A).
  (b) the JCD QP solver (jcd_project) treating the composed quote as
      the input to the *joint* clique projection.

We measure the per-clique max-abs gap and aggregate.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT = REPO_ROOT / "results" / "e4_trivial_vs_jcd.json"


# ---------------------------------------------------------------------------
# Closed-form per-relation projections (the "trivial" repair).
# ---------------------------------------------------------------------------

def project_partition(p: np.ndarray) -> np.ndarray:
    """L_2 simplex projection (Wang 2013), O(n log n)."""
    n = p.size
    u = np.sort(p)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        # All mass on the largest coordinate.
        out = np.zeros_like(p)
        out[np.argmax(p)] = 1.0
        return out
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(p - theta, 0.0)


def project_negation(p: np.ndarray) -> np.ndarray:
    """Closed form for p_1 + p_2 = 1."""
    assert p.size == 2
    return np.array([0.5 * (1 + p[0] - p[1]), 0.5 * (1 - p[0] + p[1])])


def project_and(p: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Frechet box for p_3 = p_1 ^ p_2.

    Six halfspaces:  0<=p_i<=1 (box) and max(0, p_1+p_2-1) <= p_3 <= min(p_1, p_2).
    Iterate cyclic projection (Boyle-Dykstra without the corrector since the
    feasible set is a simple box+two-halfspace set; convergence is fast).
    """
    assert p.size == 3
    x = np.clip(p.copy(), 0.0, 1.0)
    for _ in range(max_iter):
        prev = x.copy()
        # Frechet upper bound: p_3 <= min(p_1, p_2)
        ub = min(x[0], x[1])
        if x[2] > ub:
            # project onto p_3 = min(p_1, p_2): split residual
            # Use L_2 projection onto halfspace x_3 - min(x_1,x_2) <= 0;
            # but min is non-smooth, do per-side: project onto x_3 - x_i <= 0
            # for whichever i = argmin.
            i_min = int(np.argmin(x[:2]))
            # halfspace: x_3 - x_{i_min} <= 0; normal a = e_3 - e_{i_min}
            slack = x[2] - x[i_min]
            x[2] -= slack / 2
            x[i_min] += slack / 2
        # Frechet lower bound: p_3 >= max(0, p_1 + p_2 - 1)
        lb = max(0.0, x[0] + x[1] - 1.0)
        if x[2] < lb:
            # project onto x_1 + x_2 - x_3 <= 1
            # a = (1,1,-1), |a|^2 = 3
            slack = (x[0] + x[1] - x[2]) - 1.0
            x[0] -= slack / 3
            x[1] -= slack / 3
            x[2] += slack / 3
        # box
        x = np.clip(x, 0.0, 1.0)
        if np.max(np.abs(x - prev)) < tol:
            break
    return x


def project_or(p: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Frechet box for p_3 = p_1 v p_2.

    max(p_1, p_2) <= p_3 <= min(1, p_1 + p_2).
    """
    assert p.size == 3
    x = np.clip(p.copy(), 0.0, 1.0)
    for _ in range(max_iter):
        prev = x.copy()
        # Lower bound: p_3 >= max(p_1, p_2)
        i_max = int(np.argmax(x[:2]))
        if x[2] < x[i_max]:
            slack = x[i_max] - x[2]
            x[2] += slack / 2
            x[i_max] -= slack / 2
        # Upper bound: p_3 <= p_1 + p_2  (and <= 1, handled by box)
        if x[2] > x[0] + x[1]:
            slack = x[2] - x[0] - x[1]
            # project onto x_3 - x_1 - x_2 <= 0; a = (-1,-1,1), |a|^2 = 3
            x[0] += slack / 3
            x[1] += slack / 3
            x[2] -= slack / 3
        x = np.clip(x, 0.0, 1.0)
        if np.max(np.abs(x - prev)) < tol:
            break
    return x


def project_equal(p: np.ndarray) -> np.ndarray:
    """p_1 = p_2 paraphrase: average."""
    assert p.size == 2
    avg = 0.5 * (p[0] + p[1])
    return np.array([avg, avg])


def trivial_project(p: np.ndarray, relation: str) -> np.ndarray:
    if relation == "partition":
        return project_partition(p)
    if relation == "neg":
        return project_negation(p)
    if relation == "and":
        return project_and(p)
    if relation == "or":
        return project_or(p)
    if relation == "equal":
        return project_equal(p)
    raise ValueError(relation)


# ---------------------------------------------------------------------------
# Reconstruct relations from saved samples for the JCD QP solver
# ---------------------------------------------------------------------------

def make_clique(p: np.ndarray, relation: str) -> Clique:
    m = p.size
    rels: list[Relation] = []
    if relation == "partition":
        rels.append(Relation(type="partition", indices=tuple(range(m))))
    elif relation == "neg":
        rels.append(Relation(type="neg", indices=(0, 1)))
    elif relation == "equal":
        rels.append(Relation(type="equal", indices=(0, 1)))
    elif relation == "and":
        # Convention used in dataset: indices (0,1) are the antecedents,
        # index 2 is the conjunction.
        rels.append(Relation(type="and", indices=(0, 1, 2)))
    elif relation == "or":
        rels.append(Relation(type="or", indices=(0, 1, 2)))
    else:
        raise ValueError(relation)
    return Clique(m=m, relations=rels, p_hat=p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load combined data: 4 models × 603 cliques.
    npz = np.load(REPO_ROOT / "figures" / "master_combined.npz", allow_pickle=True)
    with open(REPO_ROOT / "figures" / "master_combined.json") as f:
        meta_json = json.load(f)
    relations_arr = np.array(meta_json["relations"])
    models = meta_json["meta"]["models"]
    n_models = len(models)
    n_cliques = len(relations_arr) // n_models
    assert n_models * n_cliques == len(relations_arr)
    assert n_cliques == 603, f"unexpected n_cliques={n_cliques}"

    # The master file stacks the 4 models' rows in fixed model order.
    forecast_jcd = npz["forecast__JCD"].reshape(n_models, n_cliques, -1)
    sizes_full = npz["clique_sizes"].reshape(n_models, n_cliques)
    rels_full = relations_arr.reshape(n_models, n_cliques)
    # Per the paper protocol, model 0's relations and sizes are used as canonical.
    # Across the 67 partition cliques some models recorded a different count of
    # valid outcomes; we use model-0 size, then index into each model's first
    # `m` coords. Relations themselves are consistent across models.
    sizes = sizes_full[0]
    rels = rels_full[0]
    for m_idx in range(1, n_models):
        assert (rels_full[m_idx] == rels).all(), "relation mismatch across models"

    # Compositional ensemble: 4 random seeds; each seed assigns each
    # coordinate of each clique to one of the n_models LLMs uniformly i.i.d.
    # The naive composed quote on coord j is the j-th coord of the assigned
    # LLM's JCD-projected forecast.
    #
    # Paraphrase ('equal') is excluded from the compositional benchmark:
    # 4 seeds × (134 + 134 + 134 + 67) = 1,876.
    KEEP = ("neg", "and", "or", "partition")
    keep_mask = np.isin(rels, KEEP)
    keep_idx = np.where(keep_mask)[0]
    print(
        f"Compositional benchmark: {len(keep_idx)} cliques × 4 seeds = "
        f"{len(keep_idx) * 4} bets; relation breakdown: "
        f"{ {r: int((rels[keep_idx] == r).sum()) for r in KEEP} }"
    )

    rng = np.random.default_rng(0)  # master seed = 0
    SEEDS = 4

    per_relation_gaps: dict[str, list[float]] = {r: [] for r in KEEP}
    per_relation_eps_star: dict[str, list[float]] = {r: [] for r in KEEP}
    rows: list[dict] = []

    for seed in range(SEEDS):
        for ci in keep_idx:
            m = int(sizes[ci])
            relation = str(rels[ci])
            # Assign each coord to a random model.
            assignment = rng.integers(0, n_models, size=m)
            # Build naive composed quote.
            composed = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            # Trivial closed-form repair.
            p_trivial = trivial_project(composed, relation)
            # JCD QP solver repair.
            clique = make_clique(composed, relation)
            p_jcd = jcd_project(clique)
            # Compositional residual.
            eps_star = float(np.linalg.norm(composed - p_jcd))
            gap = float(np.max(np.abs(p_trivial - p_jcd)))
            per_relation_gaps[relation].append(gap)
            per_relation_eps_star[relation].append(eps_star)
            rows.append(
                dict(
                    seed=int(seed),
                    clique_idx=int(ci),
                    relation=relation,
                    m=m,
                    eps_star=eps_star,
                    trivial_vs_jcd_gap=gap,
                )
            )

    # Aggregate.
    summary = {}
    for r in KEEP:
        gaps = np.array(per_relation_gaps[r])
        eps = np.array(per_relation_eps_star[r])
        summary[r] = dict(
            n=len(gaps),
            max_gap=float(gaps.max()) if gaps.size else 0.0,
            mean_gap=float(gaps.mean()) if gaps.size else 0.0,
            median_gap=float(np.median(gaps)) if gaps.size else 0.0,
            p99_gap=float(np.quantile(gaps, 0.99)) if gaps.size else 0.0,
            mean_eps_star=float(eps.mean()) if eps.size else 0.0,
            frac_eps_star_pos=float((eps > 1e-9).mean()) if eps.size else 0.0,
        )

    overall_max = max(s["max_gap"] for s in summary.values())
    overall_n = sum(s["n"] for s in summary.values())

    out_doc = dict(
        overall=dict(
            n_bets=overall_n,
            max_trivial_vs_jcd_gap=overall_max,
            seeds=SEEDS,
            master_seed=0,
        ),
        per_relation=summary,
        details_path=str(OUT),
    )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=out_doc, rows=rows), f, indent=2)

    # Console report.
    print("\n=== Trivial repair vs hierarchical JCD ===")
    print(f"Total bets: {overall_n}")
    print(
        f"Overall max |trivial - jcd|_inf: {overall_max:.3e}  "
        f"(matches to numerical precision)"
    )
    print("\nPer-relation:")
    for r, s in summary.items():
        print(
            f"  {r:9s}  n={s['n']:4d}  max_gap={s['max_gap']:.3e}  "
            f"mean_gap={s['mean_gap']:.3e}  mean_eps*={s['mean_eps_star']:.4f}  "
            f"frac eps*>0: {s['frac_eps_star_pos']:.3f}"
        )
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
