"""
Multi-allocation regret on the random-assignment ensemble.

Recomputes regret under three allocation rules to verify that
the JCD advantage is not specific to a single rule:

  - proportional : w_i = max(p_i, 0) / sum_j max(p_j, 0)
  - kelly_clip   : w_i = clip(p_i, eps, 1) -- no re-normalisation;
                   then re-cast as portfolio weights via division by
                   their sum FOR THE BETTOR. Implemented two ways:
                     (a) hard truncated Kelly: bet only the
                         coordinate-weighted excess of p over its
                         baseline, clipped >= 0.
                     (b) post-clip Kelly: w_i = p_i / sum_j p_j when
                         the agent's quote already sums to >= 1; else
                         pad uniform.
  - max_entropy  : w = arg max H(w) s.t. sum w = 1 and w_i >= p_i,
                   i.e. respect floors but spread uncertainty.

For each rule, we compute realised log-payoff log(w_winner) on the
1,250 ensemble bets where exactly one outcome resolved YES, and
compare regimes (naive composition vs hierarchical JCD vs single-LLM
oracle).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT = REPO_ROOT / "results" / "b2_multi_allocation_regret.json"
SEED = 0
SEEDS = 4
N_MODELS = 4
KEEP = ("neg", "and", "or", "partition")
EPS = 1e-9


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


def alloc_proportional(p: np.ndarray) -> np.ndarray:
    p_pos = np.maximum(p, 0.0)
    s = p_pos.sum()
    if s < EPS:
        return np.full_like(p_pos, 1.0 / p_pos.size)
    return p_pos / s


def alloc_truncated_kelly(p: np.ndarray) -> np.ndarray:
    """Truncated Kelly without re-normalisation: clip to [eps, 1],
    leave un-renormalised mass as 'cash' (modelled as uniform spread).
    Concretely, when sum < 1 we top up uniformly; when sum > 1 we
    truncate proportionally to keep total = 1 (same as proportional in
    that case). The asymmetry penalises naive over-allocation more
    than under-allocation."""
    p_pos = np.maximum(p, 0.0)
    s = p_pos.sum()
    if s < EPS:
        return np.full_like(p_pos, 1.0 / p_pos.size)
    if s <= 1.0:
        # Top up the deficit uniformly; this is closer to a Kelly
        # bettor's behaviour when offered a lossy quote.
        deficit = 1.0 - s
        return p_pos + deficit / p_pos.size
    # s > 1: truncate proportionally (equivalent to proportional, but
    # the mass that "doesn't fit" is lost relative to the agent's quote).
    return p_pos / s


def alloc_max_entropy(p: np.ndarray) -> np.ndarray:
    """Max-entropy allocation respecting per-coord floors p_i:
       w = arg max H(w) s.t. w_i >= max(p_i, 0), sum = 1, w_i >= 0.
    Closed form when sum(max(p,0)) <= 1: w = floor + (1 - sum(floor))/m
    distributed uniformly. When sum > 1, no feasible w exists and we
    fall back to proportional. This rule penalises over-confidence
    more than the others."""
    p_pos = np.maximum(p, 0.0)
    s = p_pos.sum()
    if s > 1.0 + EPS:
        return alloc_proportional(p)  # infeasible floors
    slack = 1.0 - s
    return p_pos + slack / p_pos.size


def log_payoff(w: np.ndarray, winner: int) -> float:
    return float(np.log(max(w[winner], EPS)))


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05):
    rng = np.random.default_rng(123)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


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
    res_full = npz["resolutions"].reshape(N_MODELS, n_cliques, -1)
    rels = rels_full[0]
    sizes = sizes_full[0]
    rng = np.random.default_rng(SEED)
    keep_idx = np.where(np.isin(rels, KEEP))[0]

    rules = {
        "proportional": alloc_proportional,
        "truncated_kelly": alloc_truncated_kelly,
        "max_entropy": alloc_max_entropy,
    }

    by_rule: dict[str, dict[str, list[float]]] = {
        rule: {"naive": [], "jcd": [], "oracle": []}
        for rule in rules
    }

    for seed in range(SEEDS):
        for ci in keep_idx:
            relation = str(rels[ci])
            m = int(sizes[ci])
            assignment = rng.integers(0, N_MODELS, size=m)
            if relation == "partition":
                if any(int(sizes_full[a, ci]) < m for a in assignment):
                    continue
            res = res_full[0, ci, :m]
            if not np.any(res > 0.5):
                continue
            winner = int(np.argmax(res))

            p_naive = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            clique = make_clique(p_naive, relation)
            p_jcd = jcd_project(clique)
            oracle_model = int(rng.integers(0, N_MODELS))
            if relation == "partition" and int(sizes_full[oracle_model, ci]) < m:
                cands = [a for a in range(N_MODELS) if int(sizes_full[a, ci]) >= m]
                if not cands:
                    continue
                oracle_model = int(rng.choice(cands))
            p_oracle = forecast_jcd[oracle_model, ci, :m]

            for rule, alloc in rules.items():
                for label, p in (("naive", p_naive), ("jcd", p_jcd), ("oracle", p_oracle)):
                    w = alloc(p)
                    by_rule[rule][label].append(log_payoff(w, winner))

    # Aggregate.
    summary = {}
    for rule in rules:
        d = {k: np.array(v) for k, v in by_rule[rule].items()}
        if d["naive"].size == 0:
            summary[rule] = dict(n=0)
            continue
        jcd_minus_naive = d["jcd"] - d["naive"]
        oracle_minus_naive = d["oracle"] - d["naive"]
        summary[rule] = dict(
            n=int(d["naive"].size),
            mean_lp_naive=float(d["naive"].mean()),
            mean_lp_jcd=float(d["jcd"].mean()),
            mean_lp_oracle=float(d["oracle"].mean()),
            jcd_minus_naive_mean=float(jcd_minus_naive.mean()),
            jcd_minus_naive_ci=list(paired_bootstrap_ci(jcd_minus_naive)),
            oracle_minus_naive_mean=float(oracle_minus_naive.mean()),
            oracle_minus_naive_ci=list(paired_bootstrap_ci(oracle_minus_naive)),
        )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Multi-allocation regret on the 1,250-bet ensemble ===")
    print(f"\n{'rule':<18}{'n':>6}{'lp(naive)':>12}{'lp(jcd)':>12}{'lp(oracle)':>12}"
          f"{'jcd-naive':>12}  95% CI                  oracle-naive  95% CI")
    for rule in rules:
        s = summary[rule]
        if s.get("n", 0) == 0:
            continue
        ci_jn = s["jcd_minus_naive_ci"]
        ci_on = s["oracle_minus_naive_ci"]
        print(
            f"  {rule:<16s}{s['n']:>6d}"
            f"{s['mean_lp_naive']:>12.4f}{s['mean_lp_jcd']:>12.4f}"
            f"{s['mean_lp_oracle']:>12.4f}"
            f"{s['jcd_minus_naive_mean']:>+12.4f}  [{ci_jn[0]:+.4f}, {ci_jn[1]:+.4f}]  "
            f"{s['oracle_minus_naive_mean']:>+10.4f}  [{ci_on[0]:+.4f}, {ci_on[1]:+.4f}]"
        )
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
