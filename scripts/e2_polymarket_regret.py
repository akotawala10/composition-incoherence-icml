"""
Downstream-decision regret on resolved cliques.

For each ensemble bet (random-assignment seeds × cliques), compute
realized log-payoff and Brier under three quote regimes:
  - naive: the per-coord assigned-LLM JCD-projected marginal (composed
           but not jointly projected; can violate cross-coord coupling).
  - jcd:   the same composed quote, jointly projected onto the clique
           polytope (hierarchical JCD).
  - oracle: a single LLM (drawn uniformly per bet) answers all
           coordinates with its own JCD-projected forecast (already
           coherent on the clique).

Allocation rule: w_i = max(p_i, 0) / sum_j max(p_j, 0). Payoff:
log(w_winner) where winner is the resolved coordinate. We use the
master_combined.npz resolution data (which is per-(model, clique));
since the underlying clique is the same across models, the winner
index agrees on cliques where all models record the same size; for
partition cliques where sizes disagree we skip the bet to avoid
mis-alignment.

Reports per relation and overall: mean log-payoff, mean Brier, and
paired-bootstrap CIs on (oracle - naive) and (jcd - naive).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT = REPO_ROOT / "results" / "e2_polymarket_regret.json"
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


def proportional_allocation(p: np.ndarray) -> np.ndarray:
    p_pos = np.maximum(p, 0.0)
    s = p_pos.sum()
    if s < EPS:
        # Degenerate quote: uniform allocation.
        return np.full_like(p_pos, 1.0 / p_pos.size)
    return p_pos / s


def log_payoff(w: np.ndarray, winner: int) -> float:
    # log(w_winner); guard against w_winner = 0.
    return float(np.log(max(w[winner], EPS)))


def brier(p: np.ndarray, res: np.ndarray) -> float:
    """Sum of squared coord-wise errors against the binary resolution
    vector."""
    return float(np.sum((p - res) ** 2))


def paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(123)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    return (
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


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
    print(
        f"Compositional benchmark: {len(keep_idx)} cliques × {SEEDS} seeds"
    )

    rows: list[dict] = []
    skipped_misalign = 0

    for seed in range(SEEDS):
        for ci in keep_idx:
            relation = str(rels[ci])
            m = int(sizes[ci])
            assignment = rng.integers(0, N_MODELS, size=m)

            # Skip if any assigned model has fewer than m valid outcomes
            # for this clique. This affects only partition cliques where
            # per-model sizes can disagree (data layout artifact).
            if relation == "partition":
                if any(int(sizes_full[a, ci]) < m for a in assignment):
                    skipped_misalign += 1
                    continue

            # Resolution: pull from model 0 (canonical). For non-partition
            # relations, all models record the same resolution at the same
            # indices; for partitions we already gated on size match.
            res = res_full[0, ci, :m]
            # Brier is well-defined for any 0/1 resolution vector;
            # log-payoff only makes sense when at least one coord resolved
            # True (so a "winner" exists). For partitions exactly one coord
            # resolves True; for neg exactly one of {Q, ¬Q} resolves True;
            # for and/or zero or more can resolve True. We always compute
            # Brier; we report log-payoff only when a winner exists.
            has_winner = bool(np.any(res > 0.5))
            winner = int(np.argmax(res)) if has_winner else -1

            # Naive composed quote.
            p_naive = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            # JCD-repaired (joint projection).
            clique = make_clique(p_naive, relation)
            p_jcd = jcd_project(clique)
            # Single-LLM oracle: pick one model uniformly per bet (independent
            # of the per-coord assignment so it's a separate decision).
            oracle_model = int(rng.integers(0, N_MODELS))
            if relation == "partition" and int(sizes_full[oracle_model, ci]) < m:
                # Pick another oracle that has full coverage.
                cands = [
                    a for a in range(N_MODELS)
                    if int(sizes_full[a, ci]) >= m
                ]
                if not cands:
                    continue
                oracle_model = int(rng.choice(cands))
            p_oracle = forecast_jcd[oracle_model, ci, :m]

            # Allocations and payoffs.
            for label, p in (("naive", p_naive), ("jcd", p_jcd), ("oracle", p_oracle)):
                w = proportional_allocation(p)
                b = brier(p, res)
                logp = log_payoff(w, winner) if has_winner else None
                rows.append(
                    dict(
                        seed=int(seed),
                        clique_idx=int(ci),
                        relation=relation,
                        m=m,
                        regime=label,
                        winner=winner,
                        has_winner=has_winner,
                        log_payoff=logp,
                        brier=b,
                        sum_p=float(p.sum()),
                    )
                )

    print(
        f"Logged {len(rows)} rows from "
        f"{len(rows) // 3} bets × 3 regimes; skipped {skipped_misalign} bets "
        f"due to partition size misalignment."
    )

    # Aggregate per relation.
    summary = {}
    by_seed_clique = {}
    for r in rows:
        key = (r["seed"], r["clique_idx"])
        by_seed_clique.setdefault(key, {})[r["regime"]] = r

    paired = {r: dict(naive_lp=[], jcd_lp=[], oracle_lp=[],
                      naive_b=[], jcd_b=[], oracle_b=[])
              for r in KEEP}
    for key, regimes in by_seed_clique.items():
        if not all(k in regimes for k in ("naive", "jcd", "oracle")):
            continue
        rel = regimes["naive"]["relation"]
        # Brier always available.
        paired[rel]["naive_b"].append(regimes["naive"]["brier"])
        paired[rel]["jcd_b"].append(regimes["jcd"]["brier"])
        paired[rel]["oracle_b"].append(regimes["oracle"]["brier"])
        # Log-payoff only when there's a winner.
        if regimes["naive"]["has_winner"]:
            paired[rel]["naive_lp"].append(regimes["naive"]["log_payoff"])
            paired[rel]["jcd_lp"].append(regimes["jcd"]["log_payoff"])
            paired[rel]["oracle_lp"].append(regimes["oracle"]["log_payoff"])

    for r in KEEP:
        d = {k: np.array(v) for k, v in paired[r].items()}
        if d["naive_b"].size == 0:
            summary[r] = dict(n=0)
            continue
        # Brier-based summary (always available).
        jcd_minus_naive_b = d["jcd_b"] - d["naive_b"]
        oracle_minus_naive_b = d["oracle_b"] - d["naive_b"]
        s = dict(
            n_brier=int(d["naive_b"].size),
            mean_brier_naive=float(d["naive_b"].mean()),
            mean_brier_jcd=float(d["jcd_b"].mean()),
            mean_brier_oracle=float(d["oracle_b"].mean()),
            jcd_minus_naive_brier_mean=float(jcd_minus_naive_b.mean()),
            jcd_minus_naive_brier_ci=list(paired_bootstrap_ci(jcd_minus_naive_b)),
            oracle_minus_naive_brier_mean=float(oracle_minus_naive_b.mean()),
            oracle_minus_naive_brier_ci=list(paired_bootstrap_ci(oracle_minus_naive_b)),
        )
        # Log-payoff summary (only on bets with winners).
        if d["naive_lp"].size > 0:
            jcd_minus_naive_lp = d["jcd_lp"] - d["naive_lp"]
            oracle_minus_naive_lp = d["oracle_lp"] - d["naive_lp"]
            s.update(dict(
                n_logp=int(d["naive_lp"].size),
                mean_logp_naive=float(d["naive_lp"].mean()),
                mean_logp_jcd=float(d["jcd_lp"].mean()),
                mean_logp_oracle=float(d["oracle_lp"].mean()),
                jcd_minus_naive_logp_mean=float(jcd_minus_naive_lp.mean()),
                jcd_minus_naive_logp_ci=list(paired_bootstrap_ci(jcd_minus_naive_lp)),
                oracle_minus_naive_logp_mean=float(oracle_minus_naive_lp.mean()),
                oracle_minus_naive_logp_ci=list(paired_bootstrap_ci(oracle_minus_naive_lp)),
            ))
        summary[r] = s

    # Pooled across all relations.
    pooled_jn_b, pooled_on_b = [], []
    pooled_jn_lp, pooled_on_lp = [], []
    for key, regimes in by_seed_clique.items():
        if not all(k in regimes for k in ("naive", "jcd", "oracle")):
            continue
        pooled_jn_b.append(regimes["jcd"]["brier"] - regimes["naive"]["brier"])
        pooled_on_b.append(regimes["oracle"]["brier"] - regimes["naive"]["brier"])
        if regimes["naive"]["has_winner"]:
            pooled_jn_lp.append(regimes["jcd"]["log_payoff"] - regimes["naive"]["log_payoff"])
            pooled_on_lp.append(regimes["oracle"]["log_payoff"] - regimes["naive"]["log_payoff"])
    pooled_jn_b = np.array(pooled_jn_b)
    pooled_on_b = np.array(pooled_on_b)
    pooled_jn_lp = np.array(pooled_jn_lp)
    pooled_on_lp = np.array(pooled_on_lp)
    pooled = dict(
        n_brier=int(pooled_jn_b.size),
        jcd_minus_naive_brier_mean=float(pooled_jn_b.mean()),
        jcd_minus_naive_brier_ci=list(paired_bootstrap_ci(pooled_jn_b)),
        oracle_minus_naive_brier_mean=float(pooled_on_b.mean()),
        oracle_minus_naive_brier_ci=list(paired_bootstrap_ci(pooled_on_b)),
        n_logp=int(pooled_jn_lp.size),
        jcd_minus_naive_logp_mean=float(pooled_jn_lp.mean()),
        jcd_minus_naive_logp_ci=list(paired_bootstrap_ci(pooled_jn_lp)),
        oracle_minus_naive_logp_mean=float(pooled_on_lp.mean()),
        oracle_minus_naive_logp_ci=list(paired_bootstrap_ci(pooled_on_lp)),
    )

    out_doc = dict(
        meta=dict(
            seed=SEED, seeds=SEEDS, n_models=N_MODELS,
            skipped_misalign=skipped_misalign,
            allocation_rule="proportional",
        ),
        per_relation=summary,
        pooled=pooled,
    )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=out_doc, rows=rows), f, indent=2)

    # Console report.
    print("\n=== Downstream-decision regret ===")
    print("Brier per coord (sum); allocation = proportional; logp = log(w_winner).")
    print("\nPer relation — Brier (paired):")
    print(f"  {'relation':<10}{'n':>5}{'B(naive)':>10}{'B(jcd)':>10}{'B(oracle)':>11}"
          f"{'jcd-naive':>12}  95% CI")
    for r in KEEP:
        s = summary[r]
        if s.get("n_brier", 0) == 0:
            continue
        ci = s["jcd_minus_naive_brier_ci"]
        print(
            f"  {r:<10s}{s['n_brier']:>5d}"
            f"{s['mean_brier_naive']:>10.4f}{s['mean_brier_jcd']:>10.4f}"
            f"{s['mean_brier_oracle']:>11.4f}"
            f"{s['jcd_minus_naive_brier_mean']:>+12.4f}"
            f"  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    print("\nPer relation — log-payoff (only on bets with a winner):")
    print(f"  {'relation':<10}{'n':>5}{'lp(naive)':>11}{'lp(jcd)':>10}{'lp(oracle)':>12}"
          f"{'jcd-naive':>12}  95% CI")
    for r in KEEP:
        s = summary[r]
        if s.get("n_logp", 0) == 0:
            continue
        ci = s["jcd_minus_naive_logp_ci"]
        print(
            f"  {r:<10s}{s['n_logp']:>5d}"
            f"{s['mean_logp_naive']:>11.4f}{s['mean_logp_jcd']:>10.4f}"
            f"{s['mean_logp_oracle']:>12.4f}"
            f"{s['jcd_minus_naive_logp_mean']:>+12.4f}"
            f"  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    print(f"\nPooled (Brier, n={pooled['n_brier']}):")
    print(f"  jcd-naive   = {pooled['jcd_minus_naive_brier_mean']:+.4f}  "
          f"95% CI {pooled['jcd_minus_naive_brier_ci']}")
    print(f"  oracle-naive= {pooled['oracle_minus_naive_brier_mean']:+.4f}  "
          f"95% CI {pooled['oracle_minus_naive_brier_ci']}")
    print(f"\nPooled (log-payoff, n={pooled['n_logp']}):")
    print(f"  jcd-naive   = {pooled['jcd_minus_naive_logp_mean']:+.4f}  "
          f"95% CI {pooled['jcd_minus_naive_logp_ci']}")
    print(f"  oracle-naive= {pooled['oracle_minus_naive_logp_mean']:+.4f}  "
          f"95% CI {pooled['oracle_minus_naive_logp_ci']}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
