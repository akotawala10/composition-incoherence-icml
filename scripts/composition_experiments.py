"""
Phase 1 follow-up experiments on existing per-component data
(figures/combined_*.npz). No new LLM calls.

Implements:
  E1  Compositional projection ablation:
        A = raw composed (no per-component JCD, no compositional projection)
        B = per-component JCD composed (current "naive ensemble")
        C = raw composed, then compositional projection
        D = per-component JCD composed, then compositional projection ("hierarchical")
  E2  Aggregator ablation: owner-selection vs. coord-wise mean vs. log-pool.
  E3  On partitions: full QP projection vs. naive simplex normalisation.
  E4  Partial K-sweep: K in {2, 4, 8} from existing 8-sample dumps.
  E5' Same-model proxy: split each LLM's 8 samples into 4 K=2 specialists.
  E6  Compositional Brier vs. resolved labels.
  E7  Disagreement mechanism check (regress eps_star on disagreement).
  E8  Downstream decision quality (threshold action; expected regret).

Outputs structured numbers to stdout; writes a CSV summary to
``results_phase1_followups.csv`` for easy reference.
"""

from __future__ import annotations

import os
import csv
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
SEEDS = 4
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


def naive_simplex(p: np.ndarray) -> np.ndarray:
    """Naive normalisation: clip to >=0 then divide by sum (else uniform)."""
    q = np.clip(p, 0.0, None)
    s = q.sum()
    if s <= 1e-12:
        return np.full_like(p, 1.0 / len(p))
    return q / s


def exposure_per_relation(rel: str, x: np.ndarray) -> float:
    """Closed-form unit-stake LMSR exposure diagnostic per relation."""
    if rel == "neg":
        return float(abs(x[0] + x[1] - 1.0))
    if rel == "partition":
        return float(abs(x.sum() - 1.0))
    if rel == "and":
        # x_3 must be in [max(0, x_1+x_2-1), min(x_1, x_2)]
        lo = max(0.0, x[0] + x[1] - 1.0)
        hi = min(x[0], x[1])
        v = max(0.0, x[2] - hi) + max(0.0, lo - x[2])
        return float(v)
    if rel == "or":
        # x_3 must be in [max(x_1, x_2), min(1, x_1+x_2)]
        lo = max(x[0], x[1])
        hi = min(1.0, x[0] + x[1])
        v = max(0.0, x[2] - hi) + max(0.0, lo - x[2])
        return float(v)
    return 0.0


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def regret_threshold(p: np.ndarray, y: np.ndarray, theta: float = 0.5) -> float:
    """Per-coord regret of acting iff p > theta vs. always-correct oracle.
    Loss = 1 if action != label; expected regret over coords."""
    a = (p > theta).astype(float)
    return float(np.mean(np.abs(a - y)))


def load_data() -> dict:
    data = {}
    for n in MODELS:
        npz = np.load(FIG / f"combined_{n}.npz", allow_pickle=True)
        rels = json.load(open(FIG / f"combined_{n}.json"))["relations"]
        data[n] = dict(
            jcd=npz["forecast__JCD"],
            raw=npz["forecast__B2_ksample_mean"],
            samples=npz["samples"],         # (603, 8, 8)
            sizes=npz["clique_sizes"],
            resolutions=npz["resolutions"],
            rels=rels,
        )
    rels_ref = data[MODELS[0]]["rels"]
    sizes_ref = data[MODELS[0]]["sizes"]
    res_ref = data[MODELS[0]]["resolutions"]
    return data, rels_ref, sizes_ref, res_ref


def kmean_jcd(samples_K: np.ndarray, clique: Clique) -> np.ndarray:
    """Build JCD output from K samples: take mean then project."""
    raw = samples_K.mean(axis=-1)  # (m,)
    return jcd_project(clique, raw)


def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    data, relations, sizes, resolutions = load_data()

    # Pre-allocate canonical (K=8) JCD per (model, clique, coord) for the 4
    # relations of interest.  Raw means are also pre-stacked.
    jcd_clean = np.full((len(MODELS), len(relations), 8), np.nan)
    raw_stack = np.full((len(MODELS), len(relations), 8), np.nan)
    for t, rel in enumerate(relations):
        if rel not in RELATIONS_OF_INTEREST:
            continue
        m = int(sizes[t])
        try:
            clique = build_clique(rel, m)
        except ValueError:
            continue
        for b, n in enumerate(MODELS):
            jcd_full = data[n]["jcd"][t, :m]
            raw_full = data[n]["raw"][t, :m]
            jcd_clean[b, t, :m] = jcd_project(clique, jcd_full)
            raw_stack[b, t, :m] = raw_full

    # Per-clique outputs: stratified by (relation, seed)
    rows = []  # one dict per (clique, seed)
    for s in range(SEEDS):
        # each seed reseeds assignments deterministically
        rng_s = np.random.default_rng(MASTER_SEED + s)
        for t, rel in enumerate(relations):
            if rel not in RELATIONS_OF_INTEREST:
                continue
            m = int(sizes[t])
            try:
                clique = build_clique(rel, m)
            except ValueError:
                continue
            jcd_t = jcd_clean[:, t, :m]   # (M, m)
            raw_t = raw_stack[:, t, :m]   # (M, m)
            y = resolutions[t, :m]

            assign = rng_s.integers(0, len(MODELS), size=m)

            # E1: four operators
            x_raw_compose      = raw_t[assign, np.arange(m)]
            x_jcd_compose      = jcd_t[assign, np.arange(m)]
            x_raw_then_proj    = jcd_project(clique, x_raw_compose)
            x_jcd_then_proj    = jcd_project(clique, x_jcd_compose)

            # E2: aggregator alternatives (mean, log-pool)
            x_avg = jcd_t.mean(axis=0)                 # coord-wise mean of 4
            # log-pool on (eps,1-eps)-clipped probabilities, normalise per relation
            eps = 1e-6
            log_p = np.log(np.clip(jcd_t, eps, 1 - eps))
            x_logp_unnorm = np.exp(log_p.mean(axis=0))   # geometric mean
            x_logp_unnorm = np.clip(x_logp_unnorm, 0.0, 1.0)
            x_avg_proj  = jcd_project(clique, x_avg)
            x_logp_proj = jcd_project(clique, x_logp_unnorm)

            # E3: naive normalisation as alternative to QP on partition only
            x_naive_simplex = naive_simplex(x_jcd_compose) if rel == "partition" else None

            # E5': same-model proxy via split-K
            #   split the 8-sample tensor of one model into 4 K=2 chunks; each chunk
            #   becomes a "specialist".  We rotate the model used so all four sit in
            #   the panel across cliques (master seed determines which model per
            #   (relation, seed) combination).
            sm_model = (MASTER_SEED + s + t) % len(MODELS)
            samp = data[MODELS[sm_model]]["samples"][t, :m, :]   # (m, 8)
            split = samp.reshape(m, 4, 2).mean(axis=-1)            # (m, 4) K=2 means
            split_jcd = np.stack([jcd_project(clique, split[:, b]) for b in range(4)])  # (4, m)
            assign_sm = rng_s.integers(0, 4, size=m)
            x_sm_compose = split_jcd[assign_sm, np.arange(m)]
            x_sm_proj    = jcd_project(clique, x_sm_compose)

            # E4: K-sweep (K=2, K=4, K=8) per LLM, then ensemble.  K=8 == jcd_t above.
            # We build, for each model, the K-sample-mean JCD at K in {2,4} from samples.
            # samples shape: (603, 8, 8) per model -> per model: (m, 8) for this clique.
            ks_records = {}
            for K in (2, 4, 8):
                jcd_K = np.full((len(MODELS), m), np.nan)
                for b, n in enumerate(MODELS):
                    s_full = data[n]["samples"][t, :m, :K]  # (m, K)
                    if np.isnan(s_full).any():
                        # samples can be NaN-padded for unused coords
                        s_full = np.where(np.isnan(s_full), 0.0, s_full)
                    raw_K = s_full.mean(axis=-1)
                    jcd_K[b] = jcd_project(clique, raw_K)
                x_K_compose = jcd_K[assign, np.arange(m)]
                x_K_proj    = jcd_project(clique, x_K_compose)
                ks_records[K] = (
                    float(np.linalg.norm(x_K_compose - x_K_proj)),
                    exposure_per_relation(rel, x_K_compose),
                )

            # ε* under each operator
            eps_naive   = float(np.linalg.norm(x_jcd_compose - jcd_project(clique, x_jcd_compose)))
            eps_avg     = float(np.linalg.norm(x_avg        - x_avg_proj))
            eps_logp    = float(np.linalg.norm(x_logp_unnorm - x_logp_proj))
            eps_raw     = float(np.linalg.norm(x_raw_compose - x_raw_then_proj))
            eps_sm      = float(np.linalg.norm(x_sm_compose - x_sm_proj))

            exposure_naive = exposure_per_relation(rel, x_jcd_compose)
            exposure_proj  = exposure_per_relation(rel, x_jcd_then_proj)
            exposure_avg   = exposure_per_relation(rel, x_avg)
            exposure_logp  = exposure_per_relation(rel, x_logp_unnorm)
            exposure_raw   = exposure_per_relation(rel, x_raw_compose)
            exposure_sm    = exposure_per_relation(rel, x_sm_compose)

            # disagreement: max-pair L2 distance among the 4 JCD'd LLMs
            d2 = []
            for a in range(len(MODELS)):
                for b in range(a + 1, len(MODELS)):
                    d2.append(float(np.linalg.norm(jcd_t[a] - jcd_t[b])))
            disagreement_max = max(d2)
            disagreement_mean = float(np.mean(d2))
            # Prop 3.7-flavoured assignment-aware bound:
            #   eps* <= min_beta || composed - jcd_beta ||_2
            disagreement_prop = float(min(np.linalg.norm(x_jcd_compose - jcd_t[b]) for b in range(len(MODELS))))

            # Brier vs labels (compositional)
            brier_naive   = brier(x_jcd_compose, y)
            brier_proj    = brier(x_jcd_then_proj, y)
            brier_avg_proj = brier(x_avg_proj, y)
            brier_logp_proj = brier(x_logp_proj, y)
            brier_raw     = brier(x_raw_compose, y)
            brier_raw_proj = brier(x_raw_then_proj, y)
            if x_naive_simplex is not None:
                brier_naive_simplex = brier(x_naive_simplex, y)
                eps_naive_simplex = float(np.linalg.norm(x_naive_simplex - jcd_project(clique, x_naive_simplex)))
            else:
                brier_naive_simplex = float("nan")
                eps_naive_simplex = float("nan")

            # Decision regret
            reg_naive = regret_threshold(x_jcd_compose, y)
            reg_proj  = regret_threshold(x_jcd_then_proj, y)
            reg_avg   = regret_threshold(x_avg_proj, y)
            reg_logp  = regret_threshold(x_logp_proj, y)

            rows.append(dict(
                relation=rel, m=m, seed=s, clique_idx=t,
                eps_naive=eps_naive, eps_proj=0.0,            # post-projection eps is 0
                eps_avg=eps_avg, eps_logp=eps_logp,
                eps_raw=eps_raw, eps_sm=eps_sm,
                exposure_naive=exposure_naive, exposure_proj=exposure_proj,
                exposure_avg=exposure_avg, exposure_logp=exposure_logp,
                exposure_raw=exposure_raw, exposure_sm=exposure_sm,
                eps_K2=ks_records[2][0], eps_K4=ks_records[4][0], eps_K8=ks_records[8][0],
                exposure_K2=ks_records[2][1], exposure_K4=ks_records[4][1], exposure_K8=ks_records[8][1],
                disagreement_max=disagreement_max,
                disagreement_mean=disagreement_mean,
                disagreement_prop=disagreement_prop,
                brier_naive=brier_naive, brier_proj=brier_proj,
                brier_avg_proj=brier_avg_proj, brier_logp_proj=brier_logp_proj,
                brier_raw=brier_raw, brier_raw_proj=brier_raw_proj,
                brier_naive_simplex=brier_naive_simplex, eps_naive_simplex=eps_naive_simplex,
                reg_naive=reg_naive, reg_proj=reg_proj,
                reg_avg=reg_avg, reg_logp=reg_logp,
            ))

    # Write CSV
    out_csv = CTB / "results_phase1_followups.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_csv}\n")

    # ---------------- Summaries ----------------
    by_rel = defaultdict(list)
    for r in rows:
        by_rel[r["relation"]].append(r)

    def agg(rel_rows, key):
        v = [r[key] for r in rel_rows if not np.isnan(r[key])]
        return (np.mean(v) if v else float("nan"), np.median(v) if v else float("nan"))

    print("============================================================")
    print("E1: Compositional projection ablation")
    print("    (mean eps* and mean exposure under each operator)")
    print("============================================================")
    print(f"{'rel':<10}{'op':<22}{'<eps>':>10}{'<exposure>':>14}{'<Brier>':>12}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        for op_label, eps_key, expo_key, brier_key in [
            ("A: raw composed",      "eps_raw",   "exposure_raw",   "brier_raw"),
            ("B: JCD composed",      "eps_naive", "exposure_naive", "brier_naive"),
            ("C: raw + Π*",          "eps_raw",   "exposure_raw",   "brier_raw_proj"),  # eps_raw also -> 0 post-proj
            ("D: JCD + Π* (hier)",   "eps_naive", "exposure_naive", "brier_proj"),
        ]:
            # post-projection eps is 0 by construction; use composed eps for A/B
            eps_m, _ = agg(rs, eps_key)
            expo_m, _ = agg(rs, expo_key)
            brier_m, _ = agg(rs, brier_key)
            tag = "(pre-proj)" if op_label.startswith(("A:", "B:")) else "(post-proj)"
            display_eps = eps_m if op_label.startswith(("A:", "B:")) else 0.0
            display_expo = expo_m if op_label.startswith(("A:", "B:")) else 0.0
            print(f"{rel:<10}{op_label:<22}{display_eps:>10.4f}{display_expo:>14.4f}{brier_m:>12.4f}")
        print()

    print("============================================================")
    print("E2: Aggregator ablation (owner-selection vs. mean vs. log-pool)")
    print("    (mean eps* of composed quote, before joint projection)")
    print("============================================================")
    print(f"{'rel':<10}{'owner-sel':>12}{'mean':>12}{'log-pool':>12}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        own, _ = agg(rs, "eps_naive")
        avg, _ = agg(rs, "eps_avg")
        logp, _ = agg(rs, "eps_logp")
        print(f"{rel:<10}{own:>12.4f}{avg:>12.4f}{logp:>12.4f}")

    print()
    print(f"{'rel':<10}{'frac_pos owner':>16}{'frac_pos mean':>16}{'frac_pos logp':>16}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        pos_own = np.mean([r["eps_naive"] > 1e-4 for r in rs])
        pos_avg = np.mean([r["eps_avg"] > 1e-4 for r in rs])
        pos_lp  = np.mean([r["eps_logp"] > 1e-4 for r in rs])
        print(f"{rel:<10}{pos_own:>16.3f}{pos_avg:>16.3f}{pos_lp:>16.3f}")

    print()
    print("============================================================")
    print("E3: On partitions, hierarchical QP vs. naive simplex")
    print("============================================================")
    rs = by_rel["partition"]
    qp_eps = np.mean([r["eps_naive"] for r in rs])
    naive_eps = np.mean([r["eps_naive_simplex"] for r in rs])
    qp_brier = np.mean([r["brier_proj"] for r in rs])
    naive_brier = np.mean([r["brier_naive_simplex"] for r in rs])
    print(f"  hierarchical QP:    pre-proj eps = {qp_eps:.4f}, post-proj Brier = {qp_brier:.4f}")
    print(f"  naive normalise:    post-norm eps  = {naive_eps:.4f}, post-norm Brier = {naive_brier:.4f}")

    print()
    print("============================================================")
    print("E4: Partial K-sweep on compositional residual")
    print("    (mean eps* by relation; lower-K means each LLM averages 2 or 4 samples)")
    print("============================================================")
    print(f"{'rel':<10}{'K=2':>10}{'K=4':>10}{'K=8':>10}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        e2, _ = agg(rs, "eps_K2")
        e4, _ = agg(rs, "eps_K4")
        e8, _ = agg(rs, "eps_K8")
        print(f"{rel:<10}{e2:>10.4f}{e4:>10.4f}{e8:>10.4f}")

    print()
    print("============================================================")
    print("E5': Same-model proxy (4 K=2 specialists from one model)")
    print("============================================================")
    print(f"{'rel':<10}{'cross-model eps':>18}{'same-model eps':>18}{'ratio':>10}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        cm, _ = agg(rs, "eps_naive")
        sm, _ = agg(rs, "eps_sm")
        ratio = sm / cm if cm > 0 else float("nan")
        print(f"{rel:<10}{cm:>18.4f}{sm:>18.4f}{ratio:>10.3f}")

    print()
    print("============================================================")
    print("E6: Compositional Brier vs. resolved labels")
    print("    (paired per-clique, post-projection)")
    print("============================================================")
    print(f"{'rel':<10}{'Brier naive':>14}{'Brier hier':>14}{'ΔBrier':>12}{'p (paired-t)':>14}{'N':>6}")
    from scipy import stats as scistats
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        bn = np.array([r["brier_naive"] for r in rs])
        bp = np.array([r["brier_proj"] for r in rs])
        if len(bn) < 3 or np.std(bn - bp) < 1e-12:
            pval = float("nan")
        else:
            pval = float(scistats.ttest_rel(bn, bp).pvalue)
        print(f"{rel:<10}{bn.mean():>14.5f}{bp.mean():>14.5f}{(bn - bp).mean():>12.5f}{pval:>14.2e}{len(bn):>6}")

    print()
    print("============================================================")
    print("E7: Disagreement mechanism check")
    print("    (Prop-3.7 bound: eps* <= min_beta ||composed - jcd_beta||)")
    print("============================================================")
    print(f"{'rel':<10}{'slope':>10}{'R^2':>10}{'<bound/eps>':>14}{'N':>6}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        e = np.array([r["eps_naive"] for r in rs])
        d = np.array([r["disagreement_prop"] for r in rs])
        mask = e > 1e-4  # only positive-eps cliques
        if mask.sum() < 3 or np.std(d[mask]) < 1e-12:
            print(f"{rel:<10}{'nan':>10}{'nan':>10}{'nan':>14}{int(mask.sum()):>6}")
            continue
        slope, intercept = np.polyfit(d[mask], e[mask], 1)
        ssr = np.sum((e[mask] - (slope * d[mask] + intercept)) ** 2)
        sst = np.sum((e[mask] - e[mask].mean()) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
        ratio = (d[mask] / np.maximum(e[mask], 1e-12)).mean()
        print(f"{rel:<10}{slope:>10.3f}{r2:>10.3f}{ratio:>14.3f}{int(mask.sum()):>6}")

    print()
    print("============================================================")
    print("E8: Downstream decision quality (threshold = 0.5)")
    print("============================================================")
    print(f"{'rel':<10}{'reg naive':>12}{'reg hier':>12}{'Δreg':>12}")
    for rel in RELATIONS_OF_INTEREST:
        rs = by_rel[rel]
        rn = np.array([r["reg_naive"] for r in rs])
        rp = np.array([r["reg_proj"] for r in rs])
        print(f"{rel:<10}{rn.mean():>12.4f}{rp.mean():>12.4f}{(rn-rp).mean():>12.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
