"""
Threshold-calibration / ROC for the eps_star-as-gate recommendation.

The paper recommends "use eps_star > tau as a gate" but never quantifies
what tau should be. This script answers two questions on the existing
1,770-bet downstream-regret dataset (no new compute, no API calls):

  Q1. Discrimination: does eps_star rank bets by realised harm?
      That is, do bets with high eps_star tend to have high realised
      log-payoff or Brier regret (after vs before JCD repair)?

  Q2. Operating point: what tau gives the best regret-saved per
      false-alarm tradeoff? We report ROC, AUC, and recommended
      tau values for two operating modes (high-recall, high-precision).

Outputs:
  - results/g1_threshold_roc.json  -- numeric results
  - figures/g1_eps_star_roc.pdf    -- ROC + cost-benefit figure

This is a DATA-only experiment; the eps_star and regret values are
recomputed from master_combined.npz to ensure each bet has both
quantities under the same seed/assignment.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

OUT_JSON = REPO_ROOT / "results" / "g1_threshold_roc.json"
OUT_FIG = REPO_ROOT / "figures" / "g1_eps_star_roc.pdf"

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


def proportional(p: np.ndarray) -> np.ndarray:
    pp = np.maximum(p, 0.0)
    s = pp.sum()
    if s < EPS:
        return np.full_like(pp, 1.0 / pp.size)
    return pp / s


def build_dataset() -> dict[str, np.ndarray]:
    """Reproduce the random-assignment ensemble; for each bet,
    record eps_star, Brier(naive), Brier(jcd), log-payoff(naive),
    log-payoff(jcd), winner index, has_winner, relation."""
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

    eps_list, b_n, b_j, lp_n, lp_j, has_w, rel_list, m_list = [], [], [], [], [], [], [], []

    for seed in range(SEEDS):
        for ci in keep_idx:
            relation = str(rels[ci])
            m = int(sizes[ci])
            assignment = rng.integers(0, N_MODELS, size=m)
            if relation == "partition":
                if any(int(sizes_full[a, ci]) < m for a in assignment):
                    continue
            res = res_full[0, ci, :m]
            has_winner = bool(np.any(res > 0.5))
            winner = int(np.argmax(res)) if has_winner else -1

            p_naive = np.array(
                [forecast_jcd[assignment[j], ci, j] for j in range(m)]
            )
            clique = make_clique(p_naive, relation)
            p_jcd = jcd_project(clique)

            eps_star = float(np.linalg.norm(p_naive - p_jcd))
            br_n = float(np.sum((p_naive - res) ** 2))
            br_j = float(np.sum((p_jcd - res) ** 2))
            if has_winner:
                w_n = proportional(p_naive)
                w_j = proportional(p_jcd)
                lp_naive = float(np.log(max(w_n[winner], EPS)))
                lp_jcd = float(np.log(max(w_j[winner], EPS)))
            else:
                lp_naive = lp_jcd = float("nan")

            eps_list.append(eps_star)
            b_n.append(br_n)
            b_j.append(br_j)
            lp_n.append(lp_naive)
            lp_j.append(lp_jcd)
            has_w.append(has_winner)
            rel_list.append(relation)
            m_list.append(m)

    return dict(
        eps=np.array(eps_list),
        brier_naive=np.array(b_n),
        brier_jcd=np.array(b_j),
        logp_naive=np.array(lp_n),
        logp_jcd=np.array(lp_j),
        has_winner=np.array(has_w, dtype=bool),
        relation=np.array(rel_list),
        m=np.array(m_list),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Simple Spearman implementation, no scipy dependency."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(np.dot(rx, ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + EPS))


def roc_curve(score: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Score is the predictor (higher = more likely positive); label is
    binary 0/1. Returns (fpr_grid, tpr_grid, auc)."""
    n_pos = int(label.sum())
    n_neg = int((~label.astype(bool)).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-score)  # descending
    sorted_label = label[order].astype(bool)
    tps = np.cumsum(sorted_label)
    fps = np.cumsum(~sorted_label)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # trapezoidal AUC
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def main() -> None:
    print("Building dataset (eps_star + regret per bet)...")
    d = build_dataset()
    n = len(d["eps"])
    n_w = int(d["has_winner"].sum())
    print(f"  total bets: {n}; with winner: {n_w}")

    # ---- Q1: discrimination ----
    eps = d["eps"]
    brier_regret = d["brier_naive"] - d["brier_jcd"]  # >= 0 by Pythagorean
    # Log-payoff regret only on bets with a winner
    mask_w = d["has_winner"]
    logp_regret = d["logp_jcd"] - d["logp_naive"]  # > 0 means JCD better

    print("\n=== Q1: discrimination of realised regret ===")
    print(f"  Spearman rho(eps_star, Brier regret):  {spearman(eps, brier_regret):.3f}")
    print(f"  Spearman rho(eps_star, log-p regret):  {spearman(eps[mask_w], logp_regret[mask_w]):.3f}")

    # Bin by eps quartile, report mean realised regret per bin
    q_edges = np.quantile(eps, [0, 0.25, 0.5, 0.75, 1.0])
    print("\n  Per-quartile of eps_star:")
    print(f"  {'quartile':<12}{'eps range':<22}{'mean Brier-regret':>18}{'mean logp-regret':>18}")
    for k in range(4):
        lo, hi = q_edges[k], q_edges[k + 1]
        if k < 3:
            in_bin = (eps >= lo) & (eps < hi)
        else:
            in_bin = (eps >= lo) & (eps <= hi)
        in_bin_w = in_bin & mask_w
        mean_b = float(brier_regret[in_bin].mean()) if in_bin.any() else float("nan")
        mean_lp = float((d["logp_jcd"][in_bin_w] - d["logp_naive"][in_bin_w]).mean()) \
            if in_bin_w.any() else float("nan")
        print(f"  Q{k+1:<10d}[{lo:.3f}, {hi:.3f}]{mean_b:>18.4f}{mean_lp:>18.4f}")

    # ---- Q2: ROC for "ε⋆ predicts a high-harm bet" ----
    # Define "harm" two ways for robustness:
    #   harm_b = top-quartile by Brier regret
    #   harm_lp = top-quartile by log-payoff regret (winner bets only)
    th_b = np.quantile(brier_regret, 0.75)
    th_lp = np.quantile(logp_regret[mask_w], 0.75)
    label_b = (brier_regret > th_b)
    label_lp = (logp_regret[mask_w] > th_lp)

    fpr_b, tpr_b, auc_b = roc_curve(eps, label_b)
    fpr_lp, tpr_lp, auc_lp = roc_curve(eps[mask_w], label_lp)

    print("\n=== Q2: ROC of eps_star as a harm predictor ===")
    print(f"  Brier-regret top-quartile harm threshold: {th_b:.4f}")
    print(f"  Log-p regret top-quartile harm threshold: {th_lp:.4f}")
    print(f"  AUC(eps -> top-quartile Brier regret):    {auc_b:.3f}")
    print(f"  AUC(eps -> top-quartile log-p regret):    {auc_lp:.3f}")

    # ---- Q3: operating thresholds ----
    # For a grid of tau, compute:
    #   alert rate (fraction of bets above tau)
    #   harm-capture rate (TPR on top-quartile harm)
    #   false-alarm rate (FPR on bottom three quartiles)
    #   regret saved per alert (mean regret in alerted bets)
    taus = np.unique(np.concatenate([
        np.linspace(0, eps.max(), 200),
        np.array([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]),
    ]))
    op_rows = []
    for tau in taus:
        alert = eps > tau
        n_alert = int(alert.sum())
        if n_alert == 0:
            continue
        tpr_brier = float(label_b[alert].mean()) if alert.any() else 0
        # Harm-capture: of all top-quartile-harm bets, what fraction are above tau?
        harm_b_caught = float((alert & label_b).sum() / max(label_b.sum(), 1))
        not_harm_alerted = float((alert & (~label_b)).sum() / max((~label_b).sum(), 1))
        regret_saved_per_alert = float(brier_regret[alert].mean())
        op_rows.append(dict(
            tau=float(tau),
            alert_rate=float(n_alert / n),
            harm_capture_rate=harm_b_caught,
            false_alarm_rate=not_harm_alerted,
            regret_saved_per_alert=regret_saved_per_alert,
            n_alerts=n_alert,
        ))

    # Recommended operating points
    # High-recall: smallest tau s.t. alert_rate <= 0.5 and harm_capture >= 0.9
    high_recall = None
    for r in op_rows:
        if r["harm_capture_rate"] >= 0.90 and r["alert_rate"] <= 0.50:
            if high_recall is None or r["tau"] > high_recall["tau"]:
                high_recall = r

    # High-precision: largest tau s.t. harm_capture >= 0.5 (catch half the harm
    # while alerting on as few bets as possible)
    high_precision = None
    for r in op_rows:
        if r["harm_capture_rate"] >= 0.50:
            if high_precision is None or r["tau"] > high_precision["tau"]:
                high_precision = r

    summary = dict(
        n_total=n,
        n_with_winner=n_w,
        spearman_eps_brier_regret=spearman(eps, brier_regret),
        spearman_eps_logp_regret=spearman(eps[mask_w], logp_regret[mask_w]),
        auc_eps_brier_top_quartile=auc_b,
        auc_eps_logp_top_quartile=auc_lp,
        brier_top_quartile_threshold=float(th_b),
        logp_top_quartile_threshold=float(th_lp),
        high_recall_op=high_recall,
        high_precision_op=high_precision,
    )

    print("\n=== Recommended operating points ===")
    if high_recall:
        print(f"  High-recall (catch >=90% of top-quartile harm, alert <=50% of bets):")
        print(f"    tau = {high_recall['tau']:.4f}, alert rate {high_recall['alert_rate']:.2%}, "
              f"harm capture {high_recall['harm_capture_rate']:.2%}, "
              f"FPR {high_recall['false_alarm_rate']:.2%}")
    if high_precision:
        print(f"  High-precision (largest tau still catching >=50% of harm):")
        print(f"    tau = {high_precision['tau']:.4f}, alert rate {high_precision['alert_rate']:.2%}, "
              f"harm capture {high_precision['harm_capture_rate']:.2%}, "
              f"FPR {high_precision['false_alarm_rate']:.2%}")

    # ---- Held-out cross-validation for thresholds and AUC ----
    # 5-fold on bets; in each fold, fit threshold on train and report
    # AUC + operating-point performance on test.
    rng_cv = np.random.default_rng(7)
    n_bets = len(eps)
    perm = rng_cv.permutation(n_bets)
    folds = np.array_split(perm, 5)
    cv_aucs_b, cv_aucs_lp = [], []
    cv_recall_at_tau = []
    cv_alert_at_tau = []
    target_tau = high_recall["tau"] if high_recall else 0.15
    for k in range(5):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
        # Fit harm threshold on train
        train_b = brier_regret[train_idx]
        th_b_train = float(np.quantile(train_b, 0.75))
        # Test set ROC against train threshold
        eps_test = eps[test_idx]
        b_test = brier_regret[test_idx]
        label_b_test = (b_test > th_b_train)
        _, _, auc_b_test = roc_curve(eps_test, label_b_test)
        cv_aucs_b.append(auc_b_test)
        # Operating point: alert rate + recall at tau
        alert_test = eps_test > target_tau
        recall = float((alert_test & label_b_test).sum() / max(label_b_test.sum(), 1))
        alert_rate = float(alert_test.mean())
        cv_recall_at_tau.append(recall)
        cv_alert_at_tau.append(alert_rate)
        # Log-payoff CV: only on winners in this fold
        idx_w_train = train_idx[mask_w[train_idx]]
        idx_w_test = test_idx[mask_w[test_idx]]
        if len(idx_w_train) > 5 and len(idx_w_test) > 5:
            th_lp_train = float(np.quantile(logp_regret[idx_w_train], 0.75))
            label_lp_test = (logp_regret[idx_w_test] > th_lp_train)
            _, _, auc_lp_test = roc_curve(eps[idx_w_test], label_lp_test)
            cv_aucs_lp.append(auc_lp_test)

    cv = dict(
        cv_auc_brier_mean=float(np.mean(cv_aucs_b)),
        cv_auc_brier_std=float(np.std(cv_aucs_b)),
        cv_auc_logp_mean=float(np.mean(cv_aucs_lp)),
        cv_auc_logp_std=float(np.std(cv_aucs_lp)),
        cv_recall_at_target_tau_mean=float(np.mean(cv_recall_at_tau)),
        cv_alert_at_target_tau_mean=float(np.mean(cv_alert_at_tau)),
        target_tau=float(target_tau),
        cv_aucs_brier=cv_aucs_b,
        cv_aucs_logp=cv_aucs_lp,
    )
    summary["cross_validation"] = cv
    print("\n=== Held-out 5-fold cross-validation ===")
    print(f"  CV-AUC (Brier top-Q):   {cv['cv_auc_brier_mean']:.3f} +/- {cv['cv_auc_brier_std']:.3f}")
    print(f"  CV-AUC (log-p top-Q):   {cv['cv_auc_logp_mean']:.3f} +/- {cv['cv_auc_logp_std']:.3f}")
    print(f"  At tau={target_tau:.3f}: alert {cv['cv_alert_at_target_tau_mean']:.2%}, "
          f"recall {cv['cv_recall_at_target_tau_mean']:.2%}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(summary=summary, op_table=op_rows), f, indent=2)

    # ---- Plotting ----
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel (a): ROC curve
    ax = axes[0]
    cv_b = float(np.mean(cv_aucs_b)) if cv_aucs_b else auc_b
    cv_lp = float(np.mean(cv_aucs_lp)) if cv_aucs_lp else auc_lp
    ax.plot(fpr_b, tpr_b, "C0-", lw=1.6,
            label=f"Brier-regret top quartile (CV-AUC={cv_b:.3f})")
    ax.plot(fpr_lp, tpr_lp, "C1-", lw=1.6,
            label=f"log-payoff regret top quartile (CV-AUC={cv_lp:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=0.8)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Harm-capture rate (TPR)")
    ax.set_title("(a) $\\epsilon^\\star$ as harm predictor")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel (b): operating curve -- alert rate vs harm capture, with tau annotated
    ax = axes[1]
    op_arr = sorted(op_rows, key=lambda r: r["tau"])
    alert_rates = [r["alert_rate"] for r in op_arr]
    harm_caps = [r["harm_capture_rate"] for r in op_arr]
    taus_sorted = [r["tau"] for r in op_arr]
    ax.plot(alert_rates, harm_caps, "C2-", lw=1.6)
    # Annotate selected tau values
    for tau_marker in [0.01, 0.05, 0.1, 0.2]:
        # find closest tau in op_rows
        diffs = [abs(r["tau"] - tau_marker) for r in op_arr]
        i = int(np.argmin(diffs))
        ax.scatter(op_arr[i]["alert_rate"], op_arr[i]["harm_capture_rate"],
                   color="C2", s=18, zorder=5)
        ax.annotate(f"$\\tau$={tau_marker}",
                    (op_arr[i]["alert_rate"], op_arr[i]["harm_capture_rate"]),
                    textcoords="offset points", xytext=(5, -3), fontsize=7)
    ax.set_xlabel("Alert rate (fraction of bets escalated)")
    ax.set_ylabel("Harm-capture rate")
    ax.set_title("(b) Operating curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"\nWritten {OUT_JSON}")
    print(f"Written {OUT_FIG}")


if __name__ == "__main__":
    main()
