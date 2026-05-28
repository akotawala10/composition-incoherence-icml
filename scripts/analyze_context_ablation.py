"""Analyze context-sharing ablation results.

Loads results/context_ablation_results.json and reports:
  - aggregate eps* and |sum-1| per condition
  - paired comparison (Wilcoxon signed-rank) across conditions
  - fraction of cliques showing strict reduction at each step
  - per-partition shifts: which partitions move most / least
  - distribution of residuals (95% bootstrap CI)
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "context_ablation_results.json"
data = json.loads(PATH.read_text())
print(f"Loaded {len(data)} partitions\n")

CONDITIONS = ("isolated", "listed", "full")
eps = {c: np.array([r[c]["eps_star"] for r in data]) for c in CONDITIONS}
sumv = {c: np.array([r[c]["sum_violation"] for r in data]) for c in CONDITIONS}

print("=" * 78)
print("Aggregate per-condition statistics (N = 100 partitions)")
print("=" * 78)
print(f"{'condition':<12}{'mean eps*':>12}{'median':>10}{'p95':>10}{'mean|s-1|':>14}{'frac eps>1e-3':>18}")
for c in CONDITIONS:
    e = eps[c]
    s = sumv[c]
    print(f"{c:<12}{e.mean():>12.4f}{np.median(e):>10.4f}{np.quantile(e, 0.95):>10.4f}"
          f"{s.mean():>14.4f}{(e > 1e-3).mean():>18.3f}")

print()
print("=" * 78)
print("Bootstrap 95% CI on mean eps* (B = 2000)")
print("=" * 78)
rng = np.random.default_rng(0)
for c in CONDITIONS:
    e = eps[c]
    boots = np.array([rng.choice(e, size=len(e), replace=True).mean() for _ in range(2000)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    print(f"  {c:<12}  mean = {e.mean():.4f}   95% CI = [{lo:.4f}, {hi:.4f}]")

print()
print("=" * 78)
print("Paired comparisons (signed Wilcoxon, two-sided)")
print("=" * 78)
def paired(a, b, label):
    diff = a - b
    pos = float(np.mean(diff > 0))
    neg = float(np.mean(diff < 0))
    try:
        stat, p = sps.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    except Exception:
        stat, p = (float("nan"), float("nan"))
    print(f"  {label}")
    print(f"    mean diff = {diff.mean():.4f}    median diff = {np.median(diff):.4f}")
    print(f"    fraction strict decrease (a > b) = {pos:.3f}    increase = {neg:.3f}")
    print(f"    Wilcoxon p = {p:.3e}")

paired(eps["isolated"], eps["listed"], "eps_isolated > eps_listed?")
paired(eps["listed"], eps["full"], "eps_listed > eps_full?")
paired(eps["isolated"], eps["full"], "eps_isolated > eps_full?")

print()
print("=" * 78)
print("Top-5 partitions by ABSOLUTE drop (isolated -> full)")
print("=" * 78)
drops = eps["isolated"] - eps["full"]
order = np.argsort(-drops)
for k in order[:5]:
    print(f"  {data[k]['label']:55s}  iso={eps['isolated'][k]:.3f}  full={eps['full'][k]:.3f}"
          f"  drop={drops[k]:.3f}")

print()
print("=" * 78)
print("Bottom-5 partitions by drop (least helped by context)")
print("=" * 78)
for k in order[-5:]:
    print(f"  {data[k]['label']:55s}  iso={eps['isolated'][k]:.3f}  full={eps['full'][k]:.3f}"
          f"  drop={drops[k]:.3f}")

print()
print("=" * 78)
print("Partitions with eps_full > eps_isolated (context HURT)")
print("=" * 78)
hurt = [k for k in range(len(data)) if eps["full"][k] > eps["isolated"][k] + 1e-4]
print(f"  N = {len(hurt)}")
for k in hurt:
    print(f"  {data[k]['label']:55s}  iso={eps['isolated'][k]:.3f}  full={eps['full'][k]:.3f}")

print()
print("=" * 78)
print("Residual at full context: how often does coherence persist?")
print("=" * 78)
e_full = eps["full"]
print(f"  fraction with eps* < 0.05: {(e_full < 0.05).mean():.3f}")
print(f"  fraction with eps* < 0.10: {(e_full < 0.10).mean():.3f}")
print(f"  fraction with eps* > 0.20: {(e_full > 0.20).mean():.3f}")
print(f"  Note: full context with sum-to-1 instruction does NOT eliminate eps*.")
