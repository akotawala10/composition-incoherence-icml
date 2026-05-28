"""Merge GPT-5.5 + Claude-Haiku planner runs and build the routing-protocol
ladder.

Outputs a unified results file and prints the headline ladder table.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CTB = Path(__file__).resolve().parent.parent / "data"

# Merge GPT-5.5: original 30 + new 70
gpt55_a = json.loads((CTB / "results/deployed_agent_results.json").read_text())
gpt55_b = json.loads((CTB / "results/deployed_agent_gpt55_30_99.json").read_text())
# Tag any missing planner field as gpt-5.5 (the original run was GPT-5.5)
for r in gpt55_a:
    r.setdefault("planner", "gpt-5.5")
for r in gpt55_b:
    r.setdefault("planner", "gpt-5.5")
gpt55_all = gpt55_a + gpt55_b
print(f"GPT-5.5 merged: {len(gpt55_all)} records ({len(gpt55_all)//2} partitions × 2 conditions)")

haiku_all = json.loads((CTB / "results/deployed_agent_haiku.json").read_text())
for r in haiku_all:
    r.setdefault("planner", "claude-haiku")
print(f"Claude-Haiku: {len(haiku_all)} records ({len(haiku_all)//2} partitions × 2 conditions)")

merged = gpt55_all + haiku_all
out = CTB / "results/deployed_agent_merged.json"
with open(out, "w") as f:
    json.dump(merged, f)
print(f"Merged -> {out}")

# Build the routing-protocol ladder
print()
print("=" * 90)
print("ROUTING-PROTOCOL LADDER")
print("=" * 90)
print(f"{'protocol':<32}{'<eps*>':>10}{'median':>10}{'frac>1e-3':>12}{'<delegate>':>14}{'N':>6}")
print("-" * 90)

# Pull context-ablation numbers (already analyzed)
ctx_path = CTB / "results/context_ablation_results.json"
if ctx_path.exists():
    ctx = json.loads(ctx_path.read_text())
    for cond in ("isolated", "listed", "full"):
        eps = np.array([r[cond]["eps_star"] for r in ctx])
        print(f"{'context: ' + cond:<32}{eps.mean():>10.4f}{np.median(eps):>10.4f}"
              f"{(eps > 1e-3).mean():>12.3f}{'1':>14}{len(eps):>6}")

# Planner runs
for planner in ("claude-haiku", "gpt-5.5"):
    for cond in ("unguided", "coherence-guided"):
        rs = [r for r in merged if r.get("planner") == planner and r.get("condition") == cond]
        if not rs:
            continue
        eps = np.array([r["eps_star"] for r in rs])
        delg = np.array([r.get("n_delegate_calls", 0) for r in rs])
        m_avg = float(np.mean([len(r.get("outcomes", [])) for r in rs]))
        delg_per_m = delg.mean() / m_avg if m_avg > 0 else 0
        label = f"{planner}/{cond}"
        print(f"{label:<32}{eps.mean():>10.4f}{np.median(eps):>10.4f}"
              f"{(eps > 1e-3).mean():>12.3f}{delg_per_m:>11.2f}× m"
              f"{len(rs):>6}")

print()
print("=" * 90)
print("HEADLINE COMPARISONS")
print("=" * 90)

# Bootstrap CIs on planner-condition mean eps*
from scipy import stats as sps
print(f"\n{'protocol':<32}{'<eps*>':>10}{'95% CI':>22}")
rng = np.random.default_rng(0)
def ci(arr, n=2000):
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)])
    return np.quantile(boots, [0.025, 0.975])

# context
if ctx_path.exists():
    for cond in ("isolated", "listed", "full"):
        eps = np.array([r[cond]["eps_star"] for r in ctx])
        lo, hi = ci(eps)
        print(f"{'context: ' + cond:<32}{eps.mean():>10.4f}    [{lo:.4f}, {hi:.4f}]")

for planner in ("claude-haiku", "gpt-5.5"):
    for cond in ("unguided", "coherence-guided"):
        rs = [r for r in merged if r.get("planner") == planner and r.get("condition") == cond]
        if not rs:
            continue
        eps = np.array([r["eps_star"] for r in rs])
        lo, hi = ci(eps)
        label = f"{planner}/{cond}"
        print(f"{label:<32}{eps.mean():>10.4f}    [{lo:.4f}, {hi:.4f}]")

print()
print("Wilcoxon paired (claude-haiku vs gpt-5.5, same condition):")
for cond in ("unguided", "coherence-guided"):
    h = sorted([r for r in merged if r.get("planner")=="claude-haiku" and r["condition"]==cond],
               key=lambda r: r["label"])
    g = sorted([r for r in merged if r.get("planner")=="gpt-5.5" and r["condition"]==cond],
               key=lambda r: r["label"])
    h_lab = {r["label"]: r["eps_star"] for r in h}
    g_lab = {r["label"]: r["eps_star"] for r in g}
    common = sorted(set(h_lab) & set(g_lab))
    a = np.array([h_lab[lbl] for lbl in common])
    b = np.array([g_lab[lbl] for lbl in common])
    if len(a) < 5:
        continue
    diff = a - b
    if (diff != 0).any():
        try:
            stat, p = sps.wilcoxon(a, b, alternative="greater")
        except Exception:
            stat, p = float("nan"), float("nan")
    else:
        p = float("nan")
    print(f"  {cond:<20}  N={len(common)}  mean(haiku - gpt55)={diff.mean():.5f}  p(haiku>gpt55)={p}")
