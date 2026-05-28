"""Analyze the deployed-agent experiment.

Loads results/deployed_agent_results.json and reports:
  - per-condition eps*, sum-violation, fraction with eps*>0
  - delegate-call counts per partition (cost picture)
  - comparison to "naive owner-selection" baseline (k_per_outcome=1)
  - which specialists the planner queried most often
  - examples of partitions where it queried lots vs. few times
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "deployed_agent_results.json"
data = json.loads(PATH.read_text())
print(f"Loaded {len(data)} records ({len(data)//2} partitions × 2 conditions)\n")

CONDS = ("unguided", "coherence-guided")
by_cond = {c: [r for r in data if r.get("condition") == c] for c in CONDS}
for c in CONDS:
    rs = by_cond[c]
    eps = np.array([r.get("eps_star", float("nan")) for r in rs])
    sumv = np.array([r.get("sum_violation", float("nan")) for r in rs])
    delg = np.array([r.get("n_delegate_calls", 0) for r in rs])
    print(f"=== {c} (N={len(rs)}) ===")
    print(f"  eps*: mean={eps.mean():.4f}  max={eps.max():.4f}  median={np.median(eps):.4f}")
    print(f"  |sum-1|: mean={sumv.mean():.4f}  max={sumv.max():.4f}")
    print(f"  delegate calls: mean={delg.mean():.1f}  median={np.median(delg):.0f}  min={delg.min()}  max={delg.max()}")
    print(f"  frac eps* > 1e-3: {(eps > 1e-3).mean():.3f}")
    # mean m to compute ratio over naive
    ms = np.array([len(r.get("outcomes", [])) for r in rs])
    naive_calls = ms.sum()
    actual_calls = delg.sum()
    print(f"  naive owner-selection cost: m calls/partition (sum across {len(rs)} = {int(naive_calls)})")
    print(f"  actual delegate-call cost: {int(actual_calls)} (ratio {actual_calls/max(naive_calls,1):.2f}× naive)")
    print()

# Specialist usage histogram
all_traces = []
for r in data:
    for t in r.get("trace", []):
        if t.get("tool") == "delegate_to_specialist":
            sid = t.get("args", {}).get("specialist_id", "?")
            all_traces.append(sid)
ctr = Counter(all_traces)
print("=== Specialist usage histogram (across all partitions × both conditions) ===")
for sid, cnt in ctr.most_common():
    print(f"  {sid:<14} {cnt:>5}")

print()
print("=== Top 5 partitions by delegate-call count (combined unguided+guided) ===")
combined = {}
for r in data:
    label = r.get("label")
    combined.setdefault(label, 0)
    combined[label] += r.get("n_delegate_calls", 0)
for lbl, n in sorted(combined.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {lbl[:55]:<55s}  {n}")

print("\n=== Bottom 5 partitions ===")
for lbl, n in sorted(combined.items(), key=lambda kv: kv[1])[:5]:
    print(f"  {lbl[:55]:<55s}  {n}")

print()
print("=== Comparison to ablation findings ===")
print("Recall context-ablation isolated/listed/full mean eps* on 100 partitions:")
print("  isolated   0.235")
print("  listed     0.090")
print("  full       0.081")
print("Deployed-agent (GPT-5.5 planner, free routing) mean eps* on 30 partitions:")
print(f"  unguided          {by_cond['unguided'][0].get('eps_star', 'NA'):.4f}" if by_cond['unguided'] else "")
print(f"  coherence-guided  {by_cond['coherence-guided'][0].get('eps_star', 'NA'):.4f}" if by_cond['coherence-guided'] else "")
e_un = np.array([r.get("eps_star", 0) for r in by_cond["unguided"]])
e_co = np.array([r.get("eps_star", 0) for r in by_cond["coherence-guided"]])
print(f"  unguided mean       {e_un.mean():.4f}   max  {e_un.max():.4f}")
print(f"  coherence-guided    {e_co.mean():.4f}   max  {e_co.max():.4f}")
