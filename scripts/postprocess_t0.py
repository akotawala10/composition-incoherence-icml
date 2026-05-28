"""Post-process the T=0 (greedy decoding) same-model control.

Mirrors the E5 same-model analysis but uses results/e5_greedy_t0/ dirs.
"""

from __future__ import annotations

import os
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
T0_DIR = CTB / "results" / "e5_greedy_t0"
SEEDS = ("20260601", "20260602", "20260603", "20260604")
CHECKERS = ("NegChecker", "AndChecker", "OrChecker")
POLY_TAG = "polymarket-partition"
RELATIONS_OF_INTEREST = ("neg", "and", "or", "partition")
ENS_SEEDS = 16
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


def relation_for_checker(checker: str) -> str:
    return {
        "NegChecker": "neg",
        "AndChecker": "and",
        "OrChecker": "or",
        POLY_TAG: "partition",
    }[checker]


def load_seed_data(seed: str) -> dict:
    out = {}
    seed_dir = T0_DIR / f"seed_{seed}" / "anthropic_claude-haiku-4-5-20251001"
    if not seed_dir.exists():
        return out
    for checker in CHECKERS + (POLY_TAG,):
        npz_p = seed_dir / f"{checker}_K8.npz"
        js_p = seed_dir / f"{checker}_K8.json"
        if not (npz_p.exists() and js_p.exists()):
            continue
        npz = np.load(npz_p, allow_pickle=True)
        meta = json.loads(js_p.read_text())
        rels = meta.get("relations") if isinstance(meta, dict) else None
        if rels is None and "relations" in npz.files:
            rels = npz["relations"].tolist()
        out[checker] = dict(
            samples=npz["samples"],
            sizes=npz["clique_sizes"],
            relations=rels,
            jcd=npz["forecast__JCD"] if "forecast__JCD" in npz.files else None,
        )
    return out


def main() -> None:
    seed_data = {s: load_seed_data(s) for s in SEEDS}
    available = [s for s, d in seed_data.items() if d]
    print(f"Available T=0 seeds: {len(available)}/{len(SEEDS)}: {available}")
    if len(available) < 2:
        print("WARNING: need >=2 seeds for ensemble; skipping")
        return

    by_rel = defaultdict(list)
    for checker in CHECKERS + (POLY_TAG,):
        rel = relation_for_checker(checker)
        per_seed = [seed_data[s].get(checker) for s in available]
        per_seed = [d for d in per_seed if d is not None]
        if len(per_seed) < 2:
            continue
        n = min(d["samples"].shape[0] for d in per_seed)
        sizes = per_seed[0]["sizes"][:n]
        for t in range(n):
            m = int(sizes[t])
            try:
                clique = build_clique(rel, m)
            except ValueError:
                continue
            Pi = []
            for d in per_seed:
                jcd_arr = d.get("jcd")
                if jcd_arr is None:
                    raw = d["samples"][t, :m, :].mean(axis=-1)
                    pij = jcd_project(clique, raw)
                else:
                    pij = jcd_project(clique, jcd_arr[t, :m])
                Pi.append(pij)
            Pi = np.stack(Pi)
            for s in range(ENS_SEEDS):
                rng = np.random.default_rng(MASTER_SEED + 17 * t + s)
                assign = rng.integers(0, len(Pi), size=m)
                x = Pi[assign, np.arange(m)]
                proj = jcd_project(clique, x)
                eps = float(np.linalg.norm(x - proj))
                by_rel[rel].append(eps)

    # Compare to T=0.7 same-model from postprocess_e5_e4.py output
    # Numbers from earlier run:
    t07_same_model = dict(neg=0.0250, and_=0.0193, or_=0.0397, partition=0.0582)
    t07_cross_model = dict(neg=0.1141, and_=0.0544, or_=0.0715, partition=0.0976)

    print()
    print("=" * 80)
    print("T=0 (greedy decoding) same-model control")
    print("=" * 80)
    print(f"{'rel':<12}{'<eps> T=0':>12}{'<eps> T=0.7':>14}{'<eps> cross':>14}{'T=0/T=0.7':>12}{'T=0/cross':>12}{'N draws':>10}")
    for rel in RELATIONS_OF_INTEREST:
        eps = np.array(by_rel[rel]) if by_rel[rel] else np.array([])
        if len(eps) == 0:
            print(f"{rel:<12}{'(no data)':>12}")
            continue
        key = rel + "_" if rel in {"and", "or"} else rel
        t07_sm = t07_same_model.get(key, float("nan"))
        t07_cm = t07_cross_model.get(key, float("nan"))
        m = eps.mean()
        ratio_sm = m / t07_sm if t07_sm > 0 else float("nan")
        ratio_cm = m / t07_cm if t07_cm > 0 else float("nan")
        print(f"{rel:<12}{m:>12.4f}{t07_sm:>14.4f}{t07_cm:>14.4f}{ratio_sm:>12.3f}{ratio_cm:>12.3f}{len(eps):>10}")

    print()
    print("Interpretation:")
    print("  - If T=0 << T=0.7 same-model: residual is sampling-noise-driven; the failure")
    print("    mode would be argued away by greedy decoding.")
    print("  - If T=0 ~~ T=0.7 same-model: residual is genuinely coordinate-isolation-")
    print("    driven; same model produces different outputs across coordinates regardless")
    print("    of decoding stochasticity (e.g., different prompt context per coord).")


if __name__ == "__main__":
    main()
