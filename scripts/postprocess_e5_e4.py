"""
Post-process E5 (4 fresh same-model seeds) and E4 (K=32 sweep) outputs from
``results/e5_samemodel/`` and ``results/e4_ksweep/``.

E5 — Same-model 4-seed cross-confound control.
    For each clique present in all 4 seed dirs, build 4 same-model "specialists"
    (one per seed). Random assign each coord to one specialist; compute eps*.
    Compare to cross-model (paper's existing 4-LLM panel) result.

E4 — K-sweep K in {8, 16, 32}.
    For each clique, take the K=32 K-sample tensor; subsample for K=8 and K=16;
    compute JCD per (model, K), then ensemble (random assignment across the 4
    cross-model panel) and compute eps*. Sweep is extension of E4 from
    composition_experiments.py beyond K=8.
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
E5_DIR = CTB / "results" / "e5_samemodel"
E4_DIR = CTB / "results" / "e4_ksweep"
SEEDS_E5 = ("20260501", "20260502", "20260503", "20260504")
CHECKERS = ("NegChecker", "AndChecker", "OrChecker")
POLY_TAG = "polymarket-partition"
RELATIONS_OF_INTEREST = ("neg", "and", "or", "partition")
ENS_SEEDS = 16
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


def find_npz(root: Path, model_dir: str, source_label: str, K: int) -> Path | None:
    """Locate the .npz produced by run_real_llms.py."""
    fn = root / model_dir / f"{source_label}_K{K}.npz"
    return fn if fn.exists() else None


def load_seed_data(seed: str) -> dict[str, dict]:
    """Load all available checker results for a single E5 seed.
    Returns dict source_label -> dict(jcd, sizes, relations, samples, resolutions).
    """
    out = {}
    seed_dir = E5_DIR / f"seed_{seed}" / "anthropic_claude-haiku-4-5-20251001"
    if not seed_dir.exists():
        return out
    for checker in CHECKERS + (POLY_TAG,):
        npz = seed_dir / f"{checker}_K8.npz"
        js = seed_dir / f"{checker}_K8.json"
        if not (npz.exists() and js.exists()):
            continue
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(js.read_text())
        out[checker] = dict(
            samples=d["samples"],
            resolutions=d["resolutions"],
            sizes=d["clique_sizes"],
            relations=meta["relations"] if isinstance(meta, dict) and "relations" in meta else d["relations"].tolist(),
            jcd=d["forecast__JCD"] if "forecast__JCD" in d.files else None,
        )
    return out


def relation_for_checker(checker: str) -> str:
    return {
        "NegChecker": "neg",
        "AndChecker": "and",
        "OrChecker": "or",
        POLY_TAG: "partition",
    }[checker]


def e5_analysis() -> None:
    print("=" * 68)
    print("E5: same-model 4-seed cross-model-confound control")
    print("    (Anthropic Claude-Haiku-4.5; 4 fresh seeds; 16 ensemble draws/clique)")
    print("=" * 68)

    seed_data = {s: load_seed_data(s) for s in SEEDS_E5}
    available = [s for s, d in seed_data.items() if d]
    if len(available) < 4:
        print(f"WARNING: only {len(available)} seeds present, expected 4; results may be partial")

    by_rel = defaultdict(list)
    for checker in CHECKERS + (POLY_TAG,):
        rel = relation_for_checker(checker)
        # find cliques present in all 4 seeds for this checker
        per_seed = [seed_data[s].get(checker) for s in available]
        if not all(per_seed):
            continue
        n_per_seed = [d["samples"].shape[0] for d in per_seed]
        n = min(n_per_seed)
        if n == 0:
            continue
        sizes = per_seed[0]["sizes"][:n]
        for t in range(n):
            m = int(sizes[t])
            try:
                clique = build_clique(rel, m)
            except ValueError:
                continue
            # Build per-seed JCD on this clique
            Pi = []
            for d in per_seed:
                jcd_arr = d.get("jcd")
                if jcd_arr is None:
                    raw = d["samples"][t, :m, :].mean(axis=-1)
                    pij = jcd_project(clique, raw)
                else:
                    pij = jcd_project(clique, jcd_arr[t, :m])
                Pi.append(pij)
            Pi = np.stack(Pi)            # (4, m): same-model specialists
            # uniform random ensemble draws
            for s in range(ENS_SEEDS):
                rng = np.random.default_rng(MASTER_SEED + 17 * t + s)
                assign = rng.integers(0, 4, size=m)
                x = Pi[assign, np.arange(m)]
                proj = jcd_project(clique, x)
                eps = float(np.linalg.norm(x - proj))
                by_rel[rel].append(eps)

    print(f"{'rel':<10}{'<eps> same-model':>20}{'frac > 1e-4':>14}{'N draws':>10}")
    for rel in RELATIONS_OF_INTEREST:
        eps = np.array(by_rel[rel])
        if len(eps) == 0:
            print(f"{rel:<10}{'(no data)':>20}")
            continue
        print(f"{rel:<10}{eps.mean():>20.4f}{(eps > 1e-4).mean():>14.3f}{len(eps):>10}")

    # Compare to cross-model from previous CSV
    print()
    print("Reference cross-model results (from results_phase1_followups.csv):")
    print(f"{'rel':<10}{'<eps> cross-model':>20}{'<eps> same-model':>20}{'ratio sm/cm':>14}")
    csv_path = CTB / "results_phase1_followups.csv"
    if csv_path.exists():
        import csv
        cm_eps = defaultdict(list)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["relation"] in RELATIONS_OF_INTEREST:
                    cm_eps[row["relation"]].append(float(row["eps_naive"]))
        for rel in RELATIONS_OF_INTEREST:
            cm = np.mean(cm_eps[rel]) if cm_eps[rel] else float("nan")
            sm = np.mean(by_rel[rel]) if by_rel[rel] else float("nan")
            ratio = sm / cm if cm > 0 else float("nan")
            print(f"{rel:<10}{cm:>20.4f}{sm:>20.4f}{ratio:>14.3f}")


def e4_analysis() -> None:
    print()
    print("=" * 68)
    print("E4: K-sweep K in {8, 16, 32} on cross-model panel (subset)")
    print("=" * 68)

    MODEL_DIRS = {
        "anthropic_claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5-20251001",
        "azure_mini": "azure/mini",
        "azure_nano": "azure/nano",
    }
    MIN_MODELS = 3   # 3-model panel suffices when groq dropped due to TPD

    # Load K=32 npz per (model, checker)
    K_VALUES = (8, 16, 32)
    per_rel = defaultdict(lambda: defaultdict(list))  # per_rel[K][rel] = list of eps

    for checker in CHECKERS + (POLY_TAG,):
        rel = relation_for_checker(checker)
        # Find K=32 file per model
        per_model = {}
        for model_dir in MODEL_DIRS:
            fn = E4_DIR / model_dir / f"{checker}_K32.npz"
            if not fn.exists():
                continue
            d = np.load(fn, allow_pickle=True)
            per_model[model_dir] = d
        if len(per_model) < MIN_MODELS:
            print(f"WARNING: {checker} has only {len(per_model)} models with K=32 (<{MIN_MODELS}); skipping")
            continue
        n_panel = len(per_model)
        any_d = next(iter(per_model.values()))
        n = any_d["samples"].shape[0]
        sizes = any_d["clique_sizes"]
        for t in range(n):
            m = int(sizes[t])
            try:
                clique = build_clique(rel, m)
            except ValueError:
                continue
            for K in K_VALUES:
                # build per-model K-sample mean from K=32 samples (subsample first K)
                Pi = []
                for model_dir in per_model:
                    samp = per_model[model_dir]["samples"][t, :m, :K]  # (m, K)
                    raw = samp.mean(axis=-1)
                    Pi.append(jcd_project(clique, raw))
                Pi = np.stack(Pi)
                for s in range(ENS_SEEDS):
                    rng = np.random.default_rng(MASTER_SEED + 17 * t + s)
                    assign = rng.integers(0, n_panel, size=m)
                    x = Pi[assign, np.arange(m)]
                    proj = jcd_project(clique, x)
                    eps = float(np.linalg.norm(x - proj))
                    per_rel[K][rel].append(eps)

    print(f"{'rel':<10}{'<eps> K=8':>12}{'<eps> K=16':>12}{'<eps> K=32':>12}{'N':>8}")
    for rel in RELATIONS_OF_INTEREST:
        e8  = np.mean(per_rel[8][rel])  if per_rel[8][rel]  else float("nan")
        e16 = np.mean(per_rel[16][rel]) if per_rel[16][rel] else float("nan")
        e32 = np.mean(per_rel[32][rel]) if per_rel[32][rel] else float("nan")
        n   = len(per_rel[8][rel])
        print(f"{rel:<10}{e8:>12.4f}{e16:>12.4f}{e32:>12.4f}{n:>8}")


def main() -> None:
    e5_analysis()
    e4_analysis()


if __name__ == "__main__":
    main()
