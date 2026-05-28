"""Rebuild Figure 1 with bound-aligned panel (b) and 95% bootstrap CI bands.

Panel (a): empirical CDF of eps_star by relation type, with shaded 95%
bootstrap CI bands on the CDF curve (B = 1000 resamples).
Panel (b): empirical CDF of the exposure bound sqrt(m_star) * eps_star
under naive ensemble (with bootstrap CI), with hierarchical JCD and the
no-composition reference shown as floor lines.

Uses results_phase1_followups.csv (1,876 rows).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "data" / "results_phase1_followups.csv"
OUT_PDF = REPO / "paper" / "figures" / "compositional_bound.pdf"

REL_LABELS = {
    "neg": "neg (N=536)",
    "and": "and (N=536)",
    "or":  "or (N=536)",
    "partition": "partition (N=268)",
}
REL_COLORS = {
    "neg": "#1f77b4",
    "and": "#2ca02c",
    "or":  "#ff7f0e",
    "partition": "#d62728",
}


def load_rows() -> list[dict]:
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(r)
    return rows


def bootstrap_cdf_band(values: np.ndarray, *, x_grid: np.ndarray, n_boot: int = 1000,
                       alpha: float = 0.05, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise (1-alpha) bootstrap CI band of the ECDF on x_grid."""
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boots = np.empty((n_boot, len(x_grid)))
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        sample.sort()
        boots[b] = np.searchsorted(sample, x_grid, side="right") / n
    lo = np.quantile(boots, alpha / 2, axis=0)
    hi = np.quantile(boots, 1 - alpha / 2, axis=0)
    return lo, hi


def main() -> None:
    rows = load_rows()
    by_rel = defaultdict(list)
    for r in rows:
        by_rel[r["relation"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Panel (a): CDF of eps_star by relation, with bootstrap CI bands
    ax = axes[0]
    x_grid_a = np.linspace(0, 0.6, 200)
    for rel in ["neg", "and", "or", "partition"]:
        eps = np.array([r["eps_naive"] for r in by_rel[rel]])
        if len(eps) == 0:
            continue
        eps_sorted = np.sort(eps)
        cdf_curve = np.searchsorted(eps_sorted, x_grid_a, side="right") / len(eps)
        lo, hi = bootstrap_cdf_band(eps, x_grid=x_grid_a, n_boot=1000, alpha=0.05)
        c = REL_COLORS[rel]
        ax.fill_between(x_grid_a, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(x_grid_a, cdf_curve, color=c, label=REL_LABELS[rel], linewidth=1.6)
    ax.set_xlabel(r"compositional residual  $\varepsilon^\star$", fontsize=11)
    ax.set_ylabel("empirical CDF", fontsize=11)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    ax.set_title("(a)  Cross-model ensemble residual " +
                 r"(95% bootstrap CI shaded)", fontsize=10.5)

    # Panel (b): empirical CDF of sqrt(m_star) * eps_star under three regimes.
    ax = axes[1]
    bound_naive = np.array([np.sqrt(float(r["m"])) * r["eps_naive"] for r in rows])
    x_grid_b = np.linspace(0, 1.2, 200)
    bound_sorted = np.sort(bound_naive)
    cdf_naive = np.searchsorted(bound_sorted, x_grid_b, side="right") / len(bound_naive)
    lo, hi = bootstrap_cdf_band(bound_naive, x_grid=x_grid_b, n_boot=1000, alpha=0.05)

    ax.fill_between(x_grid_b, lo, hi, color="#d62728", alpha=0.18, linewidth=0)
    ax.plot(x_grid_b, cdf_naive, color="#d62728", linewidth=2.0,
            label=r"naive ensemble (mean $0.137$, $p_{95}\!=\!0.52$)")
    ax.axvline(0.0, color="#1f77b4", linewidth=2.5, alpha=0.85,
               label=r"hierarchical JCD (numerical floor)")
    ax.axvline(0.0, color="#7f7f7f", linewidth=1.5, linestyle=":",
               label=r"no-composition reference")
    ax.set_xlabel(r"exposure bound  $\sqrt{m^\star}\,\varepsilon^\star$", fontsize=11)
    ax.set_ylabel("empirical CDF", fontsize=11)
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    ax.set_title(r"(b)  Exposure bound  $\sqrt{m^\star}\,\varepsilon^\star$  (Cor.~3.5) " +
                 r"(95% bootstrap CI shaded)", fontsize=10.5)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    print(f"  bound_naive: mean={np.mean(bound_naive):.4f}, "
          f"median={np.median(bound_naive):.4f}, p95={np.quantile(bound_naive, 0.95):.4f}")


if __name__ == "__main__":
    main()
