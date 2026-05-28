#!/usr/bin/env python3
"""Generate the intro figure: per-sector specialist quotes from the
'Largest US AI startup IPO 2026' partition of the planner-routing
harness. Each specialist's quote is below the unit ceiling, but they
sum to 2.50 against a partition constraint of 1.

Sober vertical-bar chart for an ML academic audience.
"""
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures" / "intro_failure_bars.pdf"

# (sector, quote)
quotes = [
    ("infrastructure", 0.73),
    ("model",          0.67),
    ("applications",   0.71),
    ("other",          0.39),
]
sectors = [s for s, _ in quotes]
values  = [v for _, v in quotes]
total   = sum(values)

plt.rcParams.update({
    "font.family":       "serif",
    "axes.labelsize":    8.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.titlesize":    9,
})

fig, ax = plt.subplots(figsize=(3.3, 2.2))

xs = list(range(len(quotes)))
ax.bar(xs, values, width=0.62, color="0.40", edgecolor="black",
       linewidth=0.5)

for x, v in zip(xs, values):
    ax.text(x, v + 0.025, f"{v:.2f}", ha="center", va="bottom",
            fontsize=8)

ax.set_xticks(xs)
ax.set_xticklabels(sectors)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel(r"specialist quote $p_i$")
ax.set_xlabel("partition outcome")

# Mathematical annotation of the violation, placed under the x-axis label
ax.text(
    0.5, -0.42,
    r"$\sum_{i=1}^{4} p_i = 2.50$" +
    r"  $\neq$  " +
    r"$1$  (partition constraint)",
    transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout(pad=0.2)
plt.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
print(f"wrote {OUT}")
