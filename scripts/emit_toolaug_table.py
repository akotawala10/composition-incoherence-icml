"""Emit a LaTeX longtable summarising the tool-augmented A/B for App. J."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "tool_augmented_results.json") as f:
    tool_rows = json.load(f)
with open(ROOT / "real_agent_results.json") as f:
    base_rows = json.load(f)

base_by_label = {r["label"]: r for r in base_rows}

# Sort by absolute change (largest disagreement first) for editorial value.
joined = []
for r in tool_rows:
    b = base_by_label.get(r["label"])
    if b is None:
        continue
    joined.append((r, b, r["eps_star"] - b["eps_star"]))
joined.sort(key=lambda triple: -abs(triple[2]))

lines = [
    r"\begin{longtable}{p{0.46\linewidth} r r r r}",
    r"\caption{Tool-augmented specialist A/B, all $30$ matched partitions, sorted by $|\Delta\eps^\star|$. $\sum_t$ is the partition mass under tool-augmented retrieval; $\eps^\star_b$ and $\eps^\star_t$ are the no-tool baseline and tool-augmented compositional residuals.}\label{tab:toolaug}\\",
    r"\toprule",
    r"Partition & $\sum_t$ & $\eps^\star_b$ & $\eps^\star_t$ & $\Delta\eps^\star$ \\",
    r"\midrule",
    r"\endfirsthead",
    r"\multicolumn{5}{l}{\small \emph{(continued)}}\\",
    r"\toprule",
    r"Partition & $\sum_t$ & $\eps^\star_b$ & $\eps^\star_t$ & $\Delta\eps^\star$ \\",
    r"\midrule",
    r"\endhead",
    r"\midrule",
    r"\multicolumn{5}{r}{\small \emph{(continued on next page)}}\\",
    r"\endfoot",
    r"\bottomrule",
    r"\endlastfoot",
]
for r, b, d in joined:
    label = r["label"].replace("&", r"\&").replace("%", r"\%")
    if len(label) > 70:
        label = label[:68] + "..."
    arrow = r"$\uparrow$" if d > 0 else (r"$\downarrow$" if d < 0 else "")
    lines.append(
        f"{label} & {r['sum']:.3f} & {b['eps_star']:.3f} & "
        f"{r['eps_star']:.3f} & {d:+.3f}~{arrow} \\\\"
    )
lines.append(r"\end{longtable}")

out = ROOT / "figures/tool_aug_table.tex"
out.write_text("\n".join(lines))
print(f"Wrote {out}")
