"""Emit a LaTeX longtable of all N=100 case-study partitions for App. I."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "real_agent_results.json") as f:
    rows = json.load(f)

short = {
    "Claude-Haiku":    "C",
    "GPT-5.4-mini":    "M",
    "GPT-5.4-nano":    "N",
    "Llama-3.3-70b":   "L",
}

# Sort descending by eps^* so the worst cases appear first.
rows.sort(key=lambda r: -r["eps_star"])

lines = [
    r"\begin{longtable}{p{0.36\linewidth} c l r r}",
    r"\caption{Deployed-style multi-tool case study, all $N{=}100$ partitions, sorted by $\eps^\star$. Each specialist sees a single Bernoulli question; the agent assembles into a partition quote. LLM key: \texttt{C}=Claude-Haiku-4.5, \texttt{M}=GPT-5.4-mini, \texttt{N}=GPT-5.4-nano, \texttt{L}=Llama-3.3-70b. All 100 partitions exhibit positive compositional residual.}\label{tab:realagent}\\",
    r"\toprule",
    r"Partition & $m$ & per-outcome quote (LLM) & $\sum p_i$ & $\eps^\star$ \\",
    r"\midrule",
    r"\endfirsthead",
    r"\multicolumn{5}{l}{\small \emph{(continued from previous page)}}\\",
    r"\toprule",
    r"Partition & $m$ & per-outcome quote (LLM) & $\sum p_i$ & $\eps^\star$ \\",
    r"\midrule",
    r"\endhead",
    r"\midrule",
    r"\multicolumn{5}{r}{\small \emph{(continued on next page)}}\\",
    r"\endfoot",
    r"\bottomrule",
    r"\endlastfoot",
]
for r in rows:
    label = r["label"].replace("&", r"\&").replace("%", r"\%")
    if len(label) > 60:
        label = label[:58] + "..."
    quotes = ", ".join(
        f"{m:.2f}({short.get(sp, sp[0])})"
        for m, sp in zip(r["per_outcome_means"], r["assigned_specialists"])
    )
    m = len(r["per_outcome_means"])
    lines.append(f"{label} & {m} & \\scriptsize {quotes} & {r['sum']:.3f} & {r['eps_star']:.3f} \\\\")
lines.append(r"\end{longtable}")

out = ROOT / "figures/real_agent_table_full.tex"
out.write_text("\n".join(lines))
print(f"Wrote {out}")
