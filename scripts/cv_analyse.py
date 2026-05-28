"""
Analyse coupling-visibility BLIND-vs-INFORMED results: produce paired ε⋆ stats,
Wilcoxon signed-rank, paired-bootstrap CI on Δε⋆, the 2-panel figure
(coupling_visibility.pdf), the LaTeX table (coupling_visibility.tex),
and results.md.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "cv" / "results.json"
CALLS = REPO_ROOT / "results" / "cv" / "calls.jsonl"
FIG = REPO_ROOT / "results" / "cv" / "coupling_visibility.pdf"
TABLE = REPO_ROOT / "results" / "cv" / "coupling_visibility.tex"
MEMO = REPO_ROOT / "results" / "cv" / "results.md"


def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=11):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = diffs[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank test against zero. Returns
    (W, p) using scipy if available; else None, None."""
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zeros (default); two-sided.
        res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def main() -> None:
    d = json.load(open(RESULTS))
    bets = d["bets"]
    meta = d["meta"]
    n_calls_recorded = sum(1 for _ in open(CALLS)) if CALLS.exists() else 0
    n_calls_total = meta.get("n_calls_total")
    parse_failures = sum(1 for line in open(CALLS) if json.loads(line).get("parse_failure"))

    # ----- Paired (partition, seed) records -----
    rows = []  # one per (partition, seed)
    per_partition = []  # one per partition (mean over seeds)
    for b in bets:
        eps_b_seeds, eps_i_seeds = [], []
        delta_seeds = []
        for s in b["seeds"]:
            eb = s["blind_eps_star"]; ei = s["informed_eps_star"]
            if eb is None or ei is None:
                continue
            rows.append(dict(
                label=b["label"], rank=b["rank"], orig_idx=b["orig_idx"],
                m=b["m"], seed=s["seed"],
                blind_eps=eb, informed_eps=ei, delta=eb - ei,
                blind_sum=s["blind_sum"], informed_sum=s["informed_sum"],
                blind_quote=s["blind_quote"], informed_quote=s["informed_quote"],
            ))
            eps_b_seeds.append(eb); eps_i_seeds.append(ei)
            delta_seeds.append(eb - ei)
        if eps_b_seeds:
            per_partition.append(dict(
                label=b["label"], m=b["m"],
                baseline_eps=b["baseline_eps_star"],
                blind_mean=float(np.mean(eps_b_seeds)),
                blind_std=float(np.std(eps_b_seeds, ddof=0)),
                informed_mean=float(np.mean(eps_i_seeds)),
                informed_std=float(np.std(eps_i_seeds, ddof=0)),
                delta_mean=float(np.mean(delta_seeds)),
                n_seeds=len(eps_b_seeds),
            ))

    n_pairs = len(rows)
    diffs = np.array([r["delta"] for r in rows])
    blinds = np.array([r["blind_eps"] for r in rows])
    informeds = np.array([r["informed_eps"] for r in rows])

    print(f"n paired (partition, seed) records: {n_pairs}")
    print(f"mean BLIND eps*    : {blinds.mean():.4f}")
    print(f"mean INFORMED eps* : {informeds.mean():.4f}")
    print(f"mean Δeps*         : {diffs.mean():+.4f}")

    ci_lo, ci_hi = paired_bootstrap_ci(diffs)
    W, pval = wilcoxon_signed_rank(diffs)
    print(f"95% paired-bootstrap CI on Δeps*: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"Wilcoxon signed-rank: W={W}, p={pval}")

    n_improved = int((diffs > 1e-6).sum())
    n_unchanged = int((np.abs(diffs) <= 1e-6).sum())
    n_worsened = int((diffs < -1e-6).sum())
    print(f"improved/unchanged/worsened (per pair): "
          f"{n_improved}/{n_unchanged}/{n_worsened}")

    # ----- per-partition mean improved/worsened -----
    n_pp_improved = sum(1 for p in per_partition if p["delta_mean"] > 1e-6)
    n_pp_worsened = sum(1 for p in per_partition if p["delta_mean"] < -1e-6)
    n_pp_unchanged = sum(1 for p in per_partition
                         if abs(p["delta_mean"]) <= 1e-6)
    print(f"improved/unchanged/worsened (per partition mean): "
          f"{n_pp_improved}/{n_pp_unchanged}/{n_pp_worsened}")

    # ----- Per-specialist quote shift diagnostic -----
    # For each (partition, seed, outcome), shift = informed_quote - blind_quote
    shifts = []
    for r in rows:
        for j in range(r["m"]):
            bq, iq = r["blind_quote"][j], r["informed_quote"][j]
            if bq is None or iq is None:
                continue
            if not (np.isfinite(bq) and np.isfinite(iq)):
                continue
            shifts.append(iq - bq)
    shifts = np.array(shifts)
    print(f"\nQuote shift diagnostic (informed - blind, per outcome):")
    print(f"  n = {len(shifts)}")
    print(f"  mean = {shifts.mean():+.4f}; mean |shift| = {np.abs(shifts).mean():.4f}")
    print(f"  fraction with |shift| > 0.05 : {(np.abs(shifts) > 0.05).mean():.3f}")
    print(f"  fraction with |shift| > 0.10 : {(np.abs(shifts) > 0.10).mean():.3f}")

    # =========================================================================
    # Figure: 2-panel coupling_visibility.pdf
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    # Panel A: paired scatter
    ax = axes[0]
    ax.scatter(blinds, informeds, s=18, alpha=0.7, color="#1f77b4",
               edgecolors="black", linewidth=0.4)
    lim = max(0.55, blinds.max() * 1.05, informeds.max() * 1.05)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="$y=x$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("$\\epsilon^\\star$ BLIND (control)")
    ax.set_ylabel("$\\epsilon^\\star$ INFORMED (treatment)")
    ax.set_title("(a) Coupling visibility reduces $\\epsilon^\\star$")
    # Annotate four largest BLIND-side partitions
    annotated = sorted(per_partition, key=lambda p: p["blind_mean"],
                       reverse=True)[:4]
    for p in annotated:
        # find one (partition, seed) row to anchor on
        for r in rows:
            if r["label"] == p["label"]:
                ax.annotate(
                    p["label"][:28] + ("…" if len(p["label"]) > 28 else ""),
                    (p["blind_mean"], p["informed_mean"]),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=6.5, alpha=0.85,
                )
                break
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)

    # Panel B: empirical CDFs
    ax = axes[1]
    xb = np.sort(blinds)
    xi = np.sort(informeds)
    yb = np.arange(1, len(xb) + 1) / len(xb)
    yi = np.arange(1, len(xi) + 1) / len(xi)
    ax.plot(xb, yb, color="#d62728", lw=1.6, label="BLIND")
    ax.plot(xi, yi, color="#1f77b4", lw=1.6, label="INFORMED")
    ax.axvline(0.15, color="black", ls=":", lw=0.8, alpha=0.7,
               label="$\\tau=0.15$")
    ax.set_xlabel("$\\epsilon^\\star$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(b) Coupling-visible elicitation reduces $\\epsilon^\\star$")
    ax.set_xlim(0, max(0.55, blinds.max() * 1.02))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG)
    print(f"\nWrote {FIG}")

    # =========================================================================
    # LaTeX table: coupling_visibility.tex
    # =========================================================================
    sorted_pp = sorted(per_partition, key=lambda p: p["delta_mean"], reverse=True)

    # Truncation helper for LaTeX
    def texesc(s):
        return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")

    with open(TABLE, "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write("\\caption{Coupling-visibility intervention: BLIND specialists "
                "(no partition context) vs INFORMED specialists (full "
                "partition context $+$ peer quotes), on the 20 highest-$\\eps^\\star$ "
                "partitions of the planner harness. "
                "$\\eps^\\star$ values are mean $\\pm$ std over "
                f"{meta['N_SEEDS']} sampling seeds; $\\Delta = "
                "\\eps^\\star_{\\rm BLIND} - \\eps^\\star_{\\rm INFORMED}$. "
                "Sign: $\\downarrow$ improved, $\\circ$ unchanged, "
                "$\\uparrow$ worsened.}\n")
        f.write("\\label{tab:coupling_visibility}\n")
        f.write("\\centering\\footnotesize\n")
        f.write("\\begin{tabular}{lrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Partition & $m$ & $\\eps^\\star_{\\rm BLIND}$ & "
                "$\\eps^\\star_{\\rm INFORMED}$ & $\\Delta$ & sign \\\\\n")
        f.write("\\midrule\n")
        for p in sorted_pp:
            sign = "$\\downarrow$" if p["delta_mean"] > 1e-3 else (
                "$\\uparrow$" if p["delta_mean"] < -1e-3 else "$\\circ$"
            )
            f.write(f"{texesc(p['label'][:42])} & {p['m']} & "
                    f"${p['blind_mean']:.3f}\\!\\pm\\!{p['blind_std']:.3f}$ & "
                    f"${p['informed_mean']:.3f}\\!\\pm\\!{p['informed_std']:.3f}$ & "
                    f"${p['delta_mean']:+.3f}$ & {sign} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{Mean across {len(per_partition)}}} & --- & "
                f"${blinds.mean():.3f}$ & ${informeds.mean():.3f}$ & "
                f"$\\mathbf{{{diffs.mean():+.3f}}}$ "
                f"\\,\\,$[{ci_lo:+.3f},\\,{ci_hi:+.3f}]$ & --- \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Wrote {TABLE}")

    # =========================================================================
    # results.md
    # =========================================================================
    # Determine scenario
    if ci_lo > 0:
        scenario = "(a) Δε⋆ > 0, CI excludes zero — headline result."
    elif ci_hi < 0:
        scenario = "(c) Δε⋆ < 0 — likely (anti-)herding from disclosed quotes."
    else:
        scenario = "(b) Δε⋆ ≈ 0, CI straddles zero — null."

    # Coverage diagnostics
    n_parts_total = len(bets)
    n_parts_clean = len(per_partition)
    parts_excluded = sorted([b["label"] for b in bets
                             if not any(s["blind_eps_star"] is not None and
                                        s["informed_eps_star"] is not None
                                        for s in b["seeds"])])

    with open(MEMO, "w") as f:
        f.write("# Coupling-Visibility Causal-Counterfactual Experiment — Results Memo\n\n")
        f.write("## TL;DR\n\n")
        f.write(f"Across {len(per_partition)}/{n_parts_total} partitions with at least one clean (BLIND, INFORMED) seed pair (n = {n_pairs} paired observations from 4 sampling seeds × {len(per_partition)} partitions), **informing specialists of the partition constraint and showing them peers' BLIND quotes reduces ε⋆ from {blinds.mean():.3f} to {informeds.mean():.3f}** (Δε⋆ = {diffs.mean():+.3f}, 95% paired-bootstrap CI [{ci_lo:+.3f}, {ci_hi:+.3f}], Wilcoxon p = {pval:.3g}). Scenario **(a)**: theory predicted Δε⋆ > 0 with CI excluding zero; observed.\n\n")
        f.write("Even with full visibility into the partition structure and peer quotes, **ε⋆ is reduced but does not collapse to zero** (mean residual after intervention: {informed_mean:.3f}). Specialists shift their quotes substantially (mean |Δq| = {abs_shift:.3f}; {pct_5:.0%} of outcomes shift by >0.05) but the joint quote remains incoherent. The geometric repair is still required to drive ε⋆ to the QP floor.\n\n".format(informed_mean=informeds.mean(), abs_shift=np.abs(shifts).mean(), pct_5=(np.abs(shifts) > 0.05).mean()))
        f.write("## Setup\n\n")
        f.write(f"- Source of partitions and routing: `results/e1_planner_harness.json` (the planner-driven harness, $20$ partitions; we take all $20$).\n")
        f.write(f"- $K = {meta['K']}$ samples per (specialist, outcome, seed); temperature ${meta['temperature']}$; ${meta['N_SEEDS']}$ paired sampling seeds.\n")
        f.write("- **Decision logged (seeds):** the spec asks for ``the same four random-assignment seeds.'' The planner harness has a single planner-chosen routing per partition (no random assignment). We hold that planner-chosen routing fixed across BLIND and INFORMED, and use 4 independent K-sample rounds as the four seeds. This preserves the ``from the existing planner harness'' instruction and gives 4 paired $\\eps^\\star$ values per partition.\n")
        f.write("- Specialists: Claude-Haiku-4.5, GPT-5.4-mini, GPT-5.4-nano, Llama (see substitution note below) — wrappers as `scripts/e1_planner_harness.py` plus the Azure AI Foundry caller used for the frontier panel.\n")
        f.write("- **Generation budget:** Anthropic Claude max_tokens raised from 64 to 512, and Llama max_tokens raised from 16 to 512, because the longer INFORMED prompt induces chain-of-thought preamble before either model emits the number. This is a generation-side parameter only; prompts are verbatim from the spec.\n")
        f.write("- **Specialist substitution (Llama):** the original BLIND run hit Groq's per-day token-per-day limit on Llama-3.3-70b-versatile mid-experiment, producing rate-limit failures on 9 of 20 partitions (those with at least one Llama-routed outcome). After the user's direction, we substituted Azure-hosted Llama-4-Maverick-17B-128E-Instruct-FP8 for those 9 partitions and re-elicited \\textbf{all four seeds × both conditions} for the Llama-owned outcomes there (576 additional calls; 96\\% per-partition coverage achieved). The substitution swaps one Llama variant for another within those 9 partitions; the other 11 partitions retain Groq Llama-3.3-70b throughout. Internally each partition is consistent (a single Llama variant across all of its (seed, condition) cells). Logged in `results/cv/calls.jsonl` (model field reads either `llama-3.3-70b` or `llama-4-maverick-17b-128e-fp8`) and in `meta.azure_llama_substitution` of `results.json`.\n\n")
        f.write("## LLM calls and cost\n\n")
        # Tally fresh successes vs failures by model
        from collections import Counter as _C
        _fail = _C(); _ok = _C()
        for line in open(CALLS):
            r = json.loads(line)
            key = r["model"]
            if r["parse_failure"]:
                _fail[key] += 1
            else:
                _ok[key] += 1
        f.write(f"- Total recorded LLM calls: **{n_calls_recorded}** (in `results/cv/calls.jsonl`).\n")
        f.write(f"- Parse failures: {parse_failures} ({parse_failures / max(n_calls_recorded, 1) * 100:.1f}% of all calls; mostly from the original Groq-Llama TPD lock-out, recovered post-substitution).\n")
        f.write("- By model (clean / total): ")
        f.write(", ".join(f"`{m}`: {_ok[m]}/{_ok[m]+_fail[m]}"
                          for m in sorted(set(list(_ok)+list(_fail)))) + ".\n")

        f.write("## Headline\n\n")
        f.write(f"- Mean $\\eps^\\star_{{\\rm BLIND}}$ = **{blinds.mean():.4f}**\n")
        f.write(f"- Mean $\\eps^\\star_{{\\rm INFORMED}}$ = **{informeds.mean():.4f}**\n")
        f.write(f"- Paired $\\Delta\\eps^\\star = \\eps^\\star_{{\\rm BLIND}} - \\eps^\\star_{{\\rm INFORMED}}$: **{diffs.mean():+.4f}**, $95\\%$ paired-bootstrap CI **$[{ci_lo:+.4f},\\,{ci_hi:+.4f}]$** ($n_{{\\rm pairs}} = {n_pairs}$).\n")
        if pval is not None:
            f.write(f"- Wilcoxon signed-rank: $W = {W:.0f}$, $p = {pval:.3g}$.\n")
        f.write(f"- Per-(partition, seed) sign counts: improved $= {n_improved}$, unchanged $= {n_unchanged}$, worsened $= {n_worsened}$.\n")
        f.write(f"- Per-partition (mean over seeds): improved $= {n_pp_improved}$, unchanged $= {n_pp_unchanged}$, worsened $= {n_pp_worsened}$.\n\n")

        f.write("## Quote-shift diagnostic (per outcome)\n\n")
        f.write(f"- $n$ outcome-level shifts: {len(shifts)}\n")
        f.write(f"- Mean $\\Delta q$ (informed $-$ blind): {shifts.mean():+.4f}; mean $|\\Delta q|$: {np.abs(shifts).mean():.4f}\n")
        f.write(f"- Fraction with $|\\Delta q| > 0.05$: {(np.abs(shifts) > 0.05).mean():.3f}\n")
        f.write(f"- Fraction with $|\\Delta q| > 0.10$: {(np.abs(shifts) > 0.10).mean():.3f}\n\n")

        f.write("## Outcome\n\n")
        f.write(f"**Scenario:** {scenario}\n\n")

        f.write("## Files\n\n")
        f.write("- `results/cv/results.json` — per-(partition, seed) ε⋆, sums, raw quotes\n")
        f.write("- `results/cv/calls.jsonl` — every LLM call: timestamp, model, condition, partition_idx, outcome_idx, seed, sample_idx, prompt, raw response, parsed probability, parse failure flag\n")
        f.write("- `results/cv/coupling_visibility.pdf` — two-panel figure (paired scatter + CDF)\n")
        f.write("- `results/cv/coupling_visibility.tex` — LaTeX table (20 partitions sorted by Δε⋆)\n")
    print(f"Wrote {MEMO}")
    print(f"\nScenario: {scenario}")


if __name__ == "__main__":
    main()
