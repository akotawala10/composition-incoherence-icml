"""
Tool-augmented specialist routing pilot (Option 2).

Setup. For each partition question, each specialist LLM:
  (i)   issues a single web-search query (DDG, no API key) for its assigned
        outcome text;
  (ii)  formats the top-N retrieved snippets as context;
  (iii) emits a single Bernoulli probability under that context.

The specialist still answers exactly ONE Bernoulli (its assigned outcome),
matching the no-tool baseline. The novel quantity is whether grounding each
specialist with retrieval reduces the COMPOSITIONAL residual eps^star or
whether cross-specialist disagreement persists.

We hold the assignment seed and partition list constant with the no-tool
baseline so per-partition eps^star is directly comparable.

Outputs:
  tool_augmented_results.json     -- per-partition raw data
  tool_vs_notool_comparison.tex   -- LaTeX summary table
  figures/tool_vs_notool.pdf      -- per-partition eps^star comparison plot
"""

from __future__ import annotations
import json, os, sys, random, time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT.parent.parent / ".env")

from jcd.eval.sample import (
    AnthropicClient, AzureOpenAIClient, GroqClient,
    parse_verbalized_probability, DEFAULT_PROMPT,
)
from jcd.data.paleka import PalekaQuestion
from jcd.types import Clique, Relation
from jcd.qp.solver import project as jcd_project

# Re-import the partition list and specialists from the no-tool case study.
sys.path.insert(0, str(ROOT / "scripts"))
from real_agent_case_study import (
    PARTITIONS, build_specialists, project_partition, K, SEED,
)

from ddgs import DDGS

# Hold the same partition slice as the no-tool baseline's first 30 entries
# (matches the original N=30 pilot scope) so the A/B is apples-to-apples.
SUBSET_N = 30
TOOL_SAMPLES_K = 8


PROMPT_WITH_CONTEXT = (
    "You are a probabilistic forecaster. Below is the question, followed "
    "by web-search snippets retrieved at decision time. Use the snippets to "
    "ground your estimate; if they conflict or are sparse, weigh them with "
    "your own judgment.\n\n"
    "Question: {title}\n"
    "Resolution criteria: {body}\n"
    "Resolution date: {resolution_date}\n\n"
    "=== Web-search snippets ===\n"
    "{context}\n"
    "=== End snippets ===\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)


def web_search(query: str, max_results: int = 5, max_retries: int = 3) -> str:
    """Return formatted search snippets for ``query``."""
    for attempt in range(max_retries):
        try:
            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                return "(no search results returned)"
            lines = []
            for i, r in enumerate(results, 1):
                title = (r.get("title") or "").strip()[:120]
                body = (r.get("body") or "").strip().replace("\n", " ")[:300]
                lines.append(f"[{i}] {title}\n    {body}")
            return "\n".join(lines)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return f"(search error after {max_retries} attempts: {e})"


def forecast_with_context(client, question, context: str, *, K: int = TOOL_SAMPLES_K,
                           temperature: float = 0.7) -> np.ndarray:
    """Reuse the client.forecast_one machinery but with a context-augmented prompt.

    We monkey-swap the prompt_template for the duration of this call.
    """
    original = client.prompt_template
    client.prompt_template = PROMPT_WITH_CONTEXT.replace(
        "{context}", context.replace("{", "{{").replace("}", "}}")
    )
    try:
        out: list[float] = []
        for _ in range(K):
            p = client.forecast_one(question, temperature=temperature)
            if p is not None:
                out.append(p)
        return np.asarray(out, dtype=float)
    finally:
        client.prompt_template = original


def make_question(qid: str, outcome_text: str, date: str) -> PalekaQuestion:
    return PalekaQuestion(
        id=qid,
        title=outcome_text,
        body=outcome_text,
        resolution_date=date,
        question_type="binary",
        data_source="tool_augmented_case_study",
        url=None,
        resolution=None,
    )


def main() -> None:
    rng = random.Random(SEED)
    specialists = build_specialists()
    short_names = ["Claude-Haiku", "GPT-5.4-mini", "GPT-5.4-nano", "Llama-3.3-70b"]

    subset = PARTITIONS[:SUBSET_N]
    print(f"Running tool-augmented pilot on first {len(subset)} partitions "
          f"(matches original N=30 pilot scope).\n")

    results = []
    for partition_idx, partition in enumerate(subset):
        m = len(partition["outcomes"])
        # Replay the SAME assignment used by the no-tool run.
        if m <= len(specialists):
            assign = rng.sample(range(len(specialists)), m)
        else:
            assign = [rng.randrange(len(specialists)) for _ in range(m)]

        per_outcome_means = []
        per_outcome_specialist = []
        per_outcome_search_query = []
        per_outcome_context_chars = []
        for j, outcome_text in enumerate(partition["outcomes"]):
            sp_idx = assign[j]
            sp = specialists[sp_idx]
            search_query = outcome_text[:200]
            context = web_search(search_query, max_results=5)
            q = make_question(
                qid=f"{partition['label']}::outcome{j}",
                outcome_text=f"What is the probability that {outcome_text}?",
                date=partition["date"],
            )
            samples = forecast_with_context(sp, q, context, K=TOOL_SAMPLES_K)
            mean = 0.5 if len(samples) == 0 else float(np.mean(samples))
            per_outcome_means.append(mean)
            per_outcome_specialist.append(short_names[sp_idx])
            per_outcome_search_query.append(search_query)
            per_outcome_context_chars.append(len(context))
            print(f"  [{partition_idx+1:>3d}/{len(subset)}] "
                  f"{short_names[sp_idx]:<14s} | "
                  f"out{j} | K={len(samples)}/{TOOL_SAMPLES_K} | "
                  f"|ctx|={len(context)} | mean={mean:.3f}")

        p = np.array(per_outcome_means, dtype=float)
        proj, eps = project_partition(p)
        sum_violation = abs(p.sum() - 1.0)

        print(f"  partition: {partition['label']}")
        print(f"    raw quote = {[round(x,3) for x in p.tolist()]}")
        print(f"    sum       = {p.sum():.3f}")
        print(f"    eps^*     = {eps:.4f}\n")

        results.append(dict(
            label=partition["label"],
            outcomes=partition["outcomes"],
            assigned_specialists=per_outcome_specialist,
            per_outcome_means=per_outcome_means,
            per_outcome_search_queries=per_outcome_search_query,
            per_outcome_context_chars=per_outcome_context_chars,
            sum=p.sum(),
            sum_violation=sum_violation,
            eps_star=eps,
            projected=proj.tolist(),
        ))

    # Save raw
    out = ROOT / "tool_augmented_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")

    # Compare to no-tool baseline (same first 30)
    with open(ROOT / "real_agent_results.json") as f:
        baseline = json.load(f)
    baseline_first = baseline[:SUBSET_N]
    label_to_baseline = {r["label"]: r for r in baseline_first}

    print("\n" + "=" * 80)
    print("A/B COMPARISON: no-tool vs tool-augmented (first {} partitions)".format(SUBSET_N))
    print("=" * 80)
    print(f"{'partition':<48s}{'eps* (no tool)':>17s}{'eps* (tool)':>14s}{'delta':>10s}")
    print("-" * 90)
    deltas = []
    for r in results:
        b = label_to_baseline.get(r["label"])
        if b is None:
            continue
        d = r["eps_star"] - b["eps_star"]
        deltas.append(d)
        print(f"  {r['label'][:46]:<48s}{b['eps_star']:>17.4f}{r['eps_star']:>14.4f}"
              f"{d:>10.4f}")
    deltas = np.array(deltas)
    eps_tool = np.array([r["eps_star"] for r in results])
    eps_base = np.array([label_to_baseline[r["label"]]["eps_star"] for r in results
                         if r["label"] in label_to_baseline])
    print("-" * 90)
    print(f"  mean eps^* (no tool)  : {eps_base.mean():.4f}")
    print(f"  mean eps^* (tool)     : {eps_tool.mean():.4f}")
    print(f"  mean delta            : {deltas.mean():.4f}")
    print(f"  fraction tool > base  : {float((deltas > 0).mean()):.4f}")
    print(f"  fraction tool < base  : {float((deltas < 0).mean()):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(eps_base, eps_tool, s=24, alpha=0.7, color="#2ca02c", edgecolor="k", lw=0.4)
    lim = max(eps_base.max(), eps_tool.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=0.8, label="$y=x$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\varepsilon^\star$, no-tool baseline")
    ax.set_ylabel(r"$\varepsilon^\star$, tool-augmented")
    ax.set_title(f"Compositional residual: tool-augmented vs no-tool\n"
                 f"(N={len(eps_base)} partitions; mean eps* "
                 f"{eps_base.mean():.3f} -> {eps_tool.mean():.3f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/tool_vs_notool.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figures/tool_vs_notool.pdf")


if __name__ == "__main__":
    main()
