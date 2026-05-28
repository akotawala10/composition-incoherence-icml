"""
Context-sharing ablation on the 100-partition planner-style benchmark.

3-LLM panel (Anthropic Claude-Haiku-4.5, GPT-5.4-mini, GPT-5.4-nano).
Groq Llama excluded due to repeated TPD-quota issues. All three
conditions are re-run on this panel from scratch so isolated/listed/full
share the same assignment per partition.

Conditions:
  - isolated:  specialist sees ONLY its assigned outcome
  - listed:    specialist sees its outcome + an unordered numbered
               list of all outcomes (no normalization instruction)
  - full:      specialist sees its outcome + the full partition list
               + an explicit "probabilities must sum to 1" instruction

Each condition: same panel, K=8 verbalized samples, temperature 0.7,
same master seed.

Output: results/context_ablation_results.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JCD_ROOT = Path(os.environ.get("JCD_ROOT", str(Path(__file__).resolve().parent.parent.parent / "JCD-Forecasting")))
sys.path.insert(0, str(JCD_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(JCD_ROOT / ".env")

# Re-import the partition list, the AzureGPT54Client subclass, and the
# helper functions from real_agent_case_study.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_agent_case_study import (
    PARTITIONS, project_partition, make_question,
    K, TEMPERATURE, SEED, AzureGPT54Client,
)
from jcd.eval.sample import parse_verbalized_probability, AnthropicClient  # noqa: F401


def build_specialists_3llm() -> list:
    """3-LLM panel (Anthropic Claude-Haiku-4.5, Azure GPT mini, GPT nano).
    Groq Llama is excluded due to repeated TPD-quota issues."""
    return [
        AnthropicClient(model="claude-haiku-4-5-20251001"),
        AzureGPT54Client(deployment_env="AZURE_OPENAI_DEPLOYMENT_MINI"),
        AzureGPT54Client(deployment_env="AZURE_OPENAI_DEPLOYMENT_NANO"),
    ]


SHORT_NAMES_3 = ["Claude-Haiku", "GPT-5.4-mini", "GPT-5.4-nano"]

# ----------------------------------------------------------------------
# Three prompt templates --- the only thing that changes across conditions.
# All three share the {title}, {body}, {resolution_date} placeholders so
# we can reuse the existing client.prompt_template machinery.
# ----------------------------------------------------------------------

PROMPT_LISTED = (
    "You are a probabilistic forecaster. Below is a list of multiple "
    "candidate outcomes, exactly one of which will eventually be true. "
    "Your job is to estimate the probability of the SPECIFIC outcome "
    "marked below; the other outcomes are shown only to give you context.\n\n"
    "All candidate outcomes:\n{partition_list}\n\n"
    "Specific outcome to estimate (resolution date {resolution_date}):\n"
    "  {body}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)

PROMPT_FULL = (
    "You are a probabilistic forecaster. Below is a partition of "
    "{m} candidate outcomes: exactly ONE of these outcomes will be true "
    "(they are mutually exclusive and collectively exhaustive), so the "
    "probabilities you would assign to all {m} outcomes MUST sum to 1.\n\n"
    "All candidate outcomes:\n{partition_list}\n\n"
    "Estimate the probability of THIS outcome (resolution date "
    "{resolution_date}), bearing in mind that you should leave probability "
    "mass for the other {m_minus_1} outcomes:\n"
    "  {body}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.18). "
    "No words, no percent signs, no commentary."
)


def format_partition_list(outcomes: list[str]) -> str:
    return "\n".join(f"  ({chr(ord('a') + i)}) {o}" for i, o in enumerate(outcomes))


PROMPT_ISOLATED = (
    "You are a probabilistic forecaster. Provide your best estimate of the "
    "probability that the following question resolves YES. The probability "
    "must be a single number between 0 and 1.\n\n"
    "Question: {title}\n"
    "Resolution criteria: {body}\n"
    "Resolution date: {resolution_date}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)


def run_condition(specialists, partition: dict, assign: list[int],
                  condition: str) -> dict:
    """Run one condition (isolated | listed | full) on a single partition.

    Returns dict with per_outcome_means, per_outcome_samples, sum, eps_star.
    """
    outcomes = partition["outcomes"]
    m = len(outcomes)
    plist = format_partition_list(outcomes)

    # Build the appropriate prompt template
    if condition == "isolated":
        template = PROMPT_ISOLATED
    elif condition == "listed":
        template = PROMPT_LISTED
    elif condition == "full":
        template = PROMPT_FULL
    else:
        raise ValueError(condition)

    # Specialize the placeholders that don't vary per outcome
    template_specialized = template.replace("{partition_list}", plist) \
                                   .replace("{m}", str(m)) \
                                   .replace("{m_minus_1}", str(m - 1))

    per_outcome_means = []
    per_outcome_samples = []
    for j, outcome_text in enumerate(outcomes):
        sp = specialists[assign[j]]
        # Override prompt_template for this client. Save and restore.
        saved = getattr(sp, "prompt_template", None)
        sp.prompt_template = template_specialized
        try:
            q = make_question(
                qid=f"{partition['label']}::outcome{j}::{condition}",
                outcome_text=outcome_text,
                date=partition["date"],
            )
            samples = sp.forecast(q, K, temperature=TEMPERATURE)
        finally:
            if saved is not None:
                sp.prompt_template = saved
        if len(samples) == 0:
            mean = 0.5
        else:
            mean = float(np.mean(samples))
        per_outcome_means.append(mean)
        per_outcome_samples.append(samples.tolist() if hasattr(samples, "tolist") else list(samples))

    p = np.array(per_outcome_means, dtype=float)
    proj, eps = project_partition(p)
    sum_violation = float(abs(p.sum() - 1.0))
    return dict(
        per_outcome_means=per_outcome_means,
        per_outcome_samples=per_outcome_samples,
        sum=float(p.sum()),
        sum_violation=sum_violation,
        eps_star=eps,
        projected=proj.tolist(),
    )


def main() -> None:
    # 3-LLM panel.  Re-derive assignments fresh.
    rng = random.Random(SEED)
    specialists = build_specialists_3llm()
    short_names = SHORT_NAMES_3
    print(f"3-LLM panel: {short_names}")

    out_path = Path(__file__).resolve().parent.parent / "data" / "results" / "context_ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined = []
    for i, partition in enumerate(PARTITIONS):
        m = len(partition["outcomes"])
        if m <= len(specialists):
            assign = rng.sample(range(len(specialists)), m)
        else:
            assign = [rng.randrange(len(specialists)) for _ in range(m)]

        rec = dict(
            label=partition["label"],
            outcomes=partition["outcomes"],
            assigned_specialists=[short_names[a] for a in assign],
        )

        for condition in ("isolated", "listed", "full"):
            print(f"\n[{i+1}/{len(PARTITIONS)}] {partition['label']:60s}  ({condition})", flush=True)
            cond_rec = run_condition(specialists, partition, assign, condition)
            rec[condition] = cond_rec
            print(f"  sum = {cond_rec['sum']:.3f}   eps* = {cond_rec['eps_star']:.4f}", flush=True)
        combined.append(rec)

        # incremental save every 5 partitions (more frequent given 3 cond/clique)
        if (i + 1) % 5 == 0:
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"  -> checkpoint saved ({i+1}/{len(PARTITIONS)}) to {out_path}", flush=True)

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {len(combined)} partitions to {out_path}")

    # quick aggregate
    print("\n" + "=" * 76)
    print("CONTEXT-SHARING ABLATION SUMMARY")
    print("=" * 76)
    print(f"{'condition':<14}{'<eps*>':>10}{'<|sum-1|>':>14}{'frac eps>1e-3':>18}")
    for cond in ("isolated", "listed", "full"):
        eps = np.array([r[cond]["eps_star"] for r in combined])
        sumv = np.array([r[cond]["sum_violation"] for r in combined])
        print(f"{cond:<14}{eps.mean():>10.4f}{sumv.mean():>14.4f}{(eps > 1e-3).mean():>18.3f}")


if __name__ == "__main__":
    main()
