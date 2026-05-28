"""
Constrained-generation prompt ablation.

Tests whether telling each specialist that it is part of a partition
reduces the compositional residual eps_star.

Protocol:
  - Reuse the FIRST 30 partitions from real_agent_results.json so the
    A/B is matched to the published baseline.
  - Reuse the same per-outcome specialist assignment from that file
    (so the only thing that changes is the prompt text).
  - For each (partition, outcome) call the assigned specialist with a
    NEW prompt that includes the partition label and the FULL list of
    sibling outcomes, with explicit instructions that the agent's
    quote will be combined with sibling quotes into a partition.
  - Sample K=8 verbalized probabilities; compute composed quote;
    project; compute eps_star.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402
from jcd.eval.sample import (  # noqa: E402
    AnthropicClient, AzureOpenAIClient, GroqClient,
    parse_verbalized_probability,
)
from jcd.data.paleka import PalekaQuestion  # noqa: E402

logging.basicConfig(level=logging.WARNING)

K = 8
TEMPERATURE = 0.7
MAX_PARTITIONS = 30
HARD_CALL_CAP = 1500
OUT = REPO_ROOT / "results" / "e5_prompt_ablation.json"
BASELINE = REPO_ROOT / "results" / "real_agent_results.json"


# Custom partition-aware prompt template. Specialist sees the full list
# of sibling outcomes and is told its quote will be combined into a
# partition -- but is still asked only for the marginal probability of
# its own assigned outcome (not a joint assessment).
PARTITION_AWARE_PROMPT = (
    "You are a probabilistic forecaster. You are one of several specialists "
    "answering questions in a forecasting partition: a multi-candidate event "
    "where exactly ONE of the listed outcomes will resolve YES and the rest "
    "will resolve NO. Your assigned outcome is one element of this partition. "
    "Your colleagues are independently quoting marginal probabilities for the "
    "other outcomes; the joint quoted probabilities will be combined into a "
    "partition belief.\n\n"
    "Partition label: {partition_label}\n"
    "All sibling outcomes (one of which will resolve YES):\n"
    "{enumerated_siblings}\n\n"
    "Your assigned outcome:\n"
    "{title}\n\n"
    "Resolution date: {resolution_date}\n\n"
    "Provide ONLY your marginal probability that the assigned outcome resolves "
    "YES, as a single number between 0 and 1. Account for the partition "
    "structure when calibrating (the marginals across all outcomes will sum "
    "to 1 in any coherent joint belief).\n"
    "No words, no percent signs, no commentary. Output ONLY the number."
)


# AzureOpenAIClient subclass for GPT-5.4-mini/-nano (max_completion_tokens).
from dataclasses import dataclass


@dataclass
class AzureGPT54Client(AzureOpenAIClient):
    def forecast_one(self, question, *, temperature=0.7, seed=None):
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
            partition_label=getattr(question, "_partition_label", ""),
            enumerated_siblings=getattr(question, "_enumerated_siblings", ""),
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=64,
                )
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:
                logging.warning(
                    "Azure (%s) request failed (attempt %d): %s",
                    self._deployment, attempt + 1, e
                )
        return None


def make_specialists() -> dict:
    """Map short_name -> client. The 4 specialists used in real_agent_case_study.py."""
    return {
        "Claude-Haiku": AnthropicClient(
            model="claude-haiku-4-5-20251001",
            prompt_template=PARTITION_AWARE_PROMPT,
        ),
        "GPT-5.4-mini": AzureGPT54Client(
            deployment_env="AZURE_OPENAI_DEPLOYMENT_MINI",
            prompt_template=PARTITION_AWARE_PROMPT,
        ),
        "GPT-5.4-nano": AzureGPT54Client(
            deployment_env="AZURE_OPENAI_DEPLOYMENT_NANO",
            prompt_template=PARTITION_AWARE_PROMPT,
        ),
        "Llama-3.3-70b": GroqClient(
            model="llama-3.3-70b-versatile",
            api_key_env="GROQ_API_KEY",
            prompt_template=PARTITION_AWARE_PROMPT,
        ),
    }


class PartitionQuestion:
    """Lightweight question carrier (PalekaQuestion is frozen). The
    custom forecast_one closures below pull title/body/resolution_date
    plus the partition context fields off this object."""

    __slots__ = ("id", "title", "body", "resolution_date",
                 "_partition_label", "_enumerated_siblings")

    def __init__(self, qid, title, body, resolution_date,
                 partition_label, enumerated_siblings):
        self.id = qid
        self.title = title
        self.body = body
        self.resolution_date = resolution_date
        self._partition_label = partition_label
        self._enumerated_siblings = enumerated_siblings


def make_partition_question(partition: dict, outcome_idx: int) -> PartitionQuestion:
    outcome_text = partition["outcomes"][outcome_idx]
    enumerated = "\n".join(
        f"  {i+1}. {o}" for i, o in enumerate(partition["outcomes"])
    )
    return PartitionQuestion(
        qid=f"{partition['label']}::out{outcome_idx}",
        title=outcome_text,
        body=outcome_text,
        resolution_date=partition.get("date") or "unspecified",
        partition_label=partition["label"],
        enumerated_siblings=enumerated,
    )


def project_partition(p: np.ndarray) -> tuple[np.ndarray, float]:
    m = p.size
    clique = Clique(
        m=m, relations=[Relation(type="partition", indices=tuple(range(m)))],
        p_hat=p,
    )
    proj = jcd_project(clique)
    return proj, float(np.linalg.norm(p - proj))


def patch_anthropic_format(specialists):
    """Anthropic client uses prompt_template.format too -- its base class
    only passes (title, body, resolution_date). We wrap forecast_one to
    inject our partition context kwargs."""
    sp = specialists["Claude-Haiku"]
    original = sp.forecast_one

    def forecast_one_patched(question, *, temperature=0.7, seed=None):
        prompt = sp.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
            partition_label=getattr(question, "_partition_label", ""),
            enumerated_siblings=getattr(question, "_enumerated_siblings", ""),
        )
        for attempt in range(sp.max_retries + 1):
            try:
                resp = sp._client.messages.create(
                    model=sp.model,
                    max_tokens=64,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    blk.text for blk in resp.content if getattr(blk, "type", "") == "text"
                )
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:
                logging.warning(
                    "Anthropic request failed (attempt %d): %s", attempt + 1, e
                )
        return None

    sp.forecast_one = forecast_one_patched


def patch_groq_format(specialists):
    """Same trick for GroqClient (OpenAICompatibleClient default format
    only takes title/body/resolution_date)."""
    sp = specialists["Llama-3.3-70b"]

    def forecast_one_patched(question, *, temperature=0.7, seed=None):
        prompt = sp.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
            partition_label=getattr(question, "_partition_label", ""),
            enumerated_siblings=getattr(question, "_enumerated_siblings", ""),
        )
        for attempt in range(sp.max_retries + 1):
            try:
                resp = sp._client.chat.completions.create(
                    model=sp.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=16,
                )
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:
                logging.warning(
                    "Groq request failed (attempt %d): %s", attempt + 1, e
                )
        return None

    sp.forecast_one = forecast_one_patched


def main() -> None:
    with open(BASELINE) as f:
        baseline = json.load(f)
    partitions = baseline[:MAX_PARTITIONS]
    print(f"Re-running first {len(partitions)} baseline partitions with "
          f"partition-aware prompts.")

    specialists = make_specialists()
    patch_anthropic_format(specialists)
    patch_groq_format(specialists)

    results = []
    total_calls = 0
    t0 = time.time()
    for pi, p_baseline in enumerate(partitions):
        # Use the baseline's per-outcome assignment for fair A/B.
        assigned = p_baseline["assigned_specialists"]
        outcomes = p_baseline["outcomes"]
        m = len(outcomes)
        partition = dict(
            label=p_baseline["label"], outcomes=outcomes,
            date=p_baseline.get("date") or "unspecified",
        )
        per_outcome_means = []
        per_outcome_samples = []
        for j in range(m):
            sp_name = assigned[j]
            if sp_name not in specialists:
                print(f"  WARN: unknown specialist '{sp_name}' for partition {pi} outcome {j}")
                per_outcome_means.append(0.5)
                per_outcome_samples.append([])
                continue
            sp = specialists[sp_name]
            q = make_partition_question(partition, j)
            samples = sp.forecast(q, K, temperature=TEMPERATURE)
            total_calls += K
            if len(samples) == 0:
                per_outcome_means.append(0.5)
                per_outcome_samples.append([])
            else:
                per_outcome_means.append(float(np.mean(samples)))
                per_outcome_samples.append(samples.tolist())
            if total_calls >= HARD_CALL_CAP:
                print(f"  HARD_CALL_CAP {HARD_CALL_CAP} reached at partition {pi} outcome {j}; stopping.")
                break
        if total_calls >= HARD_CALL_CAP:
            break

        p_arr = np.array(per_outcome_means, dtype=float)
        proj, eps = project_partition(p_arr)
        sum_v = abs(p_arr.sum() - 1.0)
        results.append(dict(
            label=p_baseline["label"],
            outcomes=outcomes,
            assigned_specialists=assigned,
            per_outcome_means=per_outcome_means,
            per_outcome_samples=per_outcome_samples,
            sum=float(p_arr.sum()),
            sum_violation=float(sum_v),
            eps_star=float(eps),
            projected=proj.tolist(),
            baseline_eps_star=p_baseline["eps_star"],
            baseline_sum=p_baseline["sum"],
        ))
        elapsed = time.time() - t0
        print(
            f"  [{pi+1}/{len(partitions)}] {p_baseline['label'][:50]:<50s}  "
            f"baseline eps*={p_baseline['eps_star']:.3f}  -> "
            f"new eps*={eps:.3f}  delta={eps - p_baseline['eps_star']:+.3f}  "
            f"calls={total_calls}  ({elapsed:.0f}s)"
        )

    OUT.parent.mkdir(exist_ok=True)
    summary = dict(
        n_partitions=len(results),
        total_calls=total_calls,
        baseline_mean_eps=float(np.mean([r["baseline_eps_star"] for r in results])),
        new_mean_eps=float(np.mean([r["eps_star"] for r in results])),
        delta_mean=float(np.mean([r["eps_star"] - r["baseline_eps_star"] for r in results])),
        n_better=int(sum(1 for r in results if r["eps_star"] < r["baseline_eps_star"] - 1e-6)),
        n_worse=int(sum(1 for r in results if r["eps_star"] > r["baseline_eps_star"] + 1e-6)),
        n_unchanged=int(sum(1 for r in results
                            if abs(r["eps_star"] - r["baseline_eps_star"]) <= 1e-6)),
    )
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2)

    print("\n=== Prompt-ablation summary ===")
    print(f"n partitions: {summary['n_partitions']}")
    print(f"total LLM calls: {summary['total_calls']}")
    print(f"baseline mean eps*: {summary['baseline_mean_eps']:.4f}")
    print(f"partition-aware mean eps*: {summary['new_mean_eps']:.4f}")
    print(f"mean delta: {summary['delta_mean']:+.4f}")
    print(f"n better: {summary['n_better']}  worse: {summary['n_worse']}  "
          f"unchanged: {summary['n_unchanged']}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
