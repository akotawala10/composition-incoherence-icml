"""
Real-agent harness with planner-driven routing + planner-emitted
specialist prompts.

The existing real_agent_results.json experiment hard-codes per-outcome
single-question prompts. This harness introduces a planner LLM that,
for each partition, decides:
  (a) which specialist handles each outcome (routing)
  (b) what context to give each specialist (the planner can choose to
      mention the partition, the other outcomes, or anything else)

Specialists then run with the planner-emitted prompts. We measure
eps_star on the assembled quote and compare to:
  - the published baseline (real_agent_results.json) -- single-outcome
    prompts, random assignment.
  - the prompt-ablation result -- single-outcome prompts +
    hard-coded partition-aware boilerplate.

If eps_star remains positive on a meaningful fraction of cases
even with the planner doing its best, the failure mode is real for
deployed agents.
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass
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

logging.basicConfig(level=logging.WARNING)

K = 8
TEMPERATURE = 0.7
N_PARTITIONS = 20
HARD_CALL_CAP = 1000
OUT = REPO_ROOT / "results" / "e1_planner_harness.json"
BASELINE = REPO_ROOT / "results" / "real_agent_results.json"

SPECIALIST_ROSTER = [
    "Claude-Haiku", "GPT-5.4-mini", "GPT-5.4-nano", "Llama-3.3-70b",
]

# A specialist runs against an arbitrary prompt with substitution slots
# {context} and {outcome}. The planner controls {context}.
SPECIALIST_PROMPT = (
    "You are a probabilistic forecaster. The following context was "
    "provided to you by the planning agent that routed this question:\n\n"
    "---\n{context}\n---\n\n"
    "Question: {outcome}\n"
    "Resolution date: {resolution_date}\n\n"
    "Provide ONLY your probability (a single number between 0 and 1) that "
    "the question resolves YES. No words, no percent signs, no commentary."
)

# Planner prompt: emits a JSON object with assignment + context per outcome.
PLANNER_PROMPT = """You are the planner for a multi-specialist forecasting agent. You decompose a multi-candidate forecasting partition into per-outcome subqueries and route each to one of four specialist LLMs. The combined per-outcome marginals will be aggregated into a partition belief, which a downstream decision-maker will treat as a coherent probability distribution over the outcomes.

Your job for each outcome:
  (a) Pick the specialist who is most likely to be calibrated on it.
     Available specialists: {specialists}
     (You may use the same specialist for multiple outcomes.)
  (b) Write a short CONTEXT BLOCK (1-3 sentences) that the specialist
     should see before quoting. You may include any information you think
     the specialist needs to be calibrated and to be coherent with its
     siblings, including the partition structure if you judge that useful.

Partition label: {partition_label}
Resolution date: {resolution_date}
Outcomes (the partition will resolve with EXACTLY ONE of these as YES):
{enumerated_outcomes}

Output ONLY a JSON object with the schema:
{{
  "assignments": [
    {{"outcome_index": 0, "specialist": "<one of {specialists}>", "context": "<short context block>"}},
    ...
  ]
}}
There must be one assignment per outcome, in order. Output ONLY the JSON, no commentary, no markdown fence.
"""


def parse_planner_json(text: str, n_outcomes: int) -> list[dict] | None:
    """Lenient parser: extract the JSON object even if wrapped in
    markdown fences or trailing text."""
    if not text:
        return None
    # Strip optional markdown fence.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Find the first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    asgs = d.get("assignments")
    if not isinstance(asgs, list) or len(asgs) != n_outcomes:
        return None
    out = []
    for i, a in enumerate(asgs):
        if not isinstance(a, dict):
            return None
        sp = a.get("specialist")
        ctx = a.get("context", "")
        if sp not in SPECIALIST_ROSTER:
            return None
        out.append(dict(specialist=sp, context=str(ctx)))
    return out


# -------- Custom planner / specialist clients ---------

@dataclass
class PlannerClient:
    """Anthropic Claude Haiku as planner. Returns parsed assignments."""
    model: str = "claude-haiku-4-5-20251001"

    def __post_init__(self) -> None:
        from anthropic import Anthropic
        self._client = Anthropic()

    def plan(self, partition: dict) -> tuple[list[dict] | None, str]:
        outcomes = partition["outcomes"]
        enumerated = "\n".join(
            f"  {i}. {o}" for i, o in enumerate(outcomes)
        )
        prompt = PLANNER_PROMPT.format(
            specialists=", ".join(SPECIALIST_ROSTER),
            partition_label=partition["label"],
            resolution_date=partition.get("date") or "unspecified",
            enumerated_outcomes=enumerated,
        )
        for attempt in range(2):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    blk.text for blk in resp.content
                    if getattr(blk, "type", "") == "text"
                )
                parsed = parse_planner_json(text, len(outcomes))
                if parsed is not None:
                    return parsed, text
            except Exception as e:
                logging.warning("Planner attempt %d failed: %s", attempt + 1, e)
        return None, ""


def make_specialist_clients() -> dict:
    """Return forecast(prompt_str, K) callables for each specialist."""
    from anthropic import Anthropic
    from openai import OpenAI
    import os

    anth = Anthropic()

    def call_anthropic(prompt: str) -> float | None:
        try:
            resp = anth.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                blk.text for blk in resp.content
                if getattr(blk, "type", "") == "text"
            )
            return parse_verbalized_probability(text)
        except Exception as e:
            logging.warning("Anthropic call failed: %s", e)
            return None

    # Azure OpenAI client via openai SDK for GPT-5.4 mini/nano.
    from openai import AzureOpenAI
    azure = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    def call_azure(deployment: str):
        def fn(prompt: str) -> float | None:
            try:
                resp = azure.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_completion_tokens=64,
                )
                text = resp.choices[0].message.content or ""
                return parse_verbalized_probability(text)
            except Exception as e:
                logging.warning("Azure (%s) call failed: %s", deployment, e)
                return None
        return fn

    # Groq via OpenAI-compatible.
    groq = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    def call_groq(prompt: str) -> float | None:
        try:
            resp = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=16,
            )
            text = resp.choices[0].message.content or ""
            return parse_verbalized_probability(text)
        except Exception as e:
            logging.warning("Groq call failed: %s", e)
            return None

    return {
        "Claude-Haiku": call_anthropic,
        "GPT-5.4-mini": call_azure(os.environ["AZURE_OPENAI_DEPLOYMENT_MINI"]),
        "GPT-5.4-nano": call_azure(os.environ["AZURE_OPENAI_DEPLOYMENT_NANO"]),
        "Llama-3.3-70b": call_groq,
    }


def project_partition(p: np.ndarray) -> tuple[np.ndarray, float]:
    m = p.size
    clique = Clique(
        m=m, relations=[Relation(type="partition", indices=tuple(range(m)))],
        p_hat=p,
    )
    proj = jcd_project(clique)
    return proj, float(np.linalg.norm(p - proj))


def main() -> None:
    with open(BASELINE) as f:
        baseline = json.load(f)
    partitions = baseline[:N_PARTITIONS]
    print(f"Planner-driven harness on first {len(partitions)} partitions.")

    planner = PlannerClient()
    specialists = make_specialist_clients()

    results = []
    total_calls = 0
    t0 = time.time()
    for pi, p_baseline in enumerate(partitions):
        outcomes = p_baseline["outcomes"]
        date = p_baseline.get("date") or "unspecified"
        partition = dict(label=p_baseline["label"], outcomes=outcomes, date=date)

        plan_calls = 1
        plan, plan_raw = planner.plan(partition)
        total_calls += plan_calls
        if plan is None:
            print(f"  [{pi+1}/{len(partitions)}] WARN planner failed; "
                  "falling back to round-robin + bare prompts.")
            plan = []
            for j, outcome in enumerate(outcomes):
                plan.append(dict(
                    specialist=SPECIALIST_ROSTER[j % len(SPECIALIST_ROSTER)],
                    context=("This question is one of several outcomes in a "
                             "partition forecasting question; please be "
                             "calibrated."),
                ))

        per_outcome_means = []
        per_outcome_samples = []
        per_outcome_specialist = [a["specialist"] for a in plan]
        per_outcome_context = [a["context"] for a in plan]
        for j, outcome in enumerate(outcomes):
            sp_name = plan[j]["specialist"]
            ctx = plan[j]["context"]
            prompt = SPECIALIST_PROMPT.format(
                context=ctx, outcome=outcome, resolution_date=date,
            )
            samples: list[float] = []
            for k in range(K):
                if total_calls >= HARD_CALL_CAP:
                    break
                p = specialists[sp_name](prompt)
                total_calls += 1
                if p is not None:
                    samples.append(float(p))
            if samples:
                per_outcome_means.append(float(np.mean(samples)))
            else:
                per_outcome_means.append(0.5)
            per_outcome_samples.append(samples)
            if total_calls >= HARD_CALL_CAP:
                break
        if total_calls >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP {HARD_CALL_CAP} reached; stopping at partition {pi}.")
            break

        p_arr = np.array(per_outcome_means, dtype=float)
        proj, eps = project_partition(p_arr)
        sum_v = abs(p_arr.sum() - 1.0)

        results.append(dict(
            label=p_baseline["label"],
            outcomes=outcomes,
            assigned_specialists=per_outcome_specialist,
            per_outcome_context=per_outcome_context,
            per_outcome_means=per_outcome_means,
            per_outcome_samples=per_outcome_samples,
            sum=float(p_arr.sum()),
            sum_violation=float(sum_v),
            eps_star=float(eps),
            projected=proj.tolist(),
            baseline_eps_star=p_baseline["eps_star"],
            baseline_sum=p_baseline["sum"],
            planner_raw=plan_raw,
        ))
        elapsed = time.time() - t0
        print(
            f"  [{pi+1}/{len(partitions)}] {p_baseline['label'][:50]:<50s}  "
            f"baseline eps*={p_baseline['eps_star']:.3f}  -> "
            f"planner eps*={eps:.3f}  delta={eps - p_baseline['eps_star']:+.3f}  "
            f"calls={total_calls}  ({elapsed:.0f}s)"
        )

    OUT.parent.mkdir(exist_ok=True)
    if results:
        deltas = [r["eps_star"] - r["baseline_eps_star"] for r in results]
        summary = dict(
            n_partitions=len(results),
            total_calls=total_calls,
            baseline_mean_eps=float(np.mean([r["baseline_eps_star"] for r in results])),
            planner_mean_eps=float(np.mean([r["eps_star"] for r in results])),
            delta_mean=float(np.mean(deltas)),
            delta_median=float(np.median(deltas)),
            n_better=int(sum(1 for d in deltas if d < -1e-6)),
            n_worse=int(sum(1 for d in deltas if d > 1e-6)),
            n_unchanged=int(sum(1 for d in deltas if abs(d) <= 1e-6)),
            n_eps_star_pos=int(sum(1 for r in results if r["eps_star"] > 1e-6)),
        )
    else:
        summary = dict(n_partitions=0, total_calls=total_calls)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2)

    print("\n=== Planner-harness summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
