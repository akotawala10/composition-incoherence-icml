"""
Frontier-model spot-check.

Question: does the compositional failure mode persist when one
specialist is a frontier model rather than mid-tier?

Protocol. Re-run the planner-driven harness on the first 10
partitions from real_agent_results.json, with a modified specialist
roster that includes Claude-Opus-4.7 in place of Llama-3.3-70b.
Same planner (Claude-Haiku-4.5) and same prompt structure. Compare
eps_star to the planner-harness result on the same 10 partitions.

Result interpretation:
  - If eps_star drops dramatically with Opus in the panel, the
    failure mode is partly a capability issue and may be solved by
    scaling up.
  - If eps_star persists at similar levels, the failure mode is
    structural -- not a small-model artefact.
"""
from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402
from jcd.eval.sample import parse_verbalized_probability  # noqa: E402

logging.basicConfig(level=logging.WARNING)

K = 8
TEMPERATURE = 0.7
N_PARTITIONS = 10
HARD_CALL_CAP = 600
OUT = REPO_ROOT / "results" / "b1_frontier_spotcheck.json"
BASELINE = REPO_ROOT / "results" / "real_agent_results.json"
E1_RESULTS = REPO_ROOT / "results" / "e1_planner_harness.json"

# Specialist roster: Llama replaced by Claude-Opus-4.7.
SPECIALIST_ROSTER = [
    "Claude-Opus-4.7",  # frontier
    "Claude-Haiku-4.5",
    "GPT-5.4-mini",
    "GPT-5.4-nano",
]

SPECIALIST_PROMPT = (
    "You are a probabilistic forecaster. The following context was "
    "provided to you by the planning agent that routed this question:\n\n"
    "---\n{context}\n---\n\n"
    "Question: {outcome}\n"
    "Resolution date: {resolution_date}\n\n"
    "Provide ONLY your probability (a single number between 0 and 1) that "
    "the question resolves YES. No words, no percent signs, no commentary."
)

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
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
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
    for a in asgs:
        if not isinstance(a, dict):
            return None
        sp = a.get("specialist")
        ctx = a.get("context", "")
        if sp not in SPECIALIST_ROSTER:
            return None
        out.append(dict(specialist=sp, context=str(ctx)))
    return out


def make_callers() -> dict:
    from anthropic import Anthropic
    from openai import OpenAI, AzureOpenAI

    anth = Anthropic()
    azure = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    def call_anthropic_model(model_id: str):
        # Opus 4.7 has deprecated temperature; gate it.
        accepts_temperature = "opus" not in model_id.lower()
        def call(prompt: str) -> float | None:
            try:
                kwargs = dict(
                    model=model_id, max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                )
                if accepts_temperature:
                    kwargs["temperature"] = TEMPERATURE
                resp = anth.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                return parse_verbalized_probability(text)
            except Exception as e:
                logging.warning("Anthropic (%s) failed: %s", model_id, e)
                return None
        return call

    def call_azure(deployment: str):
        def call(prompt: str) -> float | None:
            try:
                resp = azure.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE, max_completion_tokens=64,
                )
                text = resp.choices[0].message.content or ""
                return parse_verbalized_probability(text)
            except Exception as e:
                logging.warning("Azure (%s) failed: %s", deployment, e)
                return None
        return call

    return {
        "Claude-Opus-4.7": call_anthropic_model("claude-opus-4-7"),
        "Claude-Haiku-4.5": call_anthropic_model("claude-haiku-4-5-20251001"),
        "GPT-5.4-mini": call_azure(os.environ["AZURE_OPENAI_DEPLOYMENT_MINI"]),
        "GPT-5.4-nano": call_azure(os.environ["AZURE_OPENAI_DEPLOYMENT_NANO"]),
    }


class PlannerClient:
    def __init__(self):
        from anthropic import Anthropic
        self._client = Anthropic()

    def plan(self, partition: dict) -> tuple[list[dict] | None, str]:
        outcomes = partition["outcomes"]
        enumerated = "\n".join(f"  {i}. {o}" for i, o in enumerate(outcomes))
        prompt = PLANNER_PROMPT.format(
            specialists=", ".join(SPECIALIST_ROSTER),
            partition_label=partition["label"],
            resolution_date=partition.get("date") or "unspecified",
            enumerated_outcomes=enumerated,
        )
        for attempt in range(2):
            try:
                resp = self._client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1500, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                parsed = parse_planner_json(text, len(outcomes))
                if parsed is not None:
                    return parsed, text
            except Exception as e:
                logging.warning("Planner attempt %d failed: %s", attempt + 1, e)
        return None, ""


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
    with open(E1_RESULTS) as f:
        e1 = json.load(f)
    e1_by_label = {r["label"]: r for r in e1["results"]}

    partitions = baseline[:N_PARTITIONS]
    print(f"Frontier spot-check on first {len(partitions)} partitions.")
    print(f"    Roster: {SPECIALIST_ROSTER} (Llama replaced by Opus).")

    planner = PlannerClient()
    callers = make_callers()

    results = []
    total_calls = 0
    t0 = time.time()
    for pi, p_baseline in enumerate(partitions):
        outcomes = p_baseline["outcomes"]
        date = p_baseline.get("date") or "unspecified"
        partition = dict(label=p_baseline["label"], outcomes=outcomes, date=date)

        plan, plan_raw = planner.plan(partition)
        total_calls += 1
        if plan is None:
            plan = [
                dict(specialist=SPECIALIST_ROSTER[j % len(SPECIALIST_ROSTER)],
                     context="This is one of several outcomes in a partition.")
                for j in range(len(outcomes))
            ]

        per_outcome_means = []
        per_outcome_specialist = [a["specialist"] for a in plan]
        for j, outcome in enumerate(outcomes):
            sp_name = plan[j]["specialist"]
            ctx = plan[j]["context"]
            prompt = SPECIALIST_PROMPT.format(
                context=ctx, outcome=outcome, resolution_date=date,
            )
            samples = []
            for _ in range(K):
                if total_calls >= HARD_CALL_CAP:
                    break
                p = callers[sp_name](prompt)
                total_calls += 1
                if p is not None:
                    samples.append(float(p))
            per_outcome_means.append(float(np.mean(samples)) if samples else 0.5)
            if total_calls >= HARD_CALL_CAP:
                break
        if total_calls >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP reached at partition {pi}; stopping.")
            break

        p_arr = np.array(per_outcome_means, dtype=float)
        proj, eps = project_partition(p_arr)

        e1_match = e1_by_label.get(p_baseline["label"], {})
        results.append(dict(
            label=p_baseline["label"],
            outcomes=outcomes,
            assigned_specialists=per_outcome_specialist,
            per_outcome_means=per_outcome_means,
            sum=float(p_arr.sum()),
            eps_star=float(eps),
            projected=proj.tolist(),
            baseline_eps_star=float(p_baseline["eps_star"]),
            e1_planner_eps_star=float(e1_match.get("eps_star", float("nan"))),
        ))
        elapsed = time.time() - t0
        print(
            f"  [{pi+1}/{len(partitions)}] {p_baseline['label'][:50]:<50s}  "
            f"baseline eps*={p_baseline['eps_star']:.3f}  planner={e1_match.get('eps_star', float('nan')):.3f}  "
            f"frontier eps*={eps:.3f}  ({elapsed:.0f}s)"
        )

    eps_arr = np.array([r["eps_star"] for r in results])
    e1_arr = np.array([r["e1_planner_eps_star"] for r in results])
    base_arr = np.array([r["baseline_eps_star"] for r in results])
    summary = dict(
        n_partitions=len(results), total_calls=total_calls,
        roster=SPECIALIST_ROSTER,
        mean_baseline_eps=float(base_arr.mean()),
        mean_e1_planner_eps=float(np.nanmean(e1_arr)),
        mean_frontier_eps=float(eps_arr.mean()),
        n_eps_star_pos=int((eps_arr > 1e-6).sum()),
        n_better_than_e1=int((eps_arr < e1_arr - 1e-6).sum()),
        n_worse_than_e1=int((eps_arr > e1_arr + 1e-6).sum()),
    )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2)

    print("\n=== Frontier spot-check summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
