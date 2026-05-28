"""
Deployed-agent measurement: GPT-5.5 planner with specialist tools.

A multi-turn ReAct-style forecasting agent where a frontier reasoning
model (GPT-5.5) is the planner and decides autonomously how to route
each outcome of a multi-candidate partition to a specialist forecaster.
We measure the compositional residual eps^* on the planner's final
submitted joint quote --- the failure mode in a deployed-style agent
stack rather than under our forced-isolation protocol.

Two conditions:
  - unguided:           planner is asked to "estimate each outcome's probability",
                        no coherence instruction.
  - coherence-guided:   planner is told "the outcomes are mutually exclusive;
                        the probabilities should sum to 1".

For each partition, we:
  1. Hand the planner the partition + a tool spec.
  2. Loop: planner emits tool calls, we execute them, return results.
  3. Planner submits final joint probability vector via a `submit_final` tool.
  4. Compute eps^*, log all routing decisions and tool-call traces.

Output: results/deployed_agent_results.json (list of partition records,
one per (partition, condition) pair).
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JCD_ROOT = Path(os.environ.get("JCD_ROOT", str(Path(__file__).resolve().parent.parent.parent / "JCD-Forecasting")))
sys.path.insert(0, str(JCD_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(JCD_ROOT / ".env")

from real_agent_case_study import PARTITIONS, project_partition, K
from jcd.eval.sample import AnthropicClient, parse_verbalized_probability

# ============================================================================
# Specialist clients: 5-LLM panel, each exposed by a short stable id.
# ============================================================================

@dataclass
class SpecialistSpec:
    sid: str
    label: str

SPECIALISTS = [
    SpecialistSpec("claude-haiku", "Anthropic Claude-Haiku-4.5"),
    SpecialistSpec("gpt-mini",     "OpenAI GPT-5.4-mini"),
    SpecialistSpec("gpt-nano",     "OpenAI GPT-5.4-nano"),
    SpecialistSpec("deepseek-v32", "DeepSeek V3.2"),
    SpecialistSpec("llama-4",      "Llama-4-Maverick-17B-128E"),
]


def specialist_call(sid: str, question_text: str, K_samples: int = K,
                    temperature: float = 0.7) -> tuple[float, list[float]]:
    """Send a single-question prompt to a specialist; return (mean, samples)."""
    prompt = (
        "You are a probabilistic forecaster. Provide your best estimate of the "
        "probability that the following question resolves YES. The probability "
        "must be a single number between 0 and 1.\n\n"
        f"Question: {question_text}\n\n"
        "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
        "No words, no percent signs, no commentary."
    )
    samples: list[float] = []
    for k in range(K_samples):
        try:
            text = _raw_call(sid, prompt, max_tokens=8, temperature=temperature)
            p = parse_verbalized_probability(text or "")
            if p is not None and 0.0 <= p <= 1.0:
                samples.append(float(p))
        except Exception as e:
            print(f"  [specialist {sid}] call failed (k={k}): {type(e).__name__}: {e}")
    if not samples:
        return 0.5, []
    return float(np.mean(samples)), samples


def _raw_call(sid: str, prompt: str, *, max_tokens: int = 8,
              temperature: float = 0.7) -> str | None:
    """Dispatch a single chat completion to the right provider."""
    if sid == "claude-haiku":
        from anthropic import Anthropic
        c = Anthropic()
        r = c.messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return "".join(b.text for b in r.content if getattr(b, "text", None))
    if sid == "gpt-mini":
        return _azure_openai_call("AZURE_OPENAI_DEPLOYMENT_MINI", prompt, max_tokens, temperature)
    if sid == "gpt-nano":
        return _azure_openai_call("AZURE_OPENAI_DEPLOYMENT_NANO", prompt, max_tokens, temperature)
    if sid == "deepseek-v32":
        return _foundry_call(os.environ["AZURE_FOUNDRY_DEPLOYMENT_DEEPSEEK_V32"],
                             prompt, max_tokens, temperature)
    if sid == "llama-4":
        return _foundry_call(os.environ["AZURE_FOUNDRY_DEPLOYMENT_LLAMA4"],
                             prompt, max_tokens, temperature)
    raise ValueError(f"unknown specialist sid: {sid}")


def _azure_openai_call(deployment_env: str, prompt: str, max_tokens: int,
                       temperature: float) -> str | None:
    from openai import AzureOpenAI
    c = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    deployment = os.environ[deployment_env]
    r = c.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max(max_tokens, 16),
        temperature=temperature,
    )
    return r.choices[0].message.content


def _foundry_call(deployment: str, prompt: str, max_tokens: int,
                  temperature: float) -> str | None:
    from openai import OpenAI
    endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"]
    base = endpoint.split("/chat/completions")[0]
    api_version = endpoint.split("api-version=")[-1]
    c = OpenAI(api_key=os.environ["AZURE_FOUNDRY_API_KEY"], base_url=base,
               default_query={"api-version": api_version})
    r = c.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max(max_tokens, 16), temperature=temperature,
    )
    return r.choices[0].message.content


# ============================================================================
# Planner: GPT-5.5 (reasoning frontier model) with tools
# ============================================================================

def _gpt55_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=os.environ["AZURE_GPT55_API_KEY"],
        api_version=os.environ["AZURE_GPT55_API_VERSION"],
        azure_endpoint=os.environ["AZURE_GPT55_ENDPOINT"],
    )


PLANNER_SYSTEM = (
    "You are a probabilistic forecasting agent. You will be given a "
    "multi-candidate partition question (i.e., a question with several "
    "alternative answers, exactly one of which is true). You have access "
    "to specialist forecaster LLMs that you may delegate questions to. "
    "Your job is to produce the most accurate joint probability estimate "
    "for each of the m outcomes by issuing tool calls and integrating "
    "the responses.\n\n"
    "Available specialists (use the sid in delegate_to_specialist):\n"
    + "\n".join(f"  - {s.sid:<14} ({s.label})" for s in SPECIALISTS)
    + "\n\nYou may issue as many delegate_to_specialist calls as you need. "
    "Each call returns the specialist's mean probability over K=8 verbalized "
    "samples for the question text you provided. You may phrase the question "
    "however you wish (with or without context about other outcomes).\n\n"
    "When you have your final estimate, call submit_final_partition_quote with "
    "your m probabilities in the same order as the outcomes were listed."
)


PLANNER_USER_UNGUIDED = (
    "Partition question: {label}\n"
    "Outcomes (in order):\n{outcomes_listed}\n\n"
    "Please estimate the probability of each outcome. When done, call "
    "submit_final_partition_quote(probabilities=[p_1, ..., p_m]) with "
    "your final estimates."
)

PLANNER_USER_GUIDED = (
    "Partition question: {label}\n"
    "Outcomes (in order):\n{outcomes_listed}\n\n"
    "These outcomes are mutually exclusive and collectively exhaustive --- "
    "exactly one will be true --- so the probabilities you assign to all m "
    "outcomes must sum to 1. Please estimate the joint probability vector. "
    "When done, call submit_final_partition_quote(probabilities=[p_1, ..., p_m])."
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_specialist",
            "description": (
                "Send a forecasting question to one of the specialist LLMs "
                "and receive its mean probability over K=8 verbalized samples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "specialist_id": {
                        "type": "string",
                        "description": "The sid of the specialist to query.",
                        "enum": [s.sid for s in SPECIALISTS],
                    },
                    "question_text": {
                        "type": "string",
                        "description": (
                            "The question text to send. May include context "
                            "about other outcomes if you wish."
                        ),
                    },
                },
                "required": ["specialist_id", "question_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_partition_quote",
            "description": (
                "Submit your final joint probability vector for the m outcomes. "
                "Call this once when finished."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probabilities": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "List of m probabilities in [0,1], in the same order "
                            "as the outcomes were listed."
                        ),
                    },
                },
                "required": ["probabilities"],
            },
        },
    },
]


def run_agent_on_partition(partition: dict, condition: str,
                           planner: str = "gpt-5.5",
                           max_turns: int = 12,
                           verbose: bool = False) -> dict:
    """Run the planner on a single partition under one condition.

    planner: "gpt-5.5" (frontier reasoning) or "claude-haiku" (non-reasoning).
    """
    if planner == "claude-haiku":
        return _run_claude_haiku_planner(partition, condition, max_turns, verbose)
    label = partition["label"]
    outcomes = partition["outcomes"]
    m = len(outcomes)
    outcomes_listed = "\n".join(f"  ({chr(ord('a')+i)}) {o}" for i, o in enumerate(outcomes))

    user_template = PLANNER_USER_UNGUIDED if condition == "unguided" else PLANNER_USER_GUIDED
    user_msg = user_template.format(label=label, outcomes_listed=outcomes_listed)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    client = _gpt55_client()
    deployment = os.environ["AZURE_GPT55_DEPLOYMENT"]

    trace: list[dict[str, Any]] = []
    final_probs: list[float] | None = None
    t0 = time.time()

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=4096,
                temperature=1.0,    # planner; let it reason
            )
        except Exception as e:
            print(f"  [planner turn {turn}] FAILED: {type(e).__name__}: {e}")
            break
        msg = resp.choices[0].message
        # Append the assistant message (with tool_calls if any) to history
        msg_record: dict[str, Any] = {"role": "assistant"}
        if getattr(msg, "content", None):
            msg_record["content"] = msg.content
        else:
            msg_record["content"] = None
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            msg_record["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(msg_record)

        if not tool_calls:
            # Planner produced text without tools; either it submitted via
            # text or it stopped. Try to parse a probability list from text.
            if verbose:
                print(f"  [planner turn {turn}] no tool calls; content={msg.content!r}")
            text = msg.content or ""
            parsed = _parse_probs_from_text(text, m)
            if parsed is not None:
                final_probs = parsed
            break

        # Execute each tool call and append a tool-result message
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "delegate_to_specialist":
                sid = args.get("specialist_id", "")
                qtext = args.get("question_text", "")
                if verbose:
                    print(f"  [delegate -> {sid}] {qtext[:80]!r}")
                mean, samples = specialist_call(sid, qtext)
                tool_result = {"probability": mean, "n_samples": len(samples)}
                trace.append({"turn": turn, "tool": name, "args": args, "result": tool_result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })
            elif name == "submit_final_partition_quote":
                probs = args.get("probabilities", [])
                trace.append({"turn": turn, "tool": name, "args": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"received": True, "n": len(probs)}),
                })
                if isinstance(probs, list) and len(probs) == m:
                    final_probs = [float(p) for p in probs]
                if final_probs is not None:
                    break  # break inner tool loop
            else:
                trace.append({"turn": turn, "tool": name, "args": args, "error": "unknown tool"})
        if final_probs is not None:
            break

    elapsed = time.time() - t0
    if final_probs is None or len(final_probs) != m:
        # Fallback: try last assistant text or produce uniform.
        if final_probs is not None and len(final_probs) != m:
            final_probs = None
        if final_probs is None:
            final_probs = [1.0 / m] * m

    p = np.array(final_probs, dtype=float)
    proj, eps = project_partition(p)
    sum_violation = float(abs(p.sum() - 1.0))

    return dict(
        label=label,
        outcomes=outcomes,
        condition=condition,
        planner="gpt-5.5",
        final_probs=final_probs,
        sum=float(p.sum()),
        sum_violation=sum_violation,
        eps_star=eps,
        projected=proj.tolist(),
        n_turns=len(trace),
        n_delegate_calls=sum(1 for t in trace if t["tool"] == "delegate_to_specialist"),
        elapsed_s=elapsed,
        trace=trace,
    )


def _run_claude_haiku_planner(partition: dict, condition: str,
                              max_turns: int = 12,
                              verbose: bool = False) -> dict:
    """Same harness as GPT-5.5 but using Anthropic tool-use format with
    claude-haiku-4-5 as the planner (non-reasoning baseline)."""
    from anthropic import Anthropic
    label = partition["label"]
    outcomes = partition["outcomes"]
    m = len(outcomes)
    outcomes_listed = "\n".join(f"  ({chr(ord('a')+i)}) {o}" for i, o in enumerate(outcomes))
    user_template = PLANNER_USER_UNGUIDED if condition == "unguided" else PLANNER_USER_GUIDED
    user_msg = user_template.format(label=label, outcomes_listed=outcomes_listed)

    # Anthropic tools format: name + description + input_schema
    anthropic_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in TOOLS
    ]

    client = Anthropic()
    model = "claude-haiku-4-5-20251001"

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
    trace: list[dict[str, Any]] = []
    final_probs: list[float] | None = None
    t0 = time.time()

    for turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model,
                system=PLANNER_SYSTEM,
                messages=messages,
                tools=anthropic_tools,
                max_tokens=4096,
            )
        except Exception as e:
            print(f"  [haiku planner turn {turn}] FAILED: {type(e).__name__}: {e}")
            break

        # Append assistant message with content blocks
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # No tools called; try parsing text
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
            parsed = _parse_probs_from_text(text, m)
            if parsed is not None:
                final_probs = parsed
            break

        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            name = tu.name
            args = tu.input or {}
            if name == "delegate_to_specialist":
                sid = args.get("specialist_id", "")
                qtext = args.get("question_text", "")
                if verbose:
                    print(f"  [delegate -> {sid}] {qtext[:80]!r}")
                mean, samples = specialist_call(sid, qtext)
                result = {"probability": mean, "n_samples": len(samples)}
                trace.append({"turn": turn, "tool": name, "args": args, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                })
            elif name == "submit_final_partition_quote":
                probs = args.get("probabilities", [])
                trace.append({"turn": turn, "tool": name, "args": args})
                if isinstance(probs, list) and len(probs) == m:
                    final_probs = [float(p) for p in probs]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"received": True, "n": len(probs)}),
                })
            else:
                trace.append({"turn": turn, "tool": name, "args": args, "error": "unknown"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"error": "unknown tool"}),
                    "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})
        if final_probs is not None:
            break

    elapsed = time.time() - t0
    if final_probs is None or len(final_probs) != m:
        final_probs = [1.0 / m] * m

    p = np.array(final_probs, dtype=float)
    proj, eps = project_partition(p)
    sum_violation = float(abs(p.sum() - 1.0))
    return dict(
        label=label,
        outcomes=outcomes,
        condition=condition,
        planner="claude-haiku",
        final_probs=final_probs,
        sum=float(p.sum()),
        sum_violation=sum_violation,
        eps_star=eps,
        projected=proj.tolist(),
        n_turns=len(trace),
        n_delegate_calls=sum(1 for t in trace if t["tool"] == "delegate_to_specialist"),
        elapsed_s=elapsed,
        trace=trace,
    )


def _parse_probs_from_text(text: str, m: int) -> list[float] | None:
    """Best-effort parse of a probability vector from free text."""
    # find a comma/space separated list of floats
    candidates = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.?\d*", text)
    if len(candidates) >= m:
        try:
            vals = [float(c) for c in candidates[:m]]
            if all(0.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in vals):
                return [max(0.0, min(1.0, v)) for v in vals]
        except ValueError:
            return None
    return None


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="number of partitions")
    p.add_argument("--start", type=int, default=0, help="partition start index")
    p.add_argument("--planner", type=str, default="gpt-5.5",
                   choices=["gpt-5.5", "claude-haiku"])
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "data" / "results" / "deployed_agent_results.json"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--probe", action="store_true",
                   help="run only 2 partitions for debugging")
    args = p.parse_args()

    if args.probe:
        args.n = 2
    subset = PARTITIONS[args.start : args.start + args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, partition in enumerate(subset):
        for condition in ("unguided", "coherence-guided"):
            print(f"\n[{i+1}/{len(subset)}] {partition['label']:60s}  ({condition}, planner={args.planner})", flush=True)
            try:
                rec = run_agent_on_partition(partition, condition,
                                             planner=args.planner,
                                             verbose=args.verbose)
            except Exception as e:
                print(f"  HARD FAIL: {type(e).__name__}: {e}")
                rec = dict(label=partition["label"], condition=condition,
                           error=f"{type(e).__name__}: {e}")
            print(f"  sum={rec.get('sum',float('nan')):.3f}  "
                  f"eps*={rec.get('eps_star',float('nan')):.4f}  "
                  f"delegate_calls={rec.get('n_delegate_calls','?')}", flush=True)
            results.append(rec)
        # incremental save
        if (i + 1) % 2 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} records to {out_path}")

    # Summary
    eps = {"unguided": [], "coherence-guided": []}
    sumv = {"unguided": [], "coherence-guided": []}
    for r in results:
        if "eps_star" not in r:
            continue
        eps[r["condition"]].append(r["eps_star"])
        sumv[r["condition"]].append(r["sum_violation"])
    print("\nSummary:")
    print(f"{'condition':<22}{'<eps*>':>10}{'<|sum-1|>':>14}{'frac eps>1e-3':>18}{'N':>5}")
    for cond in ("unguided", "coherence-guided"):
        e = np.array(eps[cond]) if eps[cond] else np.array([])
        s = np.array(sumv[cond]) if sumv[cond] else np.array([])
        if len(e) == 0:
            print(f"{cond:<22}{'(no data)':>10}")
            continue
        print(f"{cond:<22}{e.mean():>10.4f}{s.mean():>14.4f}{(e > 1e-3).mean():>18.3f}{len(e):>5}")


if __name__ == "__main__":
    main()
