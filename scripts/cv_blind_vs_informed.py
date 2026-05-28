"""
Causal-counterfactual experiment on coupling visibility.

The paper's claim is that the compositional failure mode is caused
by specialists not seeing the global coupling constraint. This
script intervenes on exactly that variable, holding the specialist
routing, the four-LLM panel, K=8, and temperature 0.7 fixed.

Source of partitions and routing: results/e1_planner_harness.json (the
planner-driven harness, 20 partitions). All 20 partitions are
used (sorted by their planner-harness ε⋆ descending; n=20 with
ε⋆ > 0).

Conditions:
  BLIND     -- specialist sees only its assigned outcome as a single
               Bernoulli question.
  INFORMED  -- specialist sees the partition label, sibling outcomes,
               the explicit Σpᵢ=1 constraint, and the BLIND-condition
               quotes from the other specialists for the other
               outcomes.

Decision (interpretation of spec): the planner-harness has a single
planner-chosen routing per partition, not random assignment. We
keep the planner-chosen specialist-to-outcome assignment fixed
across BLIND and INFORMED, and use 4 independent K=8 sampling rounds
per (partition, condition) as the four seeds. This gives 4 paired
ε⋆ values per partition.

Logging: every LLM call is appended to results/cv/calls.jsonl with
timestamp, model, condition, partition_idx, outcome_idx, seed,
sample_idx, prompt, raw response, parsed prob, parse_failure flag.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402
from jcd.eval.sample import parse_verbalized_probability  # noqa: E402

# Reuse the PARTITIONS list (with resolution dates) from the
# 100-partition simulation source.
from real_agent_case_study import PARTITIONS as APP_J_PARTITIONS  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 8
TEMPERATURE = 0.7
N_SEEDS = 4
N_TOP = 20  # the planner harness has exactly 20 partitions; "top 20" is all
HARD_CALL_CAP = 7000  # safety; expected ~4,500 calls
BUDGET_USD_CAP = 50.0

PLANNER_HARNESS = REPO_ROOT / "results" / "e1_planner_harness.json"
OUT_DIR = REPO_ROOT / "results" / "cv"
OUT_DIR.mkdir(exist_ok=True)
CALLS_LOG = OUT_DIR / "calls.jsonl"
RESULTS_JSON = OUT_DIR / "results.json"

BLIND_PROMPT = (
    "What is your probability estimate that the following event "
    "resolves YES by {resolution_date}? Provide a single number "
    "between 0 and 1.\n\n"
    "Event: {outcome_text}"
)

INFORMED_PROMPT_TEMPLATE = (
    "You are one of several forecasters quoting probabilities for "
    "outcomes that together form a partition: exactly one of these "
    "outcomes will resolve YES, and the probabilities across all "
    "outcomes must sum to 1.\n\n"
    "Partition: {partition_label}\n"
    "All outcomes in this partition:\n"
    "{enumerated_outcomes}\n\n"
    "Your assigned outcome: {your_outcome_text}\n"
    "Resolution date: {resolution_date}\n\n"
    "Tentative quotes from the other forecasters on the other "
    "outcomes (these may be miscalibrated and you should not "
    "blindly defer):\n"
    "{others_quotes}\n\n"
    "What is your probability estimate that YOUR assigned outcome "
    "resolves YES? Provide a single number between 0 and 1, "
    "accounting for the partition constraint and the other "
    "forecasters' quotes."
)

# ---------------------------------------------------------------------------
# Specialist clients
# ---------------------------------------------------------------------------

_LOG_LOCK = Lock()


def _log_call(record: dict) -> None:
    with _LOG_LOCK:
        with open(CALLS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")


def make_specialist_callers() -> dict:
    from anthropic import Anthropic
    from openai import OpenAI, AzureOpenAI

    anth = Anthropic()
    azure = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    groq = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    def _record(model, condition, partition_idx, outcome_idx, seed,
                sample_idx, prompt, raw, parsed, parse_failure):
        _log_call({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "condition": condition,
            "partition_idx": int(partition_idx),
            "outcome_idx": int(outcome_idx),
            "seed": int(seed),
            "sample_idx": int(sample_idx),
            "prompt": prompt,
            "raw": raw,
            "parsed": parsed,
            "parse_failure": bool(parse_failure),
        })

    def call_anthropic(prompt, *, condition, partition_idx, outcome_idx, seed, sample_idx):
        for attempt in range(3):
            try:
                resp = anth.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512, temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content
                               if getattr(b, "type", "") == "text")
                p = parse_verbalized_probability(text)
                _record("claude-haiku-4-5", condition, partition_idx, outcome_idx,
                        seed, sample_idx, prompt, text, p, p is None)
                return p
            except Exception as e:
                if attempt == 2:
                    _record("claude-haiku-4-5", condition, partition_idx, outcome_idx,
                            seed, sample_idx, prompt, f"ERROR: {e}", None, True)
                    return None
                time.sleep(0.5)
        return None

    def make_azure(deployment_env: str, model_name: str):
        deployment = os.environ[deployment_env]
        def call(prompt, *, condition, partition_idx, outcome_idx, seed, sample_idx):
            for attempt in range(3):
                try:
                    resp = azure.chat.completions.create(
                        model=deployment,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        max_completion_tokens=64,
                    )
                    text = resp.choices[0].message.content or ""
                    p = parse_verbalized_probability(text)
                    _record(model_name, condition, partition_idx, outcome_idx,
                            seed, sample_idx, prompt, text, p, p is None)
                    return p
                except Exception as e:
                    if attempt == 2:
                        _record(model_name, condition, partition_idx, outcome_idx,
                                seed, sample_idx, prompt, f"ERROR: {e}", None, True)
                        return None
                    time.sleep(0.5)
            return None
        return call

    def call_groq(prompt, *, condition, partition_idx, outcome_idx, seed, sample_idx):
        for attempt in range(3):
            try:
                resp = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=512,  # Llama tends to chain-of-thought before
                                     # emitting a number; keep generation
                                     # budget large enough for both BLIND and
                                     # the longer INFORMED prompts.
                )
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                _record("llama-3.3-70b", condition, partition_idx, outcome_idx,
                        seed, sample_idx, prompt, text, p, p is None)
                return p
            except Exception as e:
                if attempt == 2:
                    _record("llama-3.3-70b", condition, partition_idx, outcome_idx,
                            seed, sample_idx, prompt, f"ERROR: {e}", None, True)
                    return None
                time.sleep(0.5)
        return None

    return {
        "Claude-Haiku": call_anthropic,
        "GPT-5.4-mini": make_azure("AZURE_OPENAI_DEPLOYMENT_MINI", "gpt-5.4-mini"),
        "GPT-5.4-nano": make_azure("AZURE_OPENAI_DEPLOYMENT_NANO", "gpt-5.4-nano"),
        "Llama-3.3-70b": call_groq,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_partition(p: np.ndarray) -> tuple[np.ndarray, float]:
    m = p.size
    clique = Clique(
        m=m, relations=[Relation(type="partition", indices=tuple(range(m)))],
        p_hat=p,
    )
    proj = jcd_project(clique)
    return proj, float(np.linalg.norm(p - proj))


def get_resolution_date(label: str) -> str:
    for rec in APP_J_PARTITIONS:
        if rec["label"] == label:
            return rec.get("date", "unspecified")
    return "unspecified"


def gather_K(caller, prompt, condition, partition_idx, outcome_idx, seed):
    samples = []
    for k in range(K):
        p = caller(prompt, condition=condition, partition_idx=partition_idx,
                   outcome_idx=outcome_idx, seed=seed, sample_idx=k)
        if p is not None:
            samples.append(float(p))
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"BLIND vs INFORMED on the planner-harness 20 partitions")
    print(f"  K={K}, seeds={N_SEEDS}, temp={TEMPERATURE}")
    print(f"  log: {CALLS_LOG}")

    e1 = json.load(open(PLANNER_HARNESS))
    results_e1 = e1["results"]
    # Sort by eps_star desc; take top N_TOP (all 20)
    sorted_r = sorted(enumerate(results_e1),
                      key=lambda kv: kv[1]["eps_star"], reverse=True)[:N_TOP]
    print(f"  loaded {len(sorted_r)} partitions from planner harness")

    callers = make_specialist_callers()

    # Wipe call log for clean re-run.
    if CALLS_LOG.exists():
        CALLS_LOG.unlink()

    bets = []
    t0 = time.time()
    n_calls_total = 0
    for rank, (orig_idx, p_rec) in enumerate(sorted_r):
        label = p_rec["label"]
        outcomes = p_rec["outcomes"]
        m = len(outcomes)
        assigned = p_rec["assigned_specialists"]  # list of specialist names, len m
        date = get_resolution_date(label)
        print(f"\n  [{rank+1}/{len(sorted_r)}] {label[:60]:<60s}  "
              f"m={m}  baseline_eps={p_rec['eps_star']:.3f}")

        # --- BLIND condition: 4 seeds × m outcomes × K samples (parallel within seed) ---
        blind_means_per_seed = []
        for seed in range(N_SEEDS):
            with ThreadPoolExecutor(max_workers=12) as ex:
                tasks = []
                for j in range(m):
                    sp = assigned[j]
                    prompt = BLIND_PROMPT.format(
                        resolution_date=date,
                        outcome_text=outcomes[j],
                    )
                    fut = ex.submit(gather_K, callers[sp], prompt,
                                    "BLIND", orig_idx, j, seed)
                    tasks.append((j, sp, fut))
                outcome_means = np.full(m, np.nan)
                for j, sp, fut in tasks:
                    s = fut.result()
                    n_calls_total += K
                    if s:
                        outcome_means[j] = float(np.mean(s))
            blind_means_per_seed.append(outcome_means)

        # --- INFORMED condition: 4 seeds, using BLIND quote per seed as context ---
        informed_means_per_seed = []
        for seed in range(N_SEEDS):
            blind_q = blind_means_per_seed[seed]
            # Build per-outcome INFORMED prompts
            with ThreadPoolExecutor(max_workers=12) as ex:
                tasks = []
                for j in range(m):
                    sp = assigned[j]
                    enumerated = "\n".join(
                        f"{i+1}. {o}" for i, o in enumerate(outcomes)
                    )
                    others = []
                    for k, o in enumerate(outcomes):
                        if k == j:
                            continue
                        q = blind_q[k]
                        q_str = f"{q:.3f}" if np.isfinite(q) else "(missing)"
                        others.append(f"- {o}: {q_str}")
                    others_str = "\n".join(others)
                    prompt = INFORMED_PROMPT_TEMPLATE.format(
                        partition_label=label,
                        enumerated_outcomes=enumerated,
                        your_outcome_text=outcomes[j],
                        resolution_date=date,
                        others_quotes=others_str,
                    )
                    fut = ex.submit(gather_K, callers[sp], prompt,
                                    "INFORMED", orig_idx, j, seed)
                    tasks.append((j, sp, fut))
                outcome_means = np.full(m, np.nan)
                for j, sp, fut in tasks:
                    s = fut.result()
                    n_calls_total += K
                    if s:
                        outcome_means[j] = float(np.mean(s))
            informed_means_per_seed.append(outcome_means)

        # --- Compute eps_star per (condition, seed) ---
        per_seed_records = []
        for seed in range(N_SEEDS):
            b = blind_means_per_seed[seed]
            i = informed_means_per_seed[seed]
            ok_b = np.all(np.isfinite(b))
            ok_i = np.all(np.isfinite(i))
            if not (ok_b and ok_i):
                per_seed_records.append(dict(
                    seed=seed, blind_quote=b.tolist(), informed_quote=i.tolist(),
                    blind_eps_star=None, informed_eps_star=None,
                    blind_sum=None, informed_sum=None,
                    parse_failed=True,
                ))
                continue
            _, eps_b = project_partition(b)
            _, eps_i = project_partition(i)
            per_seed_records.append(dict(
                seed=seed,
                blind_quote=b.tolist(),
                informed_quote=i.tolist(),
                blind_eps_star=eps_b,
                informed_eps_star=eps_i,
                blind_sum=float(b.sum()),
                informed_sum=float(i.sum()),
                parse_failed=False,
            ))

        bets.append(dict(
            rank=rank,
            orig_idx=orig_idx,
            label=label,
            outcomes=outcomes,
            m=m,
            assigned_specialists=assigned,
            resolution_date=date,
            baseline_eps_star=p_rec["eps_star"],
            seeds=per_seed_records,
        ))

        elapsed = time.time() - t0
        # Per-partition summary
        eps_b_seeds = [s["blind_eps_star"] for s in per_seed_records
                       if s["blind_eps_star"] is not None]
        eps_i_seeds = [s["informed_eps_star"] for s in per_seed_records
                       if s["informed_eps_star"] is not None]
        if eps_b_seeds and eps_i_seeds:
            print(f"      BLIND eps* {np.mean(eps_b_seeds):.3f} (±{np.std(eps_b_seeds):.3f})  "
                  f"INFORMED eps* {np.mean(eps_i_seeds):.3f} (±{np.std(eps_i_seeds):.3f})  "
                  f"calls={n_calls_total}  ({elapsed:.0f}s)")
        if n_calls_total >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP {HARD_CALL_CAP} reached; stopping.")
            break

    # Persist
    with open(RESULTS_JSON, "w") as f:
        json.dump(dict(
            meta=dict(
                K=K, N_SEEDS=N_SEEDS, N_TOP=N_TOP,
                temperature=TEMPERATURE,
                planner_harness_source=str(PLANNER_HARNESS),
                n_calls_total=n_calls_total,
            ),
            bets=bets,
        ), f, indent=2)
    print(f"\n  wrote {RESULTS_JSON}")
    print(f"  total calls: {n_calls_total}")


if __name__ == "__main__":
    main()
