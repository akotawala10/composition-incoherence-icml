"""
K-sweep on partitions across all 4 models.

Defends the high-rate-of-positive-residual headline against the
"this is K=8 sampling noise" critique. Re-elicits 12 partition
cliques at K=16 per (model, outcome), then sub-samples to K in
{4, 8, 16} and recomputes mean eps_star.

Protocol. Pick 12 of the 100 planner-routing simulation partitions
(seeded selection). For each partition and each of the 4 specialist
LLMs, sample K=16 verbalized probabilities per outcome at temperature
0.7. For each K_target in {4, 8, 16} and each of 4 random-assignment
seeds, build the composed quote (assign each outcome to a uniformly-
random specialist; that specialist's mean of its first K_target
samples is the marginal), then compute eps_star =
|| quote - simplex_proj(quote) ||_2.

Aggregate mean eps_star per K_target. Predicted: residuals shrink
modestly as K grows but plateau well above zero -- evidence that
the failure mode is structural, not finite-sample.
"""
from __future__ import annotations
import json
import logging
import os
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

OUT = REPO_ROOT / "results" / "t11_partition_ksweep_4models.json"
BASELINE = REPO_ROOT / "results" / "real_agent_results.json"

K_MAX = 16
K_LEVELS = [4, 8, 16]
SEEDS = 4
N_PARTITIONS = 12
TEMPERATURE = 0.7
HARD_CALL_CAP = 4000

# 12 partition indices (seeded random pick from real_agent_results.json's 100)
RNG = np.random.default_rng(7)  # seed for selection
PARTITION_INDICES = list(map(int, np.sort(RNG.choice(100, N_PARTITIONS, replace=False))))

SPECIALISTS = ["Claude-Haiku", "GPT-5.4-mini", "GPT-5.4-nano", "Llama-3.3-70b"]


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

    def make_anthropic():
        def call(prompt: str) -> float | None:
            try:
                resp = anth.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=64,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                return parse_verbalized_probability(text)
            except Exception as e:
                logging.warning("Anthropic call failed: %s", e)
                return None
        return call

    def make_azure(deployment_env_var: str):
        deployment = os.environ[deployment_env_var]
        def call(prompt: str) -> float | None:
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
                logging.warning("Azure (%s) failed: %s", deployment, e)
                return None
        return call

    def make_groq():
        def call(prompt: str) -> float | None:
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
                logging.warning("Groq failed: %s", e)
                return None
        return call

    return {
        "Claude-Haiku": make_anthropic(),
        "GPT-5.4-mini": make_azure("AZURE_OPENAI_DEPLOYMENT_MINI"),
        "GPT-5.4-nano": make_azure("AZURE_OPENAI_DEPLOYMENT_NANO"),
        "Llama-3.3-70b": make_groq(),
    }


PROMPT = (
    "You are a probabilistic forecaster. Provide your best estimate of "
    "the probability that the following question resolves YES. The "
    "probability must be a single number between 0 and 1.\n\n"
    "Question: {outcome}\n"
    "Resolution date: {resolution_date}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)


def project_partition(p: np.ndarray) -> float:
    m = p.size
    clique = Clique(
        m=m, relations=[Relation(type="partition", indices=tuple(range(m)))],
        p_hat=p,
    )
    proj = jcd_project(clique)
    return float(np.linalg.norm(p - proj))


def main() -> None:
    with open(BASELINE) as f:
        baseline = json.load(f)
    callers = make_specialist_callers()

    print(f"K-sweep on partitions (master selection seed=7)")
    print(f"  Selected indices: {PARTITION_INDICES}")
    print(f"  K_MAX={K_MAX}; K_LEVELS={K_LEVELS}; SEEDS={SEEDS}")

    # Step 1: elicit K=K_MAX samples from each (model, outcome) for each
    # selected partition. samples[partition_idx][model_name][outcome_idx] = list of floats.
    # Parallelize across the 4 specialist providers (Anthropic / Azure / Groq)
    # since each blocks on a different rate-limit pool.
    from concurrent.futures import ThreadPoolExecutor

    samples: dict = {}
    total_calls = [0]  # mutable counter shared with workers
    t0 = time.time()

    def fetch_one(sp_name: str, prompt: str, K: int) -> list[float]:
        draws: list[float] = []
        caller = callers[sp_name]
        for _ in range(K):
            p = caller(prompt)
            total_calls[0] += 1
            if p is not None:
                draws.append(float(p))
        return draws

    for ki, pi in enumerate(PARTITION_INDICES):
        partition = baseline[pi]
        outcomes = partition["outcomes"]
        date = partition.get("date") or "unspecified"
        m = len(outcomes)
        samples[pi] = {sp: [[] for _ in range(m)] for sp in SPECIALISTS}
        # Within each partition, parallelize across (specialist, outcome) since
        # each call is independent. Limit to 8 in flight to be polite.
        tasks = []
        for j, outcome in enumerate(outcomes):
            prompt = PROMPT.format(outcome=outcome, resolution_date=date)
            for sp_name in SPECIALISTS:
                tasks.append((sp_name, j, prompt))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(fetch_one, sp_name, prompt, K_MAX): (sp_name, j)
                for (sp_name, j, prompt) in tasks
            }
            for fut in futures:
                sp_name, j = futures[fut]
                samples[pi][sp_name][j] = fut.result()
        elapsed = time.time() - t0
        print(
            f"  [{ki+1}/{len(PARTITION_INDICES)}] partition {pi} '{partition['label'][:45]}' "
            f"calls={total_calls[0]}  ({elapsed:.0f}s)"
        )
        if total_calls[0] >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP {HARD_CALL_CAP} reached; stopping.")
            break

    total_calls = total_calls[0]

    # Step 2: For each K_target, run random-assignment ensemble with SEEDS
    # seeds and compute eps_star per bet.
    rng = np.random.default_rng(0)  # master seed for assignment
    by_K: dict[int, list] = {K: [] for K in K_LEVELS}
    for K_target in K_LEVELS:
        # We use a fresh rng per K_target so that assignments are
        # exactly comparable across K_target levels (same random seed).
        rng = np.random.default_rng(0)
        for seed in range(SEEDS):
            for pi in PARTITION_INDICES:
                if pi not in samples:
                    continue
                outcomes = baseline[pi]["outcomes"]
                m = len(outcomes)
                # Skip partitions where any model failed too many calls.
                ok = all(
                    len(samples[pi][sp][j]) >= K_target
                    for sp in SPECIALISTS for j in range(m)
                )
                if not ok:
                    continue
                # Random assignment of outcomes to specialists.
                assignment = rng.integers(0, len(SPECIALISTS), size=m)
                composed = np.array([
                    np.mean(samples[pi][SPECIALISTS[assignment[j]]][j][:K_target])
                    for j in range(m)
                ])
                eps = project_partition(composed)
                by_K[K_target].append(dict(
                    partition_idx=int(pi),
                    seed=int(seed),
                    K=int(K_target),
                    eps_star=float(eps),
                    sum_p=float(composed.sum()),
                ))

    # Aggregate.
    summary = dict(
        partition_indices=PARTITION_INDICES,
        K_levels=K_LEVELS,
        seeds=SEEDS,
        n_calls=total_calls,
    )
    for K in K_LEVELS:
        eps_arr = np.array([r["eps_star"] for r in by_K[K]])
        if eps_arr.size == 0:
            summary[f"K{K}"] = dict(n=0)
            continue
        summary[f"K{K}"] = dict(
            n=int(eps_arr.size),
            mean_eps=float(eps_arr.mean()),
            median_eps=float(np.median(eps_arr)),
            std_eps=float(eps_arr.std()),
            frac_eps_pos=float((eps_arr > 0.05).mean()),
            frac_eps_pos_strict=float((eps_arr > 0.01).mean()),
        )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, by_K=by_K, raw_samples=samples), f, indent=2)

    print("\n=== K-sweep summary ===")
    print(f"  total LLM calls: {total_calls}")
    print(f"\n  {'K':>4}  {'n':>5}  {'mean eps*':>11}  {'median eps*':>13}  "
          f"{'frac eps* > 0.01':>17}  {'frac eps* > 0.05':>17}")
    for K in K_LEVELS:
        s = summary[f"K{K}"]
        if s.get("n", 0) == 0:
            continue
        print(
            f"  {K:>4}  {s['n']:>5d}  {s['mean_eps']:>11.4f}  "
            f"{s['median_eps']:>13.4f}  {s['frac_eps_pos_strict']:>17.3f}  "
            f"{s['frac_eps_pos']:>17.3f}"
        )
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
