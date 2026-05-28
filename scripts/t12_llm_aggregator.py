"""
LLM-as-aggregator baseline.

The most likely critique of hierarchical JCD is: "why not just
have an LLM look at the assembled marginals and re-emit a coherent
quote?" This experiment tests exactly that.

Protocol. For each of the 100 partitions in real_agent_results.json,
the per-outcome marginals from the random-assignment ensemble are
fed to a 5th LLM (the *aggregator*) along with the partition label
and outcome texts. The aggregator is instructed to emit a coherent
partition vector (sums to 1, all non-negative) that best reflects
the input marginals.

We then compute eps_star (distance from the joint coherent simplex)
on three regimes:
  - naive  : original assembled marginals (from real_agent_results.json)
  - llmagg : aggregator's emitted partition vector
  - jcd    : geometric simplex projection of the naive quote

Question: does the LLM aggregator achieve eps_star ~= 0 (it
self-coherentizes), eps_star between baseline and JCD (partial
repair), or sometimes eps_star > baseline (it can hurt)?
"""
from __future__ import annotations
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402

logging.basicConfig(level=logging.WARNING)

OUT = REPO_ROOT / "results" / "t12_llm_aggregator.json"
BASELINE = REPO_ROOT / "results" / "real_agent_results.json"
N_PARTITIONS = 100
HARD_CALL_CAP = 200  # Aggregator is a single call per partition.
TEMPERATURE = 0.0  # We want deterministic aggregator output.

AGGREGATOR_PROMPT = """You are an aggregator agent. You receive per-outcome marginal probability estimates for a multi-candidate forecasting partition (a question where exactly ONE of the listed outcomes will resolve YES; the others will resolve NO). The marginals were independently produced by different specialist forecasters who did NOT see the partition structure. Your job: reconcile these marginals into a SINGLE coherent partition probability vector.

Constraints on your output:
  - Each output probability must be in [0, 1].
  - The output probabilities must sum to 1 (since exactly one outcome resolves YES).
  - You should respect the relative ordering and rough magnitudes of the input marginals where reasonable, but you must satisfy the partition constraint.

Partition label: {partition_label}
Resolution date: {resolution_date}

Per-outcome marginals (specialist marginal -> outcome text):
{enumerated}

Output ONLY a comma-separated list of {n_outcomes} probabilities in the SAME ORDER as the outcomes above. Each must be in [0, 1] and they must sum to 1.0. Do not include any other text, no labels, no markdown, no commentary, no explanation. Just the {n_outcomes} numbers separated by commas."""


def parse_csv_floats(text: str, n: int) -> list[float] | None:
    if not text:
        return None
    text = re.sub(r"^```(?:csv|text)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    parts = re.split(r"[,\s]+", text.strip())
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    if len(out) < n:
        return None
    return out[:n]


def make_anthropic_aggregator():
    from anthropic import Anthropic
    client = Anthropic()

    def call(prompt: str) -> list[float] | None:
        for attempt in range(2):
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    blk.text for blk in resp.content
                    if getattr(blk, "type", "") == "text"
                )
                return text
            except Exception as e:
                logging.warning("Anthropic agg attempt %d failed: %s", attempt + 1, e)
        return None

    return call


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
    aggregator = make_anthropic_aggregator()

    rows = []
    failures = 0
    t0 = time.time()
    for pi, p_baseline in enumerate(partitions):
        outcomes = p_baseline["outcomes"]
        m = len(outcomes)
        marginals = p_baseline["per_outcome_means"]
        specialists = p_baseline["assigned_specialists"]
        date = p_baseline.get("date") or "unspecified"

        # Build aggregator prompt
        enumerated = "\n".join(
            f"  {marginals[j]:.3f} ({specialists[j]}) -> {outcomes[j]}"
            for j in range(m)
        )
        prompt = AGGREGATOR_PROMPT.format(
            partition_label=p_baseline["label"],
            resolution_date=date,
            enumerated=enumerated,
            n_outcomes=m,
        )
        text = aggregator(prompt)
        parsed = parse_csv_floats(text, m) if text else None
        if parsed is None or len(parsed) != m:
            print(f"  [{pi+1}/{len(partitions)}] WARN: aggregator output malformed")
            failures += 1
            continue
        agg_quote = np.array(parsed, dtype=float)

        # Naive: from real_agent_results.json (existing)
        naive_quote = np.array(marginals, dtype=float)
        eps_naive = float(p_baseline["eps_star"])
        # Geometric JCD: simplex projection of naive. By definition eps*
        # of the *projected* quote is ~0 (numerical floor); the
        # informative quantity is how much it moved (= eps_naive itself).
        jcd_quote, _ = project_partition(naive_quote)
        # Recompute eps* for jcd_quote: should be at the QP solver floor.
        _, eps_jcd_residual = project_partition(jcd_quote)
        # eps_star for the LLM aggregator output (its raw output may
        # itself violate the simplex constraint).
        _, eps_agg = project_partition(agg_quote)

        rows.append(dict(
            label=p_baseline["label"],
            n_outcomes=m,
            naive_quote=naive_quote.tolist(),
            naive_sum=float(naive_quote.sum()),
            naive_eps_star=eps_naive,
            llmagg_quote=agg_quote.tolist(),
            llmagg_sum=float(agg_quote.sum()),
            llmagg_eps_star=eps_agg,
            jcd_quote=jcd_quote.tolist(),
            jcd_eps_star=eps_jcd_residual,
        ))
        elapsed = time.time() - t0
        print(
            f"  [{pi+1}/{len(partitions)}] {p_baseline['label'][:50]:<50s}  "
            f"naive eps*={eps_naive:.3f}  llm_agg eps*={eps_agg:.3f}  "
            f"jcd eps*={eps_jcd_residual:.2e}  ({elapsed:.0f}s)"
        )
        if pi >= HARD_CALL_CAP:
            break

    # Aggregate stats.
    naive_eps = np.array([r["naive_eps_star"] for r in rows])
    agg_eps = np.array([r["llmagg_eps_star"] for r in rows])
    jcd_eps = np.array([r["jcd_eps_star"] for r in rows])

    summary = dict(
        n_partitions=len(rows),
        n_failures=failures,
        mean_naive_eps=float(naive_eps.mean()),
        mean_llmagg_eps=float(agg_eps.mean()),
        mean_jcd_eps=float(jcd_eps.mean()),
        median_naive_eps=float(np.median(naive_eps)),
        median_llmagg_eps=float(np.median(agg_eps)),
        median_jcd_eps=float(np.median(jcd_eps)),
        # How often does LLM aggregator achieve eps* < some_threshold?
        n_llmagg_below_1e3=int((agg_eps < 1e-3).sum()),
        n_llmagg_below_1e2=int((agg_eps < 1e-2).sum()),
        n_llmagg_below_naive=int((agg_eps < naive_eps - 1e-6).sum()),
        n_llmagg_above_naive=int((agg_eps > naive_eps + 1e-6).sum()),
        n_llmagg_below_jcd=int((agg_eps < jcd_eps - 1e-6).sum()),
        # Mean improvement over naive
        delta_llmagg_minus_naive_mean=float((agg_eps - naive_eps).mean()),
        delta_jcd_minus_naive_mean=float((jcd_eps - naive_eps).mean()),
    )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=2)

    print("\n=== LLM-as-aggregator summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
