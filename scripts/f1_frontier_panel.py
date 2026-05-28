"""
Frontier-only panel for the random-assignment partition ensemble.

Replaces the mid-tier panel (Claude-Haiku-4.5, GPT-5.4-mini/nano,
Llama-3.3-70b) with four frontier-tier specialists:
  - Anthropic Claude-Opus-4.7
  - Azure OpenAI GPT-5.5 (Responses-class reasoning)
  - Azure AI Foundry DeepSeek-V3.2-2
  - Azure AI Foundry Llama-4-Maverick-17B-128E (FP8)

Protocol matches the mid-tier panel exactly: each LLM forecasts each
outcome of each Polymarket partition clique with K=8 verbalised samples;
per-LLM m-vectors are simplex-projected (within-component JCD); for 4
random-assignment seeds, each outcome is assigned to a uniform-random
LLM and the assembled quote's eps_star measures cross-component
incoherence. Brier and log-payoff regret are computed against the
resolved coordinate.

The new panel is directly comparable to the mid-tier panel on the
same 67 cliques, so a drop in eps_star from this experiment can be
attributed to capability scaling.
"""
from __future__ import annotations
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
JCD_DATA = REPO_ROOT / "data" / "polymarket" / "markets.jsonl"
load_dotenv()

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402
from jcd.data.polymarket import PolymarketMarket  # noqa: E402
from jcd.eval.sample import parse_verbalized_probability  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

OUT = REPO_ROOT / "results" / "f1_frontier_panel.json"
SAMPLE_CACHE = REPO_ROOT / "results" / "f1_samples.json"
SEED = 0
SEEDS = 4
K = 8
TEMPERATURE = 0.7
HARD_CALL_CAP = 30000  # safety cap
N_PARTITIONS = 67  # match the mid-tier panel's clique count

PROMPT_TEMPLATE = (
    "You are a probabilistic forecaster. Provide your best estimate of "
    "the probability that the following question resolves YES. The "
    "probability must be a single number between 0 and 1.\n\n"
    "Question: {question}\n"
    "Resolution date: {resolution_date}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)

SPECIALISTS = ["Claude-Opus-4.7", "GPT-5.5", "DeepSeek-V3.2", "Llama-4-Maverick"]
N_MODELS = len(SPECIALISTS)


# ---------------------------------------------------------------------------
# 1) Resolved Polymarket partitions with per-outcome question text
# ---------------------------------------------------------------------------

def load_resolved_partitions() -> list[dict]:
    markets_by_event: dict[str, list[PolymarketMarket]] = {}
    event_titles: dict[str, str] = {}
    with open(JCD_DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                m = PolymarketMarket.from_gamma_record(rec)
            except Exception:
                continue
            if m.event_id is None or not m.question:
                continue
            markets_by_event.setdefault(m.event_id, []).append(m)
            if m.event_title:
                event_titles[m.event_id] = m.event_title

    out = []
    for ev_id, group in markets_by_event.items():
        if not (3 <= len(group) <= 8):
            continue
        if any(mk.resolution is None for mk in group):
            continue
        if sum(1 for mk in group if mk.resolution is True) != 1:
            continue
        outcomes = [mk.question for mk in group]
        if any(not o or len(o) < 8 for o in outcomes):
            continue
        if len(set(outcomes)) < len(outcomes):
            continue
        # Approximate "resolution date": use the latest endDate among markets.
        end_dates = [mk.end_date for mk in group if mk.end_date]
        resolution_date = max(end_dates) if end_dates else "unspecified"
        resolutions = [1.0 if mk.resolution else 0.0 for mk in group]
        out.append(dict(
            event_id=ev_id,
            event_title=event_titles.get(ev_id, ev_id),
            outcomes=outcomes,
            resolutions=resolutions,
            resolution_date=str(resolution_date)[:10],
            n=len(outcomes),
        ))
    return out


# ---------------------------------------------------------------------------
# 2) Caller per specialist
# ---------------------------------------------------------------------------

def _make_anthropic_caller():
    from anthropic import Anthropic
    client = Anthropic()
    def call(prompt: str) -> float | None:
        for attempt in range(3):
            try:
                # Opus-4.7 doesn't accept temperature.
                kwargs = dict(model="claude-opus-4-7", max_tokens=64,
                              messages=[{"role": "user", "content": prompt}])
                resp = client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content
                               if getattr(b, "type", "") == "text")
                return parse_verbalized_probability(text)
            except Exception as e:
                if attempt == 2:
                    logging.warning("Opus call failed: %s", e)
                time.sleep(0.5)
        return None
    return call


def _make_gpt55_caller():
    endpoint = os.environ["AZURE_GPT55_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_GPT55_API_KEY"]
    api_version = os.environ["AZURE_GPT55_API_VERSION"]
    deployment = os.environ["AZURE_GPT55_DEPLOYMENT"]
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    def call(prompt: str) -> float | None:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 64,
            "reasoning_effort": "none",  # apples-to-apples vs. non-reasoning peers
            "temperature": TEMPERATURE,
        }
        for attempt in range(3):
            try:
                r = requests.post(url, headers={"api-key": api_key,
                                                "Content-Type": "application/json"},
                                  json=body, timeout=60)
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"] or ""
                    return parse_verbalized_probability(text)
                if r.status_code == 429:
                    time.sleep(2.0 + attempt * 2)
                    continue
                if attempt == 2:
                    logging.warning("GPT-5.5 %d: %s", r.status_code, r.text[:120])
            except Exception as e:
                if attempt == 2:
                    logging.warning("GPT-5.5 call failed: %s", e)
                time.sleep(0.5)
        return None
    return call


def _make_foundry_caller(deployment_env: str):
    endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"]
    api_key = os.environ["AZURE_FOUNDRY_API_KEY"]
    deployment = os.environ[deployment_env]
    def call(prompt: str) -> float | None:
        body = {
            "model": deployment,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": TEMPERATURE,
        }
        for attempt in range(3):
            try:
                r = requests.post(endpoint,
                                  headers={"api-key": api_key,
                                           "Content-Type": "application/json"},
                                  json=body, timeout=60)
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"] or ""
                    return parse_verbalized_probability(text)
                if r.status_code == 429:
                    time.sleep(2.0 + attempt * 2)
                    continue
                if attempt == 2:
                    logging.warning("Foundry %s %d: %s", deployment, r.status_code,
                                    r.text[:120])
            except Exception as e:
                if attempt == 2:
                    logging.warning("Foundry %s call failed: %s", deployment, e)
                time.sleep(0.5)
        return None
    return call


def make_callers() -> dict:
    return {
        "Claude-Opus-4.7": _make_anthropic_caller(),
        "GPT-5.5": _make_gpt55_caller(),
        "DeepSeek-V3.2": _make_foundry_caller("AZURE_FOUNDRY_DEPLOYMENT_DEEPSEEK_V32"),
        "Llama-4-Maverick": _make_foundry_caller("AZURE_FOUNDRY_DEPLOYMENT_LLAMA4"),
    }


# ---------------------------------------------------------------------------
# 3) Per-LLM full-clique forecasts (with within-component pre-projection)
# ---------------------------------------------------------------------------

def project_partition(p: np.ndarray) -> tuple[np.ndarray, float]:
    m = p.size
    clique = Clique(
        m=m, relations=[Relation(type="partition", indices=tuple(range(m)))],
        p_hat=p,
    )
    proj = jcd_project(clique)
    return proj, float(np.linalg.norm(p - proj))


def proportional_alloc(p: np.ndarray) -> np.ndarray:
    pp = np.maximum(p, 0.0)
    s = pp.sum()
    if s < 1e-9:
        return np.full_like(pp, 1.0 / pp.size)
    return pp / s


def gather_samples(
    callers: dict, partitions: list[dict], cache_path: Path,
) -> dict:
    """Per-(specialist, partition_idx, outcome_idx) -> list of K floats.
    Cached to disk so reruns are cheap."""
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  loaded sample cache from {cache_path}")
        return cached

    samples: dict = {sp: {} for sp in SPECIALISTS}
    total_calls = 0
    t0 = time.time()
    for pi, ev in enumerate(partitions):
        if total_calls >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP reached at partition {pi}; stopping.")
            break
        outcomes = ev["outcomes"]
        date = ev["resolution_date"]
        n_outcomes = len(outcomes)
        # Fan out: per (specialist, outcome) gather K samples in parallel
        # within a thread pool (16 in-flight max).
        tasks = []
        for sp in SPECIALISTS:
            samples[sp].setdefault(pi, [[] for _ in range(n_outcomes)])
            for j in range(n_outcomes):
                for k in range(K):
                    tasks.append((sp, j, k))
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {
                ex.submit(callers[sp],
                          PROMPT_TEMPLATE.format(question=outcomes[j],
                                                 resolution_date=date)): (sp, j)
                for (sp, j, _) in tasks
            }
            for fut, (sp, j) in futures.items():
                p = fut.result()
                if p is not None:
                    samples[sp][pi][j].append(float(p))
                total_calls += 1
        elapsed = time.time() - t0
        # Progress: count how many (sp, j, k) cells succeeded
        success = sum(len(samples[sp][pi][j])
                      for sp in SPECIALISTS for j in range(n_outcomes))
        print(f"  [{pi+1}/{len(partitions)}] event_title={ev['event_title'][:50]:<50s} "
              f"m={n_outcomes} success={success}/{N_MODELS * n_outcomes * K} "
              f"calls={total_calls} ({elapsed:.0f}s)")
        # Periodically flush cache so we don't lose work
        if (pi + 1) % 5 == 0:
            with open(cache_path, "w") as f:
                json.dump(samples, f)
    with open(cache_path, "w") as f:
        json.dump(samples, f)
    print(f"  cached samples to {cache_path}")
    return samples


# ---------------------------------------------------------------------------
# 4) Random-assignment ensemble, eps_star, regret
# ---------------------------------------------------------------------------

def main() -> None:
    print("Frontier-only panel on Polymarket partition cliques")
    parts = load_resolved_partitions()
    print(f"  loaded {len(parts)} resolved partition cliques")
    if not parts:
        print("  no partitions; exiting.")
        return

    # Deterministic subsample to match the mid-tier panel's clique count.
    rng_pick = np.random.default_rng(SEED)
    if len(parts) > N_PARTITIONS:
        idx = sorted(rng_pick.choice(len(parts), N_PARTITIONS, replace=False).tolist())
        parts = [parts[i] for i in idx]
        print(f"  subsampled to {len(parts)} (seeded) for comparability")

    callers = make_callers()
    samples = gather_samples(callers, parts, SAMPLE_CACHE)

    # ----- compute per-LLM mean marginals + simplex projection -----
    forecast_jcd: dict[str, dict[int, np.ndarray]] = {sp: {} for sp in SPECIALISTS}
    valid_partitions: list[int] = []
    for pi, ev in enumerate(parts):
        m = len(ev["outcomes"])
        ok_all = True
        means_per_sp = {}
        for sp in SPECIALISTS:
            sps = samples[sp].get(str(pi)) or samples[sp].get(pi)  # JSON keys are strings
            if sps is None:
                ok_all = False; break
            mean_vec = np.array([
                float(np.mean(sps[j])) if sps[j] else float("nan")
                for j in range(m)
            ])
            if np.any(np.isnan(mean_vec)):
                ok_all = False; break
            means_per_sp[sp] = mean_vec
        if not ok_all:
            continue
        # Within-component JCD: simplex projection per LLM's m-vector.
        for sp in SPECIALISTS:
            proj, _ = project_partition(means_per_sp[sp])
            forecast_jcd[sp][pi] = proj
        valid_partitions.append(pi)

    print(f"\n  partitions with full per-LLM coverage: {len(valid_partitions)}/{len(parts)}")

    # ----- random-assignment ensemble -----
    rng = np.random.default_rng(SEED)
    bets = []  # one row per (seed, clique)
    for seed in range(SEEDS):
        for pi in valid_partitions:
            ev = parts[pi]
            m = len(ev["outcomes"])
            assignment = rng.integers(0, N_MODELS, size=m)
            assembled = np.array([
                forecast_jcd[SPECIALISTS[assignment[j]]][pi][j]
                for j in range(m)
            ])
            proj, eps_star = project_partition(assembled)

            res = np.array(ev["resolutions"], dtype=float)
            winner = int(np.argmax(res))
            has_winner = bool(np.any(res > 0.5))
            br_n = float(np.sum((assembled - res) ** 2))
            br_j = float(np.sum((proj - res) ** 2))
            if has_winner:
                w_n = proportional_alloc(assembled)
                w_j = proportional_alloc(proj)
                lp_n = float(np.log(max(w_n[winner], 1e-9)))
                lp_j = float(np.log(max(w_j[winner], 1e-9)))
            else:
                lp_n = lp_j = float("nan")

            # Single-LLM oracle: pick a uniform-random LLM, use its
            # full-clique JCD forecast.
            oracle_a = int(rng.integers(0, N_MODELS))
            p_oracle = forecast_jcd[SPECIALISTS[oracle_a]][pi]
            br_o = float(np.sum((p_oracle - res) ** 2))
            if has_winner:
                w_o = proportional_alloc(p_oracle)
                lp_o = float(np.log(max(w_o[winner], 1e-9)))
            else:
                lp_o = float("nan")

            bets.append(dict(
                seed=int(seed),
                clique_idx=int(pi),
                event_id=ev["event_id"],
                event_title=ev["event_title"],
                m=m,
                assignment=[SPECIALISTS[a] for a in assignment.tolist()],
                assembled=assembled.tolist(),
                jcd_quote=proj.tolist(),
                oracle_specialist=SPECIALISTS[oracle_a],
                oracle_quote=p_oracle.tolist(),
                eps_star=eps_star,
                naive_sum=float(assembled.sum()),
                naive_brier=br_n,
                jcd_brier=br_j,
                oracle_brier=br_o,
                naive_log_payoff=lp_n,
                jcd_log_payoff=lp_j,
                oracle_log_payoff=lp_o,
            ))

    # ----- aggregate -----
    eps = np.array([b["eps_star"] for b in bets])
    sums = np.array([b["naive_sum"] for b in bets])
    brier_n = np.array([b["naive_brier"] for b in bets])
    brier_j = np.array([b["jcd_brier"] for b in bets])
    brier_o = np.array([b["oracle_brier"] for b in bets])
    has_w = np.array([not np.isnan(b["naive_log_payoff"]) for b in bets])
    lp_n = np.array([b["naive_log_payoff"] for b in bets])
    lp_j = np.array([b["jcd_log_payoff"] for b in bets])
    lp_o = np.array([b["oracle_log_payoff"] for b in bets])

    summary = dict(
        n_partitions=len(valid_partitions),
        n_bets=len(bets),
        n_with_winner=int(has_w.sum()),
        seeds=SEEDS, K=K, models=SPECIALISTS,
        mean_eps_star=float(eps.mean()),
        median_eps_star=float(np.median(eps)),
        max_eps_star=float(eps.max()),
        frac_eps_above_0=float((eps > 1e-6).mean()),
        frac_eps_above_005=float((eps > 0.05).mean()),
        frac_eps_above_01=float((eps > 0.10).mean()),
        mean_naive_sum=float(sums.mean()),
        mean_brier_naive=float(brier_n.mean()),
        mean_brier_jcd=float(brier_j.mean()),
        mean_brier_oracle=float(brier_o.mean()),
        mean_delta_brier_jcd_minus_naive=float((brier_j - brier_n).mean()),
        mean_delta_brier_oracle_minus_naive=float((brier_o - brier_n).mean()),
        mean_lp_naive=float(np.nanmean(lp_n)),
        mean_lp_jcd=float(np.nanmean(lp_j)),
        mean_lp_oracle=float(np.nanmean(lp_o)),
        mean_delta_lp_jcd_minus_naive=float(np.nanmean(lp_j - lp_n)),
        mean_delta_lp_oracle_minus_naive=float(np.nanmean(lp_o - lp_n)),
    )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, bets=bets), f, indent=2)

    print("\n=== Frontier-panel summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
