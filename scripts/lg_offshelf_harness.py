"""
LangGraph off-the-shelf multi-agent harness for partition forecasting.

Goal: rule out that the compositional failure mode is an artefact of
hand-engineered routing/prompting protocols. We instrument an
off-the-shelf LangGraph multi-agent template (Researcher + Analyst +
Aggregator nodes), task it with multi-candidate partition forecasting
on resolved Polymarket events, and measure the compositional residual
eps_star on whatever joint quote the framework emits.

Design choices, in the spirit of "as deployed":
  - Three-node graph: Researcher (gathers context), Analyst (per-outcome
    probability estimates), Aggregator (assembles JSON output).
  - We use vanilla LangGraph + LangChain primitives. The agents are
    LLM nodes with no custom routing prompt beyond the framework's
    standard system messages.
  - Final node prompt asks for a coherent JSON {"outcomes": [...],
    "probabilities": [...]} -- it is told the partition structure
    explicitly, the strongest reasonable instruction.
  - We parse the JSON output, compute eps_star (distance to simplex
    \\Pi_{simplex}), apply hierarchical-JCD repair, and compare both
    quotes against the resolved outcome (Brier + log-payoff).
  - On parse failure: we record the failure mode, do NOT impute.

Reproducibility:
  - LangGraph version, LLM model (Claude-Haiku-4.5 for both nodes),
    temperature 0.0 for the aggregator (we want deterministic JSON),
    temperature 0.7 for researcher and analyst (default).
  - Master seed = 0 for partition selection.
  - Output: results/lg_offshelf_harness.json with per-event traces.
"""
from __future__ import annotations
import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
JCD_DATA = REPO_ROOT / "data" / "polymarket" / "markets.jsonl"

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Suppress LangChain deprecation chatter so the log is readable.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)

from jcd.qp.solver import project as jcd_project  # noqa: E402
from jcd.types import Clique, Relation  # noqa: E402
from jcd.data.polymarket import PolymarketMarket, mine_mutex_cliques  # noqa: E402

OUT = REPO_ROOT / "results" / "lg_offshelf_harness.json"
TRACES_DIR = REPO_ROOT / "results" / "lg_traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

N_EVENTS = 30
HARD_CALL_CAP = 200  # 30 events * ~3 LLM calls each = 90 expected
SEED = 0


# ---------------------------------------------------------------------------
# 1) Pull resolved Polymarket partitions with titles
# ---------------------------------------------------------------------------

def load_resolved_partitions(n: int, master_seed: int) -> list[dict]:
    """Return a list of partition events with event title + per-outcome
    market questions + resolutions. We use ``PolymarketMarket.question``
    (the candidate-distinguishing text, e.g.\\ 'Will Real Madrid win?')
    rather than the boilerplate resolution body."""
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
        # Require all resolved.
        if any(mk.resolution is None for mk in group):
            continue
        # Exactly one YES.
        if sum(1 for mk in group if mk.resolution is True) != 1:
            continue
        # All have non-trivial distinguishing question text.
        outcomes = [mk.question for mk in group]
        if any(not o or len(o) < 8 for o in outcomes):
            continue
        # Drop events where all market questions are identical.
        if len(set(outcomes)) < len(outcomes):
            continue
        resolutions = [1.0 if mk.resolution else 0.0 for mk in group]
        out.append(dict(
            event_id=ev_id,
            event_title=event_titles.get(ev_id, ev_id),
            outcomes=outcomes,
            resolutions=resolutions,
            n=len(outcomes),
        ))

    rng = np.random.default_rng(master_seed)
    if len(out) > n:
        idx = sorted(rng.choice(len(out), size=n, replace=False).tolist())
        out = [out[i] for i in idx]
    return out


# ---------------------------------------------------------------------------
# 2) LangGraph multi-agent template (Researcher + Analyst + Aggregator)
# ---------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # noqa: E402


class AgentState(TypedDict):
    event_label: str
    event_title: str
    outcomes: list[str]
    research_notes: str
    analyst_estimates: str
    final_json: str
    parsed: dict | None
    parse_error: str | None


# We use Claude-Haiku-4.5 for all nodes. Temperature varies by node:
# 0.7 for the reasoning nodes (default LLM usage), 0.0 for the aggregator
# (we want valid JSON).
def make_llm(temperature: float = 0.7):
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=temperature,
        max_tokens=1500,
    )


RESEARCHER_PROMPT = """You are a Researcher agent in a multi-agent forecasting team. Your role is to gather relevant context and base rates for a multi-candidate forecasting question.

You will receive a partition forecasting question (one of N candidate outcomes will resolve YES, the others NO). Provide:
1. Relevant historical base rates or reference classes.
2. Recent factors that might tilt probabilities toward specific outcomes.
3. Any salient considerations the analyst should weigh.

Be concise (≤ 250 words). Do NOT yet emit probabilities."""


ANALYST_PROMPT = """You are an Analyst agent. Given the Researcher's notes and the partition's candidate outcomes, produce per-outcome probability estimates with brief justifications.

Output format: a numbered list, one line per outcome, of the form
  i. <outcome label>: <probability> -- <one-sentence rationale>
where <probability> is a number in [0, 1].

Constraint: the partition has exactly one outcome that will resolve YES, so your probabilities should reflect a coherent multi-candidate distribution.

Be concise. Do NOT add commentary outside the numbered list."""


AGGREGATOR_PROMPT = """You are the Aggregator agent. Combine the Analyst's per-outcome estimates into a final JSON output.

Output ONLY a JSON object with EXACTLY this shape (no other text, no markdown fence):
{{
  "probabilities": [<p_0>, <p_1>, ..., <p_{n_minus_1}>]
}}
where each p_i is in [0,1] and they correspond, in order, to:
{enumerated_outcomes}

The N values are independent probability estimates per outcome. Do not add fields, comments, or explanation."""


def parse_aggregator_json(text: str, n_outcomes: int) -> tuple[dict | None, str | None]:
    if not text:
        return None, "empty response"
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no JSON object found"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json decode error: {e}"
    probs = d.get("probabilities")
    if not isinstance(probs, list) or len(probs) != n_outcomes:
        return None, f"expected list of {n_outcomes} probabilities, got {type(probs).__name__} of length {len(probs) if isinstance(probs, list) else 'N/A'}"
    try:
        probs_f = [float(x) for x in probs]
    except (TypeError, ValueError) as e:
        return None, f"non-numeric probability: {e}"
    return dict(probabilities=probs_f), None


# ---------- Node implementations ----------

def researcher_node(state: AgentState) -> dict:
    llm = make_llm(temperature=0.7)
    user = (
        f"Event: {state['event_title']}\n\n"
        f"Candidate outcomes (exactly one will resolve YES):\n"
        + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(state["outcomes"]))
    )
    resp = llm.invoke([SystemMessage(RESEARCHER_PROMPT), HumanMessage(user)])
    return {"research_notes": resp.content if isinstance(resp.content, str)
            else "".join(c.get("text", "") for c in resp.content if isinstance(c, dict))}


def analyst_node(state: AgentState) -> dict:
    llm = make_llm(temperature=0.7)
    user = (
        f"Event: {state['event_title']}\n\n"
        f"Researcher's notes:\n{state['research_notes']}\n\n"
        f"Candidate outcomes (in order):\n"
        + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(state["outcomes"]))
        + "\n\nProduce per-outcome probability estimates."
    )
    resp = llm.invoke([SystemMessage(ANALYST_PROMPT), HumanMessage(user)])
    return {"analyst_estimates": resp.content if isinstance(resp.content, str)
            else "".join(c.get("text", "") for c in resp.content if isinstance(c, dict))}


def aggregator_node(state: AgentState) -> dict:
    llm = make_llm(temperature=0.0)
    enumerated = "\n".join(f"  {i}. {o}" for i, o in enumerate(state["outcomes"]))
    sysmsg = AGGREGATOR_PROMPT.format(
        enumerated_outcomes=enumerated,
        n_minus_1=len(state["outcomes"]) - 1,
    )
    user = (
        f"Analyst estimates:\n{state['analyst_estimates']}\n\n"
        "Now emit the JSON object as instructed."
    )
    resp = llm.invoke([SystemMessage(sysmsg), HumanMessage(user)])
    text = resp.content if isinstance(resp.content, str) \
        else "".join(c.get("text", "") for c in resp.content if isinstance(c, dict))
    parsed, err = parse_aggregator_json(text, len(state["outcomes"]))
    return {"final_json": text, "parsed": parsed, "parse_error": err}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("aggregator", aggregator_node)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "aggregator")
    g.add_edge("aggregator", END)
    return g.compile()


# ---------------------------------------------------------------------------
# 3) Compute eps_star + JCD repair + downstream regret
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


def brier(p: np.ndarray, res: np.ndarray) -> float:
    return float(np.sum((p - res) ** 2))


def log_payoff(w: np.ndarray, winner: int) -> float:
    return float(np.log(max(w[winner], 1e-9)))


# ---------------------------------------------------------------------------
# 4) Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"LangGraph off-the-shelf harness -- N={N_EVENTS} events, master seed={SEED}")
    partitions = load_resolved_partitions(N_EVENTS, SEED)
    print(f"  loaded {len(partitions)} resolved Polymarket partitions")
    if not partitions:
        print("  WARN: no partitions found; exiting.")
        return

    graph = build_graph()
    rows = []
    n_calls = 0
    n_parse_failures = 0
    t0 = time.time()
    for i, ev in enumerate(partitions):
        if n_calls >= HARD_CALL_CAP:
            print(f"  HARD_CALL_CAP {HARD_CALL_CAP} reached; stopping.")
            break
        outcomes = ev["outcomes"]
        m = len(outcomes)
        # Use the first outcome as a label since gamma cache lacks event title;
        # researcher will see the full outcome list anyway.
        label = ev["event_id"]
        try:
            state: AgentState = {
                "event_label": label,
                "event_title": ev.get("event_title", label),
                "outcomes": outcomes,
                "research_notes": "",
                "analyst_estimates": "",
                "final_json": "",
                "parsed": None,
                "parse_error": None,
            }
            result = graph.invoke(state)
            n_calls += 3
        except Exception as e:
            print(f"  [{i+1}/{len(partitions)}] event_id={label} : graph run failed: {e}")
            rows.append(dict(
                event_id=label, outcomes=outcomes,
                resolutions=ev["resolutions"],
                error=str(e),
            ))
            continue

        parsed = result.get("parsed")
        parse_error = result.get("parse_error")
        # Save trace
        trace_path = TRACES_DIR / f"event_{i:03d}_{label[:30].replace('/', '_')}.json"
        with open(trace_path, "w") as f:
            json.dump(dict(
                event_id=label, outcomes=outcomes,
                resolutions=ev["resolutions"],
                research_notes=result.get("research_notes", ""),
                analyst_estimates=result.get("analyst_estimates", ""),
                final_json=result.get("final_json", ""),
                parsed=parsed, parse_error=parse_error,
            ), f, indent=2)

        if parsed is None:
            print(f"  [{i+1}/{len(partitions)}] event_id={label[:30]:<30s} PARSE FAIL: {parse_error}")
            n_parse_failures += 1
            rows.append(dict(
                event_id=label, outcomes=outcomes,
                resolutions=ev["resolutions"],
                parse_error=parse_error,
                final_json=result.get("final_json", ""),
            ))
            continue

        # Compute eps_star + JCD + regret
        p_naive = np.array(parsed["probabilities"], dtype=float)
        proj, eps = project_partition(p_naive)
        # Brier always meaningful
        res = np.array(ev["resolutions"], dtype=float)
        winner = int(np.argmax(res))
        b_naive = brier(p_naive, res)
        b_jcd = brier(proj, res)
        # Log-payoff via proportional allocation
        w_naive = proportional_alloc(p_naive)
        w_jcd = proportional_alloc(proj)
        lp_naive = log_payoff(w_naive, winner)
        lp_jcd = log_payoff(w_jcd, winner)

        rows.append(dict(
            event_id=label,
            outcomes=outcomes,
            resolutions=ev["resolutions"],
            naive_quote=p_naive.tolist(),
            naive_sum=float(p_naive.sum()),
            naive_eps_star=eps,
            jcd_quote=proj.tolist(),
            naive_brier=b_naive,
            jcd_brier=b_jcd,
            naive_log_payoff=lp_naive,
            jcd_log_payoff=lp_jcd,
            winner=winner,
            n_outcomes=m,
        ))
        elapsed = time.time() - t0
        print(
            f"  [{i+1}/{len(partitions)}] event={label[:24]:<24s} m={m} "
            f"sum={p_naive.sum():.3f} eps*={eps:.3f} "
            f"Brier naive={b_naive:.3f} jcd={b_jcd:.3f} "
            f"lp naive={lp_naive:.3f} jcd={lp_jcd:.3f} ({elapsed:.0f}s)"
        )

    # Aggregate
    valid = [r for r in rows if "naive_eps_star" in r]
    if not valid:
        print("  No valid rows; aborting summary.")
        return

    eps = np.array([r["naive_eps_star"] for r in valid])
    sums = np.array([r["naive_sum"] for r in valid])
    b_n = np.array([r["naive_brier"] for r in valid])
    b_j = np.array([r["jcd_brier"] for r in valid])
    lp_n = np.array([r["naive_log_payoff"] for r in valid])
    lp_j = np.array([r["jcd_log_payoff"] for r in valid])

    summary = dict(
        n_events=len(valid),
        n_parse_failures=n_parse_failures,
        n_calls=n_calls,
        mean_naive_eps_star=float(eps.mean()),
        median_naive_eps_star=float(np.median(eps)),
        max_naive_eps_star=float(eps.max()),
        mean_naive_sum=float(sums.mean()),
        n_eps_star_above_0=int((eps > 1e-6).sum()),
        n_eps_star_above_005=int((eps > 0.05).sum()),
        n_eps_star_above_01=int((eps > 0.1).sum()),
        mean_brier_naive=float(b_n.mean()),
        mean_brier_jcd=float(b_j.mean()),
        mean_delta_brier=float((b_j - b_n).mean()),
        mean_logp_naive=float(lp_n.mean()),
        mean_logp_jcd=float(lp_j.mean()),
        mean_delta_logp=float((lp_j - lp_n).mean()),
    )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=2)

    print("\n=== LangGraph off-the-shelf harness summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
