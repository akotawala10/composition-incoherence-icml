"""Polymarket data loader and clique miner.

We pull resolved binary markets from the Polymarket Gamma REST API
(`gamma-api.polymarket.com/markets`), group them by ``event_id`` to find
multi-market events, and mine logical-relation cliques where the structure is
unambiguous:

- **Mutex group** (size 2-N): an event whose binary markets partition the
  outcome space (their YES probabilities sum to ~1). Mapped to pairwise
  ``mutex`` relations among the markets.
- **Threshold ladder** (size 2-N): an event with multiple "X > k" markets at
  ascending thresholds. Mapped to ``implies(higher_k, lower_k)`` relations.

These are pulled from a *cached* JSON snapshot so the loader is reproducible
and offline. To refresh the cache, run :func:`fetch_polymarket_events` (which
hits the public REST API, no key required).

Output is a list of :class:`PalekaTuple` objects sharing schema with Paleka,
so the rest of the pipeline (samplers, baselines, metrics, plots) is reused
unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paleka import PalekaQuestion, PalekaTuple
from ..types import Relation

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_API = "https://gamma-api.polymarket.com/events"
DEFAULT_CACHE = Path("data/polymarket/markets.jsonl")


def _parse_json_list_floats(raw: str | list) -> list[float]:
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, str):
        return []
    try:
        return [float(x) for x in json.loads(raw)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _parse_json_list_strings(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str):
        return []
    try:
        return [str(x) for x in json.loads(raw)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


@dataclass
class PolymarketMarket:
    """One Polymarket binary market, extracted from the gamma-api response."""

    market_id: str
    question: str
    description: str
    end_date: str | None
    closed: bool
    resolved: bool
    resolution: bool | None
    event_id: str | None
    event_title: str | None
    outcome: str | None  # "YES" / "NO" / threshold label
    outcome_prices: list[float] = field(default_factory=list)
    volume_usd: float = 0.0
    raw: dict = field(default_factory=dict, repr=False, hash=False, compare=False)

    @classmethod
    def from_gamma_record(cls, rec: dict) -> "PolymarketMarket":
        ev = rec.get("events", []) or []
        first_event = ev[0] if ev else {}

        # gamma-api stores `outcomePrices` and `outcomes` as JSON-encoded strings
        prices = _parse_json_list_floats(rec.get("outcomePrices") or "[]")
        outcomes = _parse_json_list_strings(rec.get("outcomes") or "[]")

        # Resolution semantics for binary markets on Polymarket:
        #   outcomePrices == [≈1, ≈0] → YES resolved (resolution=True)
        #   outcomePrices == [≈0, ≈1] → NO  resolved (resolution=False)
        #   outcomePrices == [≈0, ≈0] → market voided / unresolved
        # We allow small noise (final settled prices are sometimes off by ~1e-7
        # due to micro-trades right before settlement); the resolution
        # threshold is 0.95.
        closed = bool(rec.get("closed"))
        resolution: bool | None = None
        outcome_str: str | None = None
        if closed and len(prices) == 2:
            p0, p1 = prices
            if p0 + p1 > 0.5:  # not voided
                # Find which outcome label is YES vs NO; default to index 0=YES.
                yes_idx = 0
                if len(outcomes) == 2:
                    for i, o in enumerate(outcomes):
                        if str(o).strip().lower() in ("yes", "true"):
                            yes_idx = i
                            break
                no_idx = 1 - yes_idx
                if prices[yes_idx] > 0.95:
                    resolution = True
                    outcome_str = "Yes"
                elif prices[no_idx] > 0.95:
                    resolution = False
                    outcome_str = "No"
        resolved = resolution is not None

        try:
            volume = float(rec.get("volume", 0) or 0)
        except (TypeError, ValueError):
            volume = 0.0

        return cls(
            market_id=str(rec.get("id") or rec.get("conditionId") or ""),
            question=str(rec.get("question") or rec.get("title") or ""),
            description=str(rec.get("description") or ""),
            end_date=rec.get("endDate"),
            closed=closed,
            resolved=resolved,
            resolution=resolution,
            event_id=str(first_event.get("id") or "") or None,
            event_title=str(first_event.get("title") or "") or None,
            outcome=outcome_str,
            outcome_prices=prices,
            volume_usd=volume,
            raw=rec,
        )

    def to_paleka_question(self) -> PalekaQuestion:
        return PalekaQuestion(
            id=self.market_id,
            title=self.question,
            body=self.description,
            resolution_date=self.end_date,
            question_type="binary",
            data_source="polymarket",
            url=f"https://polymarket.com/market/{self.market_id}",
            resolution=self.resolution,
        )


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

def fetch_polymarket_events(
    *,
    limit: int = 100,
    offset_pages: int = 20,
    closed_only: bool = True,
    min_event_volume_usd: float = 1_000.0,
    min_markets_per_event: int = 2,
    cache_path: str | Path = DEFAULT_CACHE,
    timeout: float = 30.0,
    sleep_between: float = 0.3,
) -> Path:
    """Fetch multi-market Polymarket events and cache their markets as JSONL.

    Queries the gamma-api ``/events`` endpoint ordered by volume (descending).
    Each returned event already includes its constituent markets inline; we
    flatten them and tag each market with its parent ``event.id`` so that
    :func:`mine_mutex_cliques` and :func:`mine_threshold_ladders` can group
    them.

    Pulls up to ``limit * offset_pages`` events. Polite by default (0.3s
    sleep between pages). No API key required.
    """
    try:
        import httpx
    except ImportError as e:  # pragma: no cover
        raise ImportError("install jcd[llm] for httpx") from e

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_markets = 0
    n_events_kept = 0
    with httpx.Client(timeout=timeout) as client, cache_path.open("w") as f:
        for page in range(offset_pages):
            params = {
                "limit": str(limit),
                "offset": str(page * limit),
                "order": "volume",
                "ascending": "false",
                "closed": "true" if closed_only else "false",
            }
            resp = client.get(GAMMA_EVENTS_API, params=params)
            if resp.status_code != 200:
                log.warning("events page %d: HTTP %d", page, resp.status_code)
                break
            events = resp.json()
            if not events:
                break
            for event in events:
                markets = event.get("markets") or []
                if len(markets) < min_markets_per_event:
                    continue
                try:
                    event_volume = float(event.get("volume", 0) or 0)
                except (TypeError, ValueError):
                    event_volume = 0.0
                if event_volume < min_event_volume_usd:
                    continue
                # Inject event metadata into each market record so the
                # downstream parser can find event_id via "events"[0].id.
                event_meta = {
                    "id": str(event.get("id") or ""),
                    "title": str(event.get("title") or ""),
                    "slug": event.get("slug"),
                }
                for mrec in markets:
                    mrec.setdefault("events", [])
                    if not mrec["events"]:
                        mrec["events"] = [event_meta]
                    f.write(json.dumps(mrec) + "\n")
                    n_markets += 1
                n_events_kept += 1
            time.sleep(sleep_between)
    log.info("Cached %d markets across %d events → %s",
             n_markets, n_events_kept, cache_path)
    return cache_path


def load_polymarket_markets(
    cache_path: str | Path = DEFAULT_CACHE,
) -> list[PolymarketMarket]:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Polymarket cache not found at {cache_path}. "
            "Run fetch_polymarket_events() first."
        )
    out: list[PolymarketMarket] = []
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out.append(PolymarketMarket.from_gamma_record(rec))
            except Exception as e:  # noqa: BLE001
                log.warning("skipped malformed cache line: %s", e)
    return out


# ---------------------------------------------------------------------------
# Clique mining
# ---------------------------------------------------------------------------

def mine_mutex_cliques(
    markets: list[PolymarketMarket],
    *,
    min_event_size: int = 2,
    max_event_size: int = 12,
    require_resolved: bool = True,
    require_exactly_one_yes: bool = True,
) -> list[PalekaTuple]:
    """Group markets by event_id; create a 'partition' clique per event.

    A Polymarket event with N binary markets that partition the outcome space
    (exactly one resolves YES, the rest NO) induces a single ``partition``
    relation Σ p_i = 1, encoded directly. This is the natural mathematical
    object: the simplex {(p_1,...,p_N) : Σp_i = 1, p_i ≥ 0}.

    Parameters
    ----------
    min_event_size, max_event_size : int
        Filter clique sizes (default 2 ≤ N ≤ 12 to keep QPs fast).
    require_exactly_one_yes : bool
        Polymarket multi-candidate events should resolve with exactly one YES
        (the winner) and the rest NO. If True (default), drop events with 0
        or >1 YES outcomes (they are not partitions of the outcome space).
    """
    by_event: dict[str, list[PolymarketMarket]] = {}
    for m in markets:
        if m.event_id is None:
            continue
        by_event.setdefault(m.event_id, []).append(m)

    out: list[PalekaTuple] = []
    for ev_id, group in by_event.items():
        if not (min_event_size <= len(group) <= max_event_size):
            continue
        if require_resolved and any(m.resolution is None for m in group):
            continue
        if require_exactly_one_yes:
            n_yes = sum(1 for m in group if m.resolution is True)
            if n_yes != 1:
                continue
        partition = Relation(
            type="partition",
            indices=tuple(range(len(group))),
        )
        qs = tuple(m.to_paleka_question() for m in group)
        out.append(PalekaTuple(
            checker="PolymarketPartition",
            subset="polymarket",
            questions=qs,
            relation=partition,
            metadata={"event_id": ev_id, "n_markets": len(group)},
        ))
    return out


def mine_threshold_ladders(
    markets: list[PolymarketMarket],
    *,
    min_size: int = 2,
    max_size: int = 12,
    require_resolved: bool = True,
) -> list[PalekaTuple]:
    """Find numeric-threshold ladders within an event.

    Detects markets whose questions match patterns like 'X above K' / 'X reach K'
    sharing an event_id, sorts by extracted threshold, and emits a clique with
    `implies(higher, lower)` relations chained.
    """
    import re
    threshold_re = re.compile(r"(\d{1,7}(?:[.,]\d+)?)")
    by_event: dict[str, list[PolymarketMarket]] = {}
    for m in markets:
        if m.event_id is None:
            continue
        if "above" not in m.question.lower() and "reach" not in m.question.lower() \
           and ">" not in m.question:
            continue
        by_event.setdefault(m.event_id, []).append(m)

    out: list[PalekaTuple] = []
    for ev_id, group in by_event.items():
        if len(group) < min_size:
            continue
        # Extract first numeric threshold from each question
        with_thresh: list[tuple[float, PolymarketMarket]] = []
        for m in group:
            match = threshold_re.search(m.question)
            if match is None:
                continue
            try:
                t = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            with_thresh.append((t, m))
        if len(with_thresh) < 2:
            continue
        with_thresh.sort(key=lambda x: x[0])
        ms = [m for _, m in with_thresh]
        if not (min_size <= len(ms) <= max_size):
            continue
        if require_resolved and any(m.resolution is None for m in ms):
            continue
        # implies(higher_k -> lower_k):  P(X > k_high) ≤ P(X > k_low)
        # In our convention 'implies' relation: indices=(i, j) ⇒ p_i ≤ p_j
        relations = [
            Relation(type="implies", indices=(i, j))
            for i in range(len(ms))
            for j in range(i + 1, len(ms))
        ]
        if not relations:
            continue
        primary = relations[0]
        qs = tuple(m.to_paleka_question() for m in ms)
        out.append(PalekaTuple(
            checker="PolymarketThreshold",
            subset="polymarket",
            questions=qs,
            relation=primary,
            metadata={
                "event_id": ev_id,
                "thresholds": [t for t, _ in with_thresh],
                "all_relations": [(r.type, r.indices) for r in relations],
            },
        ))
    return out


def mine_polymarket_cliques(
    markets: list[PolymarketMarket],
    *,
    require_resolved: bool = True,
) -> list[PalekaTuple]:
    """Combined mining: mutex events + threshold ladders."""
    cliques = mine_mutex_cliques(markets, require_resolved=require_resolved)
    cliques += mine_threshold_ladders(markets, require_resolved=require_resolved)
    return cliques
