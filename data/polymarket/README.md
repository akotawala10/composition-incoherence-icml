# Polymarket data

The frontier panel (`scripts/f1_frontier_panel.py`) and the LangGraph reference
harness (`scripts/lg_offshelf_harness.py`) load Polymarket events from
`markets.jsonl` in this directory. The file is not included in the repo
(~74 MB).

## Snapshot used in the paper

- Source: Polymarket gamma-api event-level pulls
- Snapshot date: 2026-04-30
- Inclusion cutoff: events resolving on or after 2026-05-01

## Reproducing the snapshot

```bash
# Reproduce the snapshot from the gamma-api (requires network access).
# The exact filter logic is documented in jcd.data.polymarket.mine_mutex_cliques.
python -m jcd.data.polymarket --snapshot-date 2026-04-30 --output markets.jsonl
```

If you have access to the original snapshot file, place it at
`data/polymarket/markets.jsonl`. The path is also overridable via the
`POLYMARKET_DATA` environment variable in the scripts that read it.

## Obtaining the original snapshot

The exact `markets.jsonl` used to produce every Polymarket-derived number
in the paper is available on request. During blind review, requests can be
routed through the workshop's reviewer-message system; post-review, the
snapshot will be hosted alongside the (de-anonymized) code release.
