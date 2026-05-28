# Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents

Code, data, and paper source for *Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents* by Anany Kotawala (Princeton University).

Paper: [`paper/main.pdf`](paper/main.pdf) — arXiv: [`arXiv:XXXX.XXXXX`](#) (TODO: replace with real ID once posted).

Preliminary versions appeared at three ICML 2026 workshops:

- *Bounding Compositional Incoherence in Foundation Models* — [Combining Theory and Benchmarks (CTB)](https://sites.google.com/view/icml-ctb/home), poster
- *Locally Coherent, Globally Incoherent: Detecting Compositional Incoherence in Multi-Component LLM Agents* — [Statistical Frameworks for Uncertainty in Agentic Systems (AgenticUQ)](https://agentic-uncertainty-icml2026.github.io/), poster
- *Locally Coherent, Globally Incoherent: A Post-Aggregation Failure Mode in Multi-Component LLM Agents* — [Failure Modes of Agentic AI (FAGEN)](https://fagen-workshop.github.io/), poster

## Repository layout

```
.
├── paper/                            LaTeX source + compiled PDF for the arXiv version
│   ├── main.tex
│   ├── main.pdf
│   ├── refs.bib
│   ├── figures/                      figure PDFs/PNGs, TikZ figures, table .tex includes
│   ├── icml2026.{sty,bst}, *.sty
│   └── Makefile
├── src/                              installable Python package (pyproject.toml)
│   ├── compositional/                core algorithm
│   │   ├── certificate.py            ε⋆ computation
│   │   ├── projection.py             hierarchical Boyle–Dykstra
│   │   ├── aggregator.py             owner-selected aggregation
│   │   └── relations.py              coupling-constraint generators
│   └── jcd/                          vendored JCD subset (self-contained)
│       ├── types.py                  Clique, Relation
│       ├── qp/{solver,closed_form,constraints}.py    OSQP-backed L2 projection
│       ├── data/{paleka,polymarket}.py               benchmark loaders
│       └── eval/sample.py            LLM client wrappers + parsing
├── scripts/                          one script per experiment / paper section
├── tests/                            unit tests for core algorithm (pytest)
├── data/                             experiment outputs + cached LLM samples
│   ├── figures/                      per-LLM JCD output dumps (.npz, .json)
│   ├── polymarket/                   Polymarket event-pull notes
│   └── results/                      per-experiment JSON outputs
│       └── fagen/                    cached outputs from the FAGEN companion experiments
├── pyproject.toml                    package metadata + dependencies
├── requirements.txt                  pinned runtime requirements
├── .env.example                      LLM API key template
├── LICENSE                           MIT
└── README.md                         this file
```

## Installation

Python 3.10 or later.

```sh
pip install -e .
```

This installs the `compositional` and `jcd` packages plus runtime dependencies (`numpy`, `scipy`, `osqp`, `anthropic`, `openai`, `groq`, `python-dotenv`, `ddgs`).

For the LangGraph reference harness (`scripts/lg_offshelf_harness.py`):

```sh
pip install -e ".[langgraph]"
```

## Building the paper

```sh
cd paper
make pdf
```

Requires `pdflatex` and `bibtex` (TeX Live 2023+). Output is `paper/main.pdf`.

## Reproducing paper numbers

Each script writes its outputs as JSON to `data/results/` (or `data/results/fagen/` for the FAGEN-companion experiments). Cached results from the paper run are already present; rerunning regenerates them with new LLM samples (verbalized-probability sampling is non-deterministic at $T\!=\!0.7$).

### Scripts that do **not** require LLM API access

These re-derive directly from the cached samples in `data/figures/` and the JSON results in `data/results/`:

```sh
# §3.3 Disagreement bound (Prop. 3.8) — Spearman, R²
python scripts/verify_disagreement_bound.py
python scripts/e6_disagreement_correlation.py        # §5.1 / App. (Table 7)

# §3.4 Magnitude prediction (Cor. 3.9 / Table 1)
python scripts/validate_rayleigh_bound.py
python scripts/validate_magnitude_bound.py

# §3.7 e-process trajectory (Theorem D.2 / Fig. 2)
python scripts/b3_compositional_eprocess.py

# §5.1 Compositional residual + exposure bound (Fig. 1, 3)
python scripts/composition_experiments.py
python scripts/build_fig1_bound.py

# §5.4 Coupling-visibility mechanism probe (Table 2, App. S)
python scripts/cv_analyse.py

# §5.6 Downstream-decision regret (§5.6, Tables for quartile + alloc rules)
python scripts/e2_polymarket_regret.py
python scripts/b2_multi_allocation_regret.py

# §5.7 Runtime gating thresholds (Table for operating points)
python scripts/g1_threshold_roc.py

# App. R Trivial vs. hierarchical projection (per-relation gap)
python scripts/e4_trivial_repair_check.py
```

### Scripts that require LLM API access (Anthropic / Azure OpenAI / Groq)

| Script | Section / table | Approx. spend |
|---|---|---|
| `e1_planner_harness.py` | §5.3 planner-discretion harness | ~\$1 |
| `cv_blind_vs_informed.py` | §5.4 mechanism probe (BLIND vs INFORMED) | ~\$2 |
| `f1_frontier_panel.py` | §5.5 frontier-panel rerun | ~\$8 |
| `b1_frontier_spotcheck.py` | §5.5 frontier spot-check | ~\$1 |
| `e5_prompt_ablation.py` | §5.6 routing-protocol ladder | ~\$3 |
| `t12_llm_aggregator.py` | §5.2 LLM-as-aggregator mitigation | <\$1 |
| `t11_partition_ksweep.py` | App. K-sweep on partitions | ~\$3 |
| `real_agent_case_study.py` | App. N planner-routing case study | ~\$2 |
| `tool_augmented_case_study.py` | App. P tool-augmented A/B | ~\$2 |
| `intro_failure_bars.py` | Fig. 1 data | <\$1 |
| `context_ablation.py` / `deployed_agent.py` | §5.6 (CTB-version of harness; superseded by `e1_planner_harness.py`) | varies |

Older CTB-codebase post-processors are kept for reproducibility of intermediate runs:

```sh
bash scripts/run_e5_e4.sh                    # T=0.7 sweep wrapper
bash scripts/run_best_try_apis.sh            # T=0 greedy control
python scripts/postprocess_t0.py             # App. J greedy-decoding control
python scripts/postprocess_e5_e4.py          # K-sweep / same-model decoupling
python scripts/analyze_context_ablation.py
python scripts/analyze_deployed_agent.py
python scripts/merge_and_analyze_planners.py
```

### Unit tests

```sh
pytest tests/
```

Covers `certificate`, `projection`, and `aggregator` modules.

## Environment

Copy `.env.example` to `.env` and fill in keys for the providers you wish to use:

```sh
ANTHROPIC_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_MINI=...
AZURE_OPENAI_DEPLOYMENT_NANO=...
GROQ_API_KEY=...
```

Scripts that need LLM access load these via `python-dotenv` on import; the data-only scripts run without any keys.

## Models

Mid-tier panel (random-assignment ensemble, $N\!=\!1{,}876$):

| Specialist | Provider | Model identifier |
|---|---|---|
| Claude-Haiku-4.5 | Anthropic | `claude-haiku-4-5-20251001` |
| GPT-5.4-mini | Azure OpenAI | (deployment name) |
| GPT-5.4-nano | Azure OpenAI | (deployment name) |
| Llama-3.3-70b | Groq | `llama-3.3-70b-versatile` |

Frontier panel (§5.5 rerun): substitutes `claude-opus-4-7`, `gpt-5.5`, `deepseek-v3.2`, and `llama-4-maverick-17b-128e` (FP8). See `scripts/f1_frontier_panel.py` for exact identifiers.

## Master seed

All experiments use `numpy.random.default_rng(0)` for the master seed; per-experiment seeds (clique selection, prompt-ablation order) are set explicitly in each script.

## Datasets

- **Paleka** consistency benchmark — `paleka/Paleka2024Consistency` from [Paleka et al., ICLR 2025](https://arxiv.org/abs/2412.18544). Cliques are pre-mined.
- **Polymarket** partition events — queried from `gamma-api.polymarket.com` at the event level. Leakage-control protocol (cutoff date 1 May 2026) documented in App. E of the paper. The 74 MB `markets.jsonl` snapshot is not bundled; see [`data/polymarket/README.md`](data/polymarket/README.md) for reproduction instructions.
- **Planner-routing simulation partitions** — 100 hand-curated multi-candidate forecasting questions, all resolving strictly after 2026-05-01 (full enumeration in `scripts/real_agent_case_study.py`).

## Citation

```bibtex
@misc{kotawala2026compositional,
  title         = {Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents},
  author        = {Kotawala, Anany},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Preliminary versions appeared at the ICML 2026 Workshops on Combining Theory and Benchmarks (CTB), Statistical Frameworks for Uncertainty in Agentic Systems (AgenticUQ), and Failure Modes of Agentic AI (FAGEN)}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Correspondence: <akotawala@princeton.edu>.
