#!/usr/bin/env bash
# Resolve repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# best_try API runs: T=0 greedy control (Item 3) + K=32 sweep finish (Item 7).
# Drops Groq (TPD-exhausted yesterday). Anthropic claude-haiku-4-5 only for
# greedy control to bound cost. ~$25 budget.
#
# Note: planner-with-context counterfactual is a small custom run that requires
# a different prompt protocol (each specialist sees the full partition list);
# we'll write a separate driver if Item 3 budget allows.

set -euo pipefail
cd "${JCD_ROOT:?Set JCD_ROOT to your JCD-Forecasting clone}"
export PYTHONPATH=src

OUT_T0=$REPO_ROOT/data/results/e5_greedy_t0
OUT_K32=$REPO_ROOT/data/results/e4_ksweep
mkdir -p "$OUT_T0" "$OUT_K32"

LOG=$REPO_ROOT/data/results/run_best_try.log
: > "$LOG"

echo "=== ITEM 3: T=0 greedy control (4 seeds × claude-haiku-4-5) ===" | tee -a "$LOG"
SEEDS=(20260601 20260602 20260603 20260604)
PALEKA_CHECKERS=(NegChecker AndChecker OrChecker)
PALEKA_N=60
POLY_N=30

for s in "${SEEDS[@]}"; do
  for c in "${PALEKA_CHECKERS[@]}"; do
    echo "--- T=0 seed=$s checker=$c ---" | tee -a "$LOG"
    python3 scripts/run_real_llms.py \
      --source paleka --checker "$c" --K 8 --max_records "$PALEKA_N" \
      --models anthropic/claude-haiku-4-5-20251001 \
      --temperature 0.0 --seed "$s" \
      --output "$OUT_T0/seed_$s" \
      --concurrency 8 \
      --max_total_cost 1.5 --max_cost_per_model 1.5 2>&1 | tee -a "$LOG"
  done
  echo "--- T=0 seed=$s polymarket partition ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source polymarket --K 8 --max_records "$POLY_N" \
    --polymarket_include partition \
    --models anthropic/claude-haiku-4-5-20251001 \
    --temperature 0.0 --seed "$s" \
    --output "$OUT_T0/seed_$s" \
    --concurrency 8 \
    --max_total_cost 1.0 --max_cost_per_model 1.0 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== ITEM 7: K=32 sweep — finish AndChecker/OrChecker/Polymarket ===" | tee -a "$LOG"
echo "(NegChecker already complete; using existing results dir)" | tee -a "$LOG"

# anthropic AndChecker K=32 was completed in earlier run (cached); azure mini/nano cached for NegChecker only.
# Use lower concurrency for Anthropic to avoid rate limit thrash; mini/nano can handle 8.
K4_PALEKA_N=40
K4_POLY_N=20

# Run anthropic separately at concurrency=4 to avoid the 7hr disaster
for c in OrChecker; do
  echo "--- K=32 anthropic checker=$c (concurrency=4 to stay below RPM ceiling) ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source paleka --checker "$c" --K 32 --max_records "$K4_PALEKA_N" \
    --models anthropic/claude-haiku-4-5-20251001 \
    --temperature 0.7 --seed 20260425 \
    --output "$OUT_K32" \
    --concurrency 4 \
    --max_total_cost 3.0 --max_cost_per_model 3.0 2>&1 | tee -a "$LOG"
done

# Azure mini + nano can finish quickly across all remaining checkers
for c in AndChecker OrChecker; do
  echo "--- K=32 azure mini/nano checker=$c ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source paleka --checker "$c" --K 32 --max_records "$K4_PALEKA_N" \
    --models azure/mini azure/nano \
    --temperature 0.7 --seed 20260425 \
    --output "$OUT_K32" \
    --concurrency 8 \
    --max_total_cost 2.0 --max_cost_per_model 1.0 2>&1 | tee -a "$LOG"
done

echo "--- K=32 azure mini/nano polymarket ---" | tee -a "$LOG"
python3 scripts/run_real_llms.py \
  --source polymarket --K 32 --max_records "$K4_POLY_N" \
  --polymarket_include partition \
  --models azure/mini azure/nano \
  --temperature 0.7 --seed 20260425 \
  --output "$OUT_K32" \
  --concurrency 8 \
  --max_total_cost 2.0 --max_cost_per_model 1.0 2>&1 | tee -a "$LOG"

# Anthropic polymarket K=32 (small N so this should be quick even at concurrency=4)
echo "--- K=32 anthropic polymarket (concurrency=4) ---" | tee -a "$LOG"
python3 scripts/run_real_llms.py \
  --source polymarket --K 32 --max_records "$K4_POLY_N" \
  --polymarket_include partition \
  --models anthropic/claude-haiku-4-5-20251001 \
  --temperature 0.7 --seed 20260425 \
  --output "$OUT_K32" \
  --concurrency 4 \
  --max_total_cost 2.0 --max_cost_per_model 2.0 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== best_try API runs DONE ===" | tee -a "$LOG"
