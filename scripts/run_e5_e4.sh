#!/usr/bin/env bash
# Resolve repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Phase 1 follow-ups: E5 (same-model 4-seed) and E4 (K-sweep K=32).
#
# Budget: <$50 total per user constraint. Conservative caps below totalling
# <$30 in expectation given probe rate ($0.00023/call on Haiku, similar order
# on Azure mini/nano, ~3x on Groq Llama-70b).

set -euo pipefail
cd "${JCD_ROOT:?Set JCD_ROOT to your JCD-Forecasting clone}"
export PYTHONPATH=src

OUT_E5=$REPO_ROOT/data/results/e5_samemodel
OUT_E4=$REPO_ROOT/data/results/e4_ksweep
mkdir -p "$OUT_E5" "$OUT_E4"

LOG=$REPO_ROOT/data/results/run_e5_e4.log
: > "$LOG"

echo "=== E5: claude-haiku-4-5, 4 fresh seeds × Paleka+Polymarket ===" | tee -a "$LOG"

SEEDS=(20260501 20260502 20260503 20260504)
PALEKA_CHECKERS=(NegChecker AndChecker OrChecker)
PALEKA_N=80
POLY_N=40

for s in "${SEEDS[@]}"; do
  for c in "${PALEKA_CHECKERS[@]}"; do
    echo "--- E5 seed=$s checker=$c ---" | tee -a "$LOG"
    python3 scripts/run_real_llms.py \
      --source paleka --checker "$c" --K 8 --max_records "$PALEKA_N" \
      --models anthropic/claude-haiku-4-5-20251001 \
      --temperature 0.7 --seed "$s" \
      --output "$OUT_E5/seed_$s" \
      --concurrency 8 \
      --max_total_cost 2.0 --max_cost_per_model 2.0 2>&1 | tee -a "$LOG"
  done
  echo "--- E5 seed=$s polymarket partition ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source polymarket --K 8 --max_records "$POLY_N" \
    --polymarket_include partition \
    --models anthropic/claude-haiku-4-5-20251001 \
    --temperature 0.7 --seed "$s" \
    --output "$OUT_E5/seed_$s" \
    --concurrency 8 \
    --max_total_cost 2.0 --max_cost_per_model 2.0 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== E4: K=32 sweep, 4 models × subset ===" | tee -a "$LOG"

K4_PALEKA_N=40
K4_POLY_N=20

for c in "${PALEKA_CHECKERS[@]}"; do
  echo "--- E4 K=32 checker=$c ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source paleka --checker "$c" --K 32 --max_records "$K4_PALEKA_N" \
    --models anthropic/claude-haiku-4-5-20251001 azure/mini azure/nano \
    --temperature 0.7 --seed 20260425 \
    --output "$OUT_E4" \
    --concurrency 8 \
    --max_total_cost 6.0 --max_cost_per_model 2.0 2>&1 | tee -a "$LOG"
done

echo "--- E4 K=32 polymarket partition ---" | tee -a "$LOG"
python3 scripts/run_real_llms.py \
  --source polymarket --K 32 --max_records "$K4_POLY_N" \
  --polymarket_include partition \
  --models anthropic/claude-haiku-4-5-20251001 azure/mini azure/nano groq/llama-3.3-70b-versatile \
  --temperature 0.7 --seed 20260425 \
  --output "$OUT_E4" \
  --concurrency 8 \
  --max_total_cost 4.0 --max_cost_per_model 1.5 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
