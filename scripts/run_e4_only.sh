#!/usr/bin/env bash
# Resolve repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Restart E4 only (E5 already complete). Drop Groq for E4 due to TPD limit.
# anthropic/mini/nano NegChecker K=32 results from prior run are cached and
# will be reused.

set -euo pipefail
cd "${JCD_ROOT:?Set JCD_ROOT to your JCD-Forecasting clone}"
export PYTHONPATH=src

OUT_E4=$REPO_ROOT/data/results/e4_ksweep
mkdir -p "$OUT_E4"

LOG=$REPO_ROOT/data/results/run_e4_only.log
: > "$LOG"

K4_PALEKA_N=40
K4_POLY_N=20
PALEKA_CHECKERS=(NegChecker AndChecker OrChecker)

for c in "${PALEKA_CHECKERS[@]}"; do
  echo "--- E4 K=32 checker=$c ---" | tee -a "$LOG"
  python3 scripts/run_real_llms.py \
    --source paleka --checker "$c" --K 32 --max_records "$K4_PALEKA_N" \
    --models anthropic/claude-haiku-4-5-20251001 azure/mini azure/nano \
    --temperature 0.7 --seed 20260425 \
    --output "$OUT_E4" \
    --concurrency 8 \
    --max_total_cost 6.0 --max_cost_per_model 2.5 2>&1 | tee -a "$LOG"
done

echo "--- E4 K=32 polymarket partition ---" | tee -a "$LOG"
python3 scripts/run_real_llms.py \
  --source polymarket --K 32 --max_records "$K4_POLY_N" \
  --polymarket_include partition \
  --models anthropic/claude-haiku-4-5-20251001 azure/mini azure/nano \
  --temperature 0.7 --seed 20260425 \
  --output "$OUT_E4" \
  --concurrency 8 \
  --max_total_cost 4.0 --max_cost_per_model 1.5 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== E4 DONE (3 models: anthropic/mini/nano; groq excluded due to TPD limit) ===" | tee -a "$LOG"
