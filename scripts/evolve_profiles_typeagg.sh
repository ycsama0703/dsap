#!/usr/bin/env bash
set -euo pipefail

# Stage B: profile evolution after SFT

BASE_MODEL="${BASE_MODEL:-/path/to/Qwen2.5-7B-Instruct}"
SFT_OUT_ROOT="${SFT_OUT_ROOT:-outputs}"
TEST_ROOT="${TEST_ROOT:-artifacts/typeagg_all/test}"
PROFILE_PATH="${PROFILE_PATH:-artifacts/features/type_profile_semantics.json}"
OUT_DIR_ROOT="${OUT_DIR_ROOT:-outputs/profile_evo}"

EVAL_SIZE="${EVAL_SIZE:-80}"
K_REASONING="${K_REASONING:-4}"
TEMPERATURE="${TEMPERATURE:-0.4}"

if [[ -n "${TYPES:-}" ]]; then
  IFS=',' read -r -a TYPE_LIST <<< "${TYPES}"
else
  TYPE_LIST=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)
fi

for t in "${TYPE_LIST[@]}"; do
  TEST_PATH="${TEST_ROOT}/test_${t}_all.jsonl"
  LORA_PATH="${SFT_OUT_ROOT}/sft_${t}"
  OUT_DIR="${OUT_DIR_ROOT}/${t}"

  PYTHONPATH=. python scripts/profile_evolution.py \
    --test-path "$TEST_PATH" \
    --investor-type "$t" \
    --eval-size "$EVAL_SIZE" \
    --k-reasoning "$K_REASONING" \
    --temperature "$TEMPERATURE" \
    --base-model "$BASE_MODEL" \
    --lora-path "$LORA_PATH" \
    --out-dir "$OUT_DIR" \
    --write-profile-path "$PROFILE_PATH"
done
