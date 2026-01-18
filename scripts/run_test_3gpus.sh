#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""

BASE_MODEL="${BASE_MODEL:-}"
EVAL_ROOT="${EVAL_ROOT:-artifacts/typeagg_all_v2}"
OUT_ROOT="${OUT_ROOT:-outputs/test_eval_v1}"
LOG_DIR="${OUT_ROOT}/logs"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ -z "$BASE_MODEL" ]]; then
  echo "[error] Set BASE_MODEL to your local Qwen model path." >&2
  exit 1
fi

GPU0_TYPES=(banks insurance_companies)
GPU1_TYPES=(investment_advisors households other)
GPU2_TYPES=(mutual_funds pension_funds)

mkdir -p "$LOG_DIR"

run_group() {
  local gpu="$1"
  local log_path="$2"
  shift 2
  local types=("$@")

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for t in "${types[@]}"; do
      TEST_PATH="${EVAL_ROOT}/test/test_${t}_all.jsonl"
      CKPT=$(ls -td outputs/grpo_${t}/checkpoint-* 2>/dev/null | head -n 1)
      if [[ ! -f "$TEST_PATH" ]]; then
        echo "[skip] missing test set for $t: $TEST_PATH"
        continue
      fi
      if [[ -z "$CKPT" ]]; then
        echo "[skip] no GRPO checkpoint for $t"
        continue
      fi

      PYTHONPATH=. python scripts/run_eval_strict.py \
        --test_path "$TEST_PATH" \
        --base_model "$BASE_MODEL" \
        --lora_path "$CKPT" \
        --out_dir "${OUT_ROOT}/${t}" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --no-force-think
    done
  ) >"$log_path" 2>&1 &
  echo "[start] gpu${gpu} log=${log_path}"
}

PIDS=()
run_group 0 "${LOG_DIR}/gpu0.log" "${GPU0_TYPES[@]}"
PIDS+=($!)
run_group 1 "${LOG_DIR}/gpu1.log" "${GPU1_TYPES[@]}"
PIDS+=($!)
run_group 2 "${LOG_DIR}/gpu2.log" "${GPU2_TYPES[@]}"
PIDS+=($!)

wait
printf "\n[done] test eval finished. logs in %s\n" "$LOG_DIR"
