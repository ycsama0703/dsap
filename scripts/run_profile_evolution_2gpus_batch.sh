#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""

BASE_MODEL=/home/macro-econ/YuncongLiu/models/Qwen2.5-7B-Instruct
EVAL_ROOT=artifacts/typeagg_all_v2
OUT_ROOT=outputs/profile_evo_exp_v5_batch
PROFILE_INIT=artifacts/features/type_profile_semantics_init.json
LOG_DIR="${OUT_ROOT}/logs"
EVAL_SIZE=100
GENERATIONS=10

GPU0_TYPES=(banks insurance_companies investment_advisors)
GPU1_TYPES=(mutual_funds pension_funds other)

mkdir -p "$LOG_DIR"

run_group() {
  local gpu="$1"
  local profile_out="$2"
  local log_path="$3"
  shift 3
  local types=("$@")

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for t in "${types[@]}"; do
      CKPT=$(ls -td outputs/sft_output/${t}/v*/checkpoint-* 2>/dev/null | head -n 1)
      if [[ -z "$CKPT" ]]; then
        echo "[skip] no checkpoint for $t"
        continue
      fi

      PYTHONPATH=. python scripts/profile_evolution_batch.py \
        --test-path "${EVAL_ROOT}/grpo/grpo_${t}.jsonl" \
        --investor-type "$t" \
        --profile-path "$PROFILE_INIT" \
        --base-model "$BASE_MODEL" \
        --lora-path "$CKPT" \
        --llm-guide \
        --llm-candidates 4 \
        --llm-only \
        --llm-max-step 0.05 \
        --population-size 10 \
        --parents 2 \
        --children-per-parent 4 \
        --eval-size "$EVAL_SIZE" \
        --generations "$GENERATIONS" \
        --k-reasoning 1 \
        --temperature 0.0 \
        --max-new-tokens 512 \
        --batch-size 32 \
        --out-dir "${OUT_ROOT}/${t}" \
        --write-profile-path "$profile_out"
    done
  ) >"$log_path" 2>&1 &
  echo "[start] gpu${gpu} log=${log_path}"
}

PIDS=()
run_group 0 artifacts/features/type_profile_semantics_gpu0.json "${LOG_DIR}/gpu0.log" "${GPU0_TYPES[@]}"
PIDS+=($!)
run_group 1 artifacts/features/type_profile_semantics_gpu1.json "${LOG_DIR}/gpu1.log" "${GPU1_TYPES[@]}"
PIDS+=($!)

progress_count() {
  local t="$1"
  local prog="${OUT_ROOT}/${t}/progress.jsonl"
  if [[ -f "$prog" ]]; then
    wc -l <"$prog" | tr -d ' '
  else
    echo 0
  fi
}

group_done() {
  local sum=0
  local t
  for t in "$@"; do
    sum=$((sum + $(progress_count "$t")))
  done
  echo "$sum"
}

group_total() {
  local count=$#
  echo $((count * GENERATIONS))
}

while :; do
  alive=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    break
  fi

  g0_done=$(group_done "${GPU0_TYPES[@]}")
  g1_done=$(group_done "${GPU1_TYPES[@]}")
  g0_total=$(group_total "${GPU0_TYPES[@]}")
  g1_total=$(group_total "${GPU1_TYPES[@]}")
  total_done=$((g0_done + g1_done))
  total_total=$((g0_total + g1_total))

  printf "\r[progress] total %d/%d | gpu0 %d/%d | gpu1 %d/%d" \
    "$total_done" "$total_total" \
    "$g0_done" "$g0_total" \
    "$g1_done" "$g1_total"
  sleep 60
done

wait
printf "\n[done] all groups finished\n"
