#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-}"
GRPO_ROOT="${GRPO_ROOT:-artifacts/typeagg_all_v2}"
EVAL_ROOT="${EVAL_ROOT:-${GRPO_ROOT}}"
OUT_ROOT="${OUT_ROOT:-outputs/test_eval_v1}"
LOG_DIR="${LOG_DIR:-outputs/grpo_test_logs}"
SFT_ROOT="${SFT_ROOT:-outputs/sft_output}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"

if [[ -n "${SWIFT_SWANLAB_TOKEN:-}" ]]; then
  SWIFT_REPORT_TO="${SWIFT_REPORT_TO:-swanlab}"
  SWIFT_SWANLAB_PROJECT="${SWIFT_SWANLAB_PROJECT:-dsap}"
  SWIFT_SWANLAB_WORKSPACE="${SWIFT_SWANLAB_WORKSPACE:-ycsama0703}"
fi

if [[ -z "$BASE_MODEL" ]]; then
  echo "[error] Set BASE_MODEL to your local Qwen model path." >&2
  exit 1
fi

if [[ "$GRPO_ROOT" != /* ]]; then
  GRPO_ROOT="${ROOT_DIR}/${GRPO_ROOT}"
fi
if [[ "$EVAL_ROOT" != /* ]]; then
  EVAL_ROOT="${ROOT_DIR}/${EVAL_ROOT}"
fi
if [[ "$OUT_ROOT" != /* ]]; then
  OUT_ROOT="${ROOT_DIR}/${OUT_ROOT}"
fi
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${ROOT_DIR}/${LOG_DIR}"
fi
if [[ "$SFT_ROOT" != /* ]]; then
  SFT_ROOT="${ROOT_DIR}/${SFT_ROOT}"
fi

mkdir -p "${LOG_DIR}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "${STATE_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

GPU0_TYPES=(banks insurance_companies)
GPU1_TYPES=(investment_advisors other)
GPU2_TYPES=(mutual_funds pension_funds)

run_group() {
  local gpu="$1"
  shift
  local types=("$@")
  local log_file="${LOG_DIR}/gpu${gpu}.log"
  local state_file="${STATE_DIR}/gpu${gpu}.txt"
  local total="${#types[@]}"
  local done_train=0
  local done_test=0

  update_state() {
    local phase="$1"
    local current="$2"
    echo "phase=${phase} train=${done_train}/${total} test=${done_test}/${total} current=${current}" > "${state_file}"
  }

  {
    echo "[gpu${gpu}] start $(date -u +"%F %T")"
    echo "[gpu${gpu}] grpo_root=${GRPO_ROOT}"
    echo "[gpu${gpu}] eval_root=${EVAL_ROOT}"
    echo "[gpu${gpu}] sft_root=${SFT_ROOT}"
    update_state "start" "-"

    for t in "${types[@]}"; do
      update_state "train" "${t}"
      local dataset="${GRPO_ROOT}/grpo/grpo_${t}.jsonl"
      local out_dir="${ROOT_DIR}/outputs/grpo_${t}"
      local adapter_root="${SFT_ROOT}/${t}"
      local adapters
      adapters=$(ls -td "${adapter_root}"/v*/checkpoint-* 2>/dev/null | head -n 1)
      if [[ ! -f "${dataset}" ]]; then
        echo "[gpu${gpu}] [skip] missing dataset: ${dataset}"
        done_train=$((done_train + 1))
        continue
      fi
      if [[ -z "${adapters}" || ! -d "${adapters}" ]]; then
        echo "[gpu${gpu}] [skip] missing adapters under: ${adapter_root}"
        done_train=$((done_train + 1))
        continue
      fi
      echo "[gpu${gpu}] [train] ${t} adapters=${adapters}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT_DIR}" \
        bash "${ROOT_DIR}/scripts/grpo.sh" \
          -m "${BASE_MODEL}" \
          -d "${dataset}" \
          -o "${out_dir}" \
          -a "${adapters}" \
          -g 4 \
          -l 512
      done_train=$((done_train + 1))
      update_state "train" "${t}"
    done

    for t in "${types[@]}"; do
      update_state "test" "${t}"
      local test_path="${EVAL_ROOT}/test/test_${t}_all.jsonl"
      local ckpt
      ckpt=$(ls -td "${ROOT_DIR}/outputs/grpo_${t}/checkpoint-"* 2>/dev/null | head -n 1)
      if [[ ! -f "${test_path}" ]]; then
        echo "[gpu${gpu}] [skip] missing test set: ${test_path}"
        done_test=$((done_test + 1))
        continue
      fi
      if [[ -z "${ckpt}" ]]; then
        echo "[gpu${gpu}] [skip] no GRPO checkpoint for ${t}"
        done_test=$((done_test + 1))
        continue
      fi
      echo "[gpu${gpu}] [test] ${t} ckpt=${ckpt}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT_DIR}" \
        python "${ROOT_DIR}/scripts/run_eval_strict.py" \
          --test_path "${test_path}" \
          --base_model "${BASE_MODEL}" \
          --lora_path "${ckpt}" \
          --out_dir "${OUT_ROOT}/${t}" \
          --batch_size "${BATCH_SIZE}" \
          --max_new_tokens "${MAX_NEW_TOKENS}" \
          --temperature 0.0 \
          --no-force-think
      done_test=$((done_test + 1))
      update_state "test" "${t}"
    done

    update_state "done" "-"
    echo "[gpu${gpu}] done $(date -u +"%F %T")"
  } >"${log_file}" 2>&1
}

run_group "${GPU0}" "${GPU0_TYPES[@]}" &
pid0=$!
run_group "${GPU1}" "${GPU1_TYPES[@]}" &
pid1=$!
run_group "${GPU2}" "${GPU2_TYPES[@]}" &
pid2=$!

print_summary() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "phase=wait train=0/0 test=0/0 current=-"
    return
  fi
  cat "${file}"
}

while true; do
  alive=0
  if kill -0 "${pid0}" 2>/dev/null; then alive=1; fi
  if kill -0 "${pid1}" 2>/dev/null; then alive=1; fi
  if kill -0 "${pid2}" 2>/dev/null; then alive=1; fi

  s0=$(print_summary "${STATE_DIR}/gpu${GPU0}.txt")
  s1=$(print_summary "${STATE_DIR}/gpu${GPU1}.txt")
  s2=$(print_summary "${STATE_DIR}/gpu${GPU2}.txt")
  printf "\r[gpu%s] %s | [gpu%s] %s | [gpu%s] %s" "${GPU0}" "${s0}" "${GPU1}" "${s1}" "${GPU2}" "${s2}"
  if [[ "${alive}" -eq 0 ]]; then
    echo
    break
  fi
  sleep "${PROGRESS_INTERVAL}"
done

wait "${pid0}" "${pid1}" "${pid2}"
echo "[done] logs: ${LOG_DIR}/gpu{${GPU0},${GPU1},${GPU2}}.log"
