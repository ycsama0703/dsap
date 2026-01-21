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
TYPE="other"
log_file="${LOG_DIR}/gpu${GPU0}.log"
state_file="${STATE_DIR}/gpu${GPU0}.txt"

update_state() {
  local phase="$1"
  local current="$2"
  echo "phase=${phase} train=${3:-0}/1 test=${4:-0}/1 current=${current}" > "${state_file}"
}

{
  echo "[gpu${GPU0}] start $(date -u +"%F %T")"
  echo "[gpu${GPU0}] grpo_root=${GRPO_ROOT}"
  echo "[gpu${GPU0}] eval_root=${EVAL_ROOT}"
  echo "[gpu${GPU0}] sft_root=${SFT_ROOT}"
  update_state "start" "-" 0 0

  dataset="${GRPO_ROOT}/grpo/grpo_${TYPE}.jsonl"
  out_dir="${ROOT_DIR}/outputs/grpo_${TYPE}"
  adapter_root="${SFT_ROOT}/${TYPE}"
  adapters=$(ls -td "${adapter_root}"/v*/checkpoint-* 2>/dev/null | head -n 1)

  if [[ ! -f "${dataset}" ]]; then
    echo "[gpu${GPU0}] [skip] missing dataset: ${dataset}"
    update_state "train" "${TYPE}" 1 0
  elif [[ -z "${adapters}" || ! -d "${adapters}" ]]; then
    echo "[gpu${GPU0}] [skip] missing adapters under: ${adapter_root}"
    update_state "train" "${TYPE}" 1 0
  else
    echo "[gpu${GPU0}] [train] ${TYPE} adapters=${adapters}"
    update_state "train" "${TYPE}" 0 0
    CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONPATH="${ROOT_DIR}" \
      bash "${ROOT_DIR}/scripts/grpo.sh" \
        -m "${BASE_MODEL}" \
        -d "${dataset}" \
        -o "${out_dir}" \
        -a "${adapters}" \
        -g 8 \
        -l 512
    update_state "train" "${TYPE}" 1 0
  fi

  test_path="${EVAL_ROOT}/test/test_${TYPE}_all.jsonl"
  ckpt=$(ls -td "${ROOT_DIR}/outputs/grpo_${TYPE}/checkpoint-"* 2>/dev/null | head -n 1)
  if [[ ! -f "${test_path}" ]]; then
    echo "[gpu${GPU0}] [skip] missing test set: ${test_path}"
    update_state "test" "${TYPE}" 1 1
  elif [[ -z "${ckpt}" ]]; then
    echo "[gpu${GPU0}] [skip] no GRPO checkpoint for ${TYPE}"
    update_state "test" "${TYPE}" 1 1
  else
    echo "[gpu${GPU0}] [test] ${TYPE} ckpt=${ckpt}"
    update_state "test" "${TYPE}" 1 0
    CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONPATH="${ROOT_DIR}" \
      python "${ROOT_DIR}/scripts/run_eval_strict.py" \
        --test_path "${test_path}" \
        --base_model "${BASE_MODEL}" \
        --lora_path "${ckpt}" \
        --out_dir "${OUT_ROOT}/${TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.0 \
        --no-force-think
    update_state "test" "${TYPE}" 1 1
  fi

  update_state "done" "-" 1 1
  echo "[gpu${GPU0}] done $(date -u +"%F %T")"
} >"${log_file}" 2>&1 &

pid=$!

print_summary() {
  if [[ ! -f "${state_file}" ]]; then
    echo "phase=wait train=0/1 test=0/1 current=-"
    return
  fi
  cat "${state_file}"
}

while true; do
  if kill -0 "${pid}" 2>/dev/null; then
    s0=$(print_summary)
    printf "\r[gpu%s] %s" "${GPU0}" "${s0}"
    sleep "${PROGRESS_INTERVAL}"
  else
    echo
    break
  fi
done

wait "${pid}"
echo "[done] log: ${log_file}"
