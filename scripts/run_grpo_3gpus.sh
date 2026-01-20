#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-}"
GRPO_ROOT="${GRPO_ROOT:-artifacts/typeagg_all_v2}"
LOG_DIR="${LOG_DIR:-outputs/grpo_logs}"
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
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${ROOT_DIR}/${LOG_DIR}"
fi

mkdir -p "${LOG_DIR}"

GPU0_TYPES=(banks insurance_companies)
GPU1_TYPES=(investment_advisors other)
GPU2_TYPES=(mutual_funds pension_funds)

run_group() {
  local gpu="$1"
  shift
  local types=("$@")
  local log_file="${LOG_DIR}/gpu${gpu}.log"

  {
    echo "[gpu${gpu}] start $(date -u +"%F %T")"
    for t in "${types[@]}"; do
      local dataset="${GRPO_ROOT}/grpo/grpo_${t}.jsonl"
      local out_dir="${ROOT_DIR}/outputs/grpo_${t}"
      local adapters="${ROOT_DIR}/outputs/sft_output/${t}"
      if [[ ! -f "${dataset}" ]]; then
        echo "[gpu${gpu}] [skip] missing dataset: ${dataset}"
        continue
      fi
      if [[ ! -d "${adapters}" ]]; then
        echo "[gpu${gpu}] [skip] missing adapters: ${adapters}"
        continue
      fi
      echo "[gpu${gpu}] [run] ${t}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT_DIR}" \
        bash "${ROOT_DIR}/scripts/grpo.sh" \
          -m "${BASE_MODEL}" \
          -d "${dataset}" \
          -o "${out_dir}" \
          -a "${adapters}" \
          -g 8 \
          -l 512
    done
    echo "[gpu${gpu}] done $(date -u +"%F %T")"
  } >"${log_file}" 2>&1
}

run_group 0 "${GPU0_TYPES[@]}" &
pid0=$!
run_group 1 "${GPU1_TYPES[@]}" &
pid1=$!
run_group 2 "${GPU2_TYPES[@]}" &
pid2=$!

wait "${pid0}" "${pid1}" "${pid2}"
echo "[done] logs: ${LOG_DIR}/gpu{0,1,2}.log"
