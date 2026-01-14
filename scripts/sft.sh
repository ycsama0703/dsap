#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/sft.sh -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR>

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="artifacts/sft/sft_train.jsonl"
OUTPUT_DIR="outputs/sft_qwen2.5_7b"

# Optional logging targets (can also be set via env, e.g. SWIFT_REPORT_TO=swanlab)
REPORT_TO="${SWIFT_REPORT_TO:-}"
SWANLAB_PROJECT="${SWIFT_SWANLAB_PROJECT:-}"
SWANLAB_TOKEN="${SWIFT_SWANLAB_TOKEN:-}"
SWANLAB_WORKSPACE="${SWIFT_SWANLAB_WORKSPACE:-}"
SWANLAB_EXP_NAME="${SWIFT_SWANLAB_EXP_NAME:-}"
SWANLAB_MODE="${SWIFT_SWANLAB_MODE:-}"

declare -a EXTRA_ARGS=()

while getopts ":m:d:o:" opt; do
  case ${opt} in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *) echo "Usage: $0 -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR>" ; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-to-swanlab)
      REPORT_TO="swanlab"
      shift
      ;;
    --report_to|--report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --swanlab_project|--swanlab-project)
      SWANLAB_PROJECT="$2"
      shift 2
      ;;
    --swanlab_token|--swanlab-token)
      SWANLAB_TOKEN="$2"
      shift 2
      ;;
    --swanlab_workspace|--swanlab-workspace)
      SWANLAB_WORKSPACE="$2"
      shift 2
      ;;
    --swanlab_exp_name|--swanlab-exp-name)
      SWANLAB_EXP_NAME="$2"
      shift 2
      ;;
    --swanlab_mode|--swanlab-mode)
      SWANLAB_MODE="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$REPORT_TO" && "$REPORT_TO" != "none" ]]; then
  EXTRA_ARGS+=(--report_to "$REPORT_TO")
fi
if [[ -n "$SWANLAB_PROJECT" ]]; then
  EXTRA_ARGS+=(--swanlab_project "$SWANLAB_PROJECT")
fi
if [[ -n "$SWANLAB_TOKEN" ]]; then
  EXTRA_ARGS+=(--swanlab_token "$SWANLAB_TOKEN")
fi
if [[ -n "$SWANLAB_WORKSPACE" ]]; then
  EXTRA_ARGS+=(--swanlab_workspace "$SWANLAB_WORKSPACE")
fi
if [[ -n "$SWANLAB_EXP_NAME" ]]; then
  EXTRA_ARGS+=(--swanlab_exp_name "$SWANLAB_EXP_NAME")
fi
if [[ -n "$SWANLAB_MODE" ]]; then
  EXTRA_ARGS+=(--swanlab_mode "$SWANLAB_MODE")
fi

swift sft \
  --model "${MODEL}" \
  --train_type lora \
  --dataset "${DATASET}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --system "You are a quantitative portfolio manager." \
  "${EXTRA_ARGS[@]}"
