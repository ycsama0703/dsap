#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/grpo.sh -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> \
#        [-a <SFT_ADAPTER_DIR>] [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-r <CKPT_DIR>] [-v] \
#        [-b <PER_DEVICE_BATCH>] [-A <GRAD_ACCUM>] [-R <LORA_RANK>] [-L <LORA_ALPHA>] \
#        [-S "</answer>"|"}"] [-T 0.9]
#   -a: initialize from existing SFT LoRA adapters (e.g., outputs/sft_*)
#   -r: resume GRPO from a checkpoint dir (e.g., outputs/grpo_*/checkpoint-1000)
#   -v: use vLLM colocate mode
#   -F: reward funcs (space-separated, order matters)
#   -W: reward weights (space-separated, align with -F)
#   -S: stop words (single string)
#   -T: temperature (float)

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="artifacts/grpo/grpo.jsonl"
OUTPUT_DIR="outputs/grpo_qwen2.5_7b"
NUM_GENERATIONS=4
MAX_COMPLETION_LEN=512
NUM_TRAIN_EPOCHS=1
USE_VLLM=0
ADAPTERS=""
RESUME_FROM=""
TEMPERATURE=0.9
STOP_WORDS=""
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUM_STEPS=4
LORA_RANK=32
LORA_ALPHA=128
# 默认使用：格式 + 数值 + Profile 约束（方向奖励关闭）
# 可按需在命令行用 -F/-W 覆盖
REWARD_FUNCS=(contract_holdings huber_holdings profile_numeric_deviation)
REWARD_WEIGHTS=(0.1 0.4 0.5)

# Optional logging targets (can also be supplied via environment variables, e.g. SWIFT_REPORT_TO=swanlab)
REPORT_TO="${SWIFT_REPORT_TO:-}"
SWANLAB_PROJECT="${SWIFT_SWANLAB_PROJECT:-}"
SWANLAB_TOKEN="${SWIFT_SWANLAB_TOKEN:-}"
SWANLAB_WORKSPACE="${SWIFT_SWANLAB_WORKSPACE:-}"
SWANLAB_EXP_NAME="${SWIFT_SWANLAB_EXP_NAME:-}"
SWANLAB_MODE="${SWIFT_SWANLAB_MODE:-}"

declare -a EXTRA_ARGS=()

while getopts ":m:d:o:g:l:a:r:vS:T:b:A:R:L:" opt; do
  case ${opt} in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) NUM_GENERATIONS="$OPTARG" ;;
    l) MAX_COMPLETION_LEN="$OPTARG" ;;
    a) ADAPTERS="$OPTARG" ;;
    r) RESUME_FROM="$OPTARG" ;;
    v) USE_VLLM=1 ;;
    S) STOP_WORDS="$OPTARG" ;;
    T) TEMPERATURE="$OPTARG" ;;
    b) PER_DEVICE_TRAIN_BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUM_STEPS="$OPTARG" ;;
    R) LORA_RANK="$OPTARG" ;;
    L) LORA_ALPHA="$OPTARG" ;;
    *) echo "Usage: $0 -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> [-a <SFT_ADAPTER_DIR>] [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-r <CKPT_DIR>] [-v] [-b <PER_DEVICE_BATCH>] [-A <GRAD_ACCUM>] [-R <LORA_RANK>] [-L <LORA_ALPHA>] [-S <stop words>] [-T <temperature>]" ; exit 1 ;;
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

EXTRA_VLLM_ARGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  EXTRA_VLLM_ARGS+=(--use_vllm true --vllm_mode colocate)
fi

ADAPTER_ARGS=()
if [[ -n "$ADAPTERS" ]]; then
  ADAPTER_ARGS+=(--adapters "$ADAPTERS")
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi

# Auto-detect latest checkpoint if none provided and output dir has checkpoints
if [[ -z "$RESUME_FROM" ]]; then
  latest_ckpt=""
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    latest_ckpt=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sed -E 's/.*checkpoint-([0-9]+)/\1 \0/' | sort -k1,1n | awk '{print $2}' | tail -1)
  fi
  if [[ -n "$latest_ckpt" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$latest_ckpt")
  fi
fi

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

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --external_plugins src/plugins/grpo/holdings_plugin.py \
  --reward_funcs ${REWARD_FUNCS[@]} \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --dataset "${DATASET}" \
  --load_from_cache_file true \
  --max_completion_length "${MAX_COMPLETION_LEN}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps "${GRADIENT_ACCUM_STEPS}" \
  --logging_steps 5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 2 \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --beta 0.02 \
  --log_completions true \
  --reward_weights ${REWARD_WEIGHTS[@]} \
  $( [[ -n "$STOP_WORDS" ]] && echo --stop_words "$STOP_WORDS" ) \
  "${EXTRA_ARGS[@]}" \
  "${EXTRA_VLLM_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
