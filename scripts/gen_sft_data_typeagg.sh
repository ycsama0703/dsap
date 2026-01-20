#!/usr/bin/env bash
set -euo pipefail

# Stage A: build SFT + TEST (neutral profile), GRPO limit=0

IN_DIR="${IN_DIR:-data/processed/type_agg_stock_yf}"
OUT_ROOT="${OUT_ROOT:-artifacts/typeagg_all}"
SFT_END="${SFT_END:-2015-12-31}"
GRPO_START="${GRPO_START:-2016-01-01}"
GRPO_END="${GRPO_END:-2019-12-31}"
TEST_START="${TEST_START:-2020-01-01}"
SFT_LIMIT="${SFT_LIMIT:-500}"
GRPO_LIMIT="${GRPO_LIMIT:-0}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-1}"
SFT_PROFILE_MODE="${SFT_PROFILE_MODE:-neutral}"

SFT_WITH_THINK="${SFT_WITH_THINK:-1}"
SFT_THINK_SOURCE="${SFT_THINK_SOURCE:-deepseek}"
SFT_THINK_WORKERS="${SFT_THINK_WORKERS:-5}"
SFT_THINK_RETRIES="${SFT_THINK_RETRIES:-3}"
SFT_THINK_BACKOFF="${SFT_THINK_BACKOFF:-1.5}"

TYPE_ARGS=()
if [[ -n "${TYPES:-}" ]]; then
  TYPE_ARGS+=(--types "$TYPES")
fi
if [[ -n "${TYPE_ALIAS:-}" ]]; then
  TYPE_ARGS+=(--type-alias "$TYPE_ALIAS")
fi

EXTRA_ARGS=()
if [[ "$RANDOM_SAMPLE" == "1" ]]; then
  EXTRA_ARGS+=(--random-sample)
fi
if [[ "$SFT_WITH_THINK" == "1" ]]; then
  EXTRA_ARGS+=(
    --sft-with-think
    --sft-think-source "$SFT_THINK_SOURCE"
    --sft-think-workers "$SFT_THINK_WORKERS"
    --sft-think-retries "$SFT_THINK_RETRIES"
    --sft-think-backoff "$SFT_THINK_BACKOFF"
  )
fi

PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir "$IN_DIR" \
  --out-root "$OUT_ROOT" \
  --sft-profile-mode "$SFT_PROFILE_MODE" \
  --sft-end "$SFT_END" \
  --grpo-start "$GRPO_START" \
  --grpo-end "$GRPO_END" \
  --test-start "$TEST_START" \
  --sft-limit "$SFT_LIMIT" \
  --grpo-limit "$GRPO_LIMIT" \
  "${TYPE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
