#!/usr/bin/env bash
set -euo pipefail

# Stage C: rebuild GRPO + TEST after profile evolution

IN_DIR="${IN_DIR:-data/processed/type_agg_stock_yf}"
OUT_ROOT="${OUT_ROOT:-artifacts/typeagg_all_v2}"
SFT_END="${SFT_END:-2015-12-31}"
GRPO_START="${GRPO_START:-2016-01-01}"
GRPO_END="${GRPO_END:-2019-12-31}"
TEST_START="${TEST_START:-2020-01-01}"
GRPO_LIMIT="${GRPO_LIMIT:-1000}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-1}"

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

PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir "$IN_DIR" \
  --out-root "$OUT_ROOT" \
  --sft-end "$SFT_END" \
  --grpo-start "$GRPO_START" \
  --grpo-end "$GRPO_END" \
  --test-start "$TEST_START" \
  --sft-limit 0 \
  --grpo-limit "$GRPO_LIMIT" \
  "${TYPE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
