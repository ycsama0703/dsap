# Minimal SFT + GRPO Repo

This repo is a minimal, runnable workspace for:
- building datasets from type-aggregated panel data
- training SFT and GRPO with ms-swift
- running eval and price-metric aggregation

It assumes you will install `ms-swift` via pip (no bundled `ms-swift/` here).

## Project layout

- `data/processed/type_agg_stock_yf/`  
  Type-aggregated panel parquet files (input for dataset building).
- `data/*.csv`  
  Market features and mappings used by the pipeline.
- `artifacts/features/type_profile_semantics.json`  
  Per-type profile semantics injected into prompts.
- `src/cli/`  
  Dataset pipeline scripts (SFT/GRPO/Test).
- `src/plugins/grpo/holdings_plugin.py`  
  Custom reward functions for GRPO.
- `scripts/debug_eval_outputs.py`  
  Run inference on test set and dump raw outputs.
- `scripts/build_price_metrics_pipeline.py`  
  Aggregate per-type outputs into price metrics.
- `outputs/sp500_eval/metrics/final_groups_5x5.csv`  
  Final 5x5 stock list for evaluation.
- `profile_scripts/`  
  Minimal profile regeneration scripts (type-level only).

## Setup

```bash
pip install -r requirements.txt
pip install ms-swift
```

Optional (only if you use DeepSeek think generation):
```bash
export DEEPSEEK_API_KEY=your_key
```

## Data pipeline (build SFT/GRPO/Test)

This generates JSONL datasets from the panel files. Adjust the date boundaries as needed.

```bash
PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir data/processed/type_agg_stock_yf \
  --out-root artifacts_typeagg_all \
  --sft-limit 1000 \
  --grpo-limit 1000 \
  --sft-end 2017-12-31 \
  --grpo-start 2018-01-01 \
  --grpo-end 2022-12-31 \
  --test-start 2023-01-01
```

Outputs:
- `artifacts_typeagg_all/sft/sft_train_<type>.jsonl`
- `artifacts_typeagg_all/grpo/grpo_<type>.jsonl`
- `artifacts_typeagg_all/test/test_<type>_all.jsonl`

If you want Think generation for SFT (slow, requires DeepSeek):
```bash
PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir data/processed/type_agg_stock_yf \
  --out-root artifacts_typeagg_all \
  --sft-with-think \
  --sft-think-source deepseek \
  --sft-think-strict
```

## SFT training (ms-swift)

Use your standard ms-swift CLI/config. The dataset should point to the generated SFT JSONL.

Example (fill in your usual training args):
```bash
python -m swift.cli.sft \
  --dataset artifacts_typeagg_all/sft/sft_train_banks.jsonl \
  --output_dir outputs/sft_banks \
  --model <your_base_model> \
  --train_type lora
```

## GRPO training (ms-swift)

Make sure the custom reward plugin is importable (this repo is on `PYTHONPATH`).

Example (fill in your usual GRPO args):
```bash
PYTHONPATH=. python -m swift.cli.rlhf \
  --rlhf_type grpo \
  --dataset artifacts_typeagg_all/grpo/grpo_banks.jsonl \
  --output_dir outputs/grpo_banks \
  --model <your_base_model>
```

The GRPO reward functions are defined in:
- `src/plugins/grpo/holdings_plugin.py`

## Evaluation

1) Run inference and dump raw outputs:
```bash
PYTHONPATH=. python scripts/debug_eval_outputs.py \
  --test-path artifacts_typeagg_all/test/test_banks_all.jsonl \
  --base-model <your_base_model> \
  --lora-path <your_checkpoint> \
  --out-csv outputs/test_banks_grpo.csv
```

2) Aggregate price metrics:
```bash
PYTHONPATH=. python scripts/build_price_metrics_pipeline.py \
  --pred-dir outputs \
  --test-dir artifacts_typeagg_all/test \
  --out-dir outputs/metrics_run \
  --pred-suffix grpo
```

Use `outputs/sp500_eval/metrics/final_groups_5x5.csv` to filter evaluation if needed.

## Profile generation (optional, minimal)

If you need to regenerate type-level profile semantics (instead of using the
existing `artifacts/features/type_profile_semantics.json`), only these two
scripts are kept:

```bash
# 1) Compute type-level weights
PYTHONPATH=. python profile_scripts/compute_type_profiles.py \
  --panel-dir data/processed/type_agg_stock_yf \
  --stock-daily-path data/stock_daily.parquet \
  --out-path artifacts/features/type_profiles.csv

# 2) Generate type-level semantics JSON
PYTHONPATH=. python profile_scripts/generate_type_profile_semantics.py \
  --weights artifacts/features/type_profiles.csv \
  --out-json artifacts/features/type_profile_semantics.json
```
