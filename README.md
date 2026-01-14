# Minimal SFT + GRPO Repo (Type-Agg + Profile Evolution)

This repo is the minimal, runnable workspace for the **type-aggregated pipeline** (no mgrno):
- build type-level datasets
- train SFT
- evolve profiles (optional)
- train GRPO

It assumes you install `ms-swift` via pip (no bundled `ms-swift/` here).

## Project layout (core)

- `data/raw/Data1.parquet`  
  Raw quarterly holdings data.
- `data/processed/panel_quarter_full.parquet/`  
  Per-type panels with mgrno (intermediate).
- `data/processed/panel_quarter_full_with_stock.parquet/`  
  Per-type panels with stock_* features.
- `data/processed/type_agg/`  
  Type-aggregated panels (type, permno, date).
- `data/processed/type_agg_stock_yf/`  
  Type-aggregated panels with stock_* features (training input).
- `artifacts/features/type_profile_semantics.json`  
  Per-type profile semantics injected into prompts.
- `data/grpo_profile_stats.csv`  
  Profile-deviation stats used by GRPO reward.
- `scripts/sft.sh` / `scripts/grpo.sh`  
  Training wrappers (SFT/GRPO).

## Setup

```bash
pip install -r requirements.txt
pip install ms-swift
```

Optional (only if you use DeepSeek think generation):
```bash
export DEEPSEEK_API_KEY=your_key
```

## End-to-end pipeline (type-agg + profile evolution)

### 1) Raw Data1 -> per-type panels (mgrno, intermediate)
```bash
PYTHONPATH=. python scripts/convert_data1_panel.py \
  --input data/raw/Data1.parquet \
  --out-dir data/processed/panel_quarter_full.parquet
```

### 2) Add stock_* features on per-type panels (yfinance)
```bash
PYTHONPATH=. python scripts/recompute_stock_quarter_features_yf.py \
  --panel-dir data/processed/panel_quarter_full.parquet \
  --out-dir data/processed/panel_quarter_full_with_stock.parquet \
  --ticker-mapping data/ticker_mapping.csv \
  --sp500-weights data/sp500_with_weights.csv \
  --refresh-daily
```

### 3) Aggregate to type-level panels (type, permno, date)
```bash
PYTHONPATH=. python scripts/aggregate_type_permno_panel.py \
  --in-dir data/processed/panel_quarter_full_with_stock.parquet \
  --out-dir data/processed/type_agg \
  --spx-weights-path data/sp500_with_weights.csv \
  --ticker-mapping-path data/ticker_mapping.csv \
  --sp500-only
```

### 4) Fill stock_* on type-agg panels (optional but recommended)
```bash
PYTHONPATH=. python scripts/recompute_stock_quarter_features_yf.py \
  --panel-dir data/processed/type_agg \
  --out-dir data/processed/type_agg_stock_yf \
  --ticker-mapping data/ticker_mapping.csv \
  --sp500-weights data/sp500_with_weights.csv \
  --fill-only
```

### 5) Build SFT/GRPO/Test (SFT uses neutral profile)
```bash
PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir data/processed/type_agg_stock_yf \
  --out-root artifacts_typeagg_all \
  --sft-profile-mode neutral \
  --sft-end 2017-12-31 \
  --grpo-start 2018-01-01 \
  --grpo-end 2022-12-31 \
  --test-start 2023-01-01 \
  --sft-limit 1000 \
  --grpo-limit 1000
```

Outputs:
- `artifacts_typeagg_all/sft/sft_train_<type>.jsonl`
- `artifacts_typeagg_all/grpo/grpo_<type>.jsonl`
- `artifacts_typeagg_all/test/test_<type>_all.jsonl`

### 6) SFT training (ms-swift, all types)
```bash
TYPES=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)
for t in "${TYPES[@]}"; do
  bash scripts/sft.sh \
    -m /path/to/Qwen2.5-7B-Instruct \
    -d "artifacts_typeagg_all/sft/sft_train_${t}.jsonl" \
    -o "outputs/sft_${t}"
done
```

### 7) Profile evolution (optional, between SFT and GRPO)
If you skip this step, go directly to GRPO using `artifacts_typeagg_all/grpo/*`.
If you run this, regenerate GRPO/Test in step 8 and train on `artifacts_typeagg_all_v2/grpo/*`.

```bash
PYTHONPATH=. python scripts/profile_evolution.py \
  --test-path artifacts_typeagg_all/test/test_banks_all.jsonl \
  --investor-type banks \
  --eval-size 80 \
  --k-reasoning 4 \
  --temperature 0.4 \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --lora-path outputs/sft_banks \
  --out-dir outputs/profile_evo/banks \
  --write-profile-path artifacts/features/type_profile_semantics.json
```

### 8) Rebuild GRPO/Test with updated profiles (after evolution)
```bash
PYTHONPATH=. python -m src.cli.build_type_datasets_typeagg_all \
  --in-dir data/processed/type_agg_stock_yf \
  --out-root artifacts_typeagg_all_v2 \
  --sft-limit 0 \
  --grpo-limit 1000 \
  --test-limit 0 \
  --sft-end 2017-12-31 \
  --grpo-start 2018-01-01 \
  --grpo-end 2022-12-31 \
  --test-start 2023-01-01
```

### 9) GRPO training (ms-swift, all types)
```bash
TYPES=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)
GRPO_ROOT=artifacts_typeagg_all_v2
for t in "${TYPES[@]}"; do
  PYTHONPATH=. bash scripts/grpo.sh \
    -m /path/to/Qwen2.5-7B-Instruct \
    -d "${GRPO_ROOT}/grpo/grpo_${t}.jsonl" \
    -o "outputs/grpo_${t}" \
    -a "outputs/sft_${t}" \
    -g 2 \
    -l 512
done
```

## GRPO reward functions

Defaults are defined in `scripts/grpo.sh`:
```
contract_holdings  huber_holdings  profile_numeric_deviation
```
No MSE reward is used by default. Adjust `REWARD_FUNCS` / `REWARD_WEIGHTS` in `scripts/grpo.sh` if needed.

