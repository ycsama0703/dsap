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

### 5) Stage A: build SFT only (neutral profile)
Use the script so GRPO is disabled and SFT uses neutral profiles.  
Note: this stage still writes a temporary TEST file, but it is **ignored** later.

```bash
SFT_END=2015-12-31 \
GRPO_START=2016-01-01 \
GRPO_END=2019-12-31 \
TEST_START=2020-01-01 \
SFT_LIMIT=1000 \
GRPO_LIMIT=0 \
SFT_PROFILE_MODE=neutral \
bash scripts/gen_sft_data_typeagg.sh
```

Outputs (used in SFT training):
- `artifacts/typeagg_all/sft/sft_train_<type>.jsonl`

### 6) SFT training (ms-swift, all types)
```bash
TYPES=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)
for t in "${TYPES[@]}"; do
  bash scripts/sft.sh \
    -m /path/to/Qwen2.5-7B-Instruct \
    -d "artifacts/typeagg_all/sft/sft_train_${t}.jsonl" \
    -o "outputs/sft_${t}"
done
```

### 7) Stage B: profile evolution (between SFT and GRPO)
This updates `artifacts/features/type_profile_semantics.json`.  
After this step, you **must** regenerate GRPO + TEST.

```bash
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
SFT_OUT_ROOT=outputs \
EVAL_ROOT=artifacts/typeagg_all \
EVAL_KIND=grpo \
PROFILE_PATH=artifacts/features/type_profile_semantics.json \
OUT_DIR_ROOT=outputs/profile_evo \
bash scripts/evolve_profiles_typeagg.sh
```

### 8) Stage C: rebuild GRPO + TEST with updated profiles
```bash
SFT_END=2015-12-31 \
GRPO_START=2016-01-01 \
GRPO_END=2019-12-31 \
TEST_START=2020-01-01 \
GRPO_LIMIT=1000 \
bash scripts/gen_grpo_test_typeagg.sh
```

### 9) GRPO training (ms-swift, all types)
```bash
TYPES=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)
GRPO_ROOT=artifacts/typeagg_all_v2
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
