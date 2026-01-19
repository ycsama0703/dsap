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
    -o "outputs/sft_output/${t}"
done
```

### 7) Stage B: profile evolution (between SFT and GRPO)
This updates `artifacts/features/type_profile_semantics.json`.  
After this step, you **must** regenerate GRPO + TEST.

Evolution logic (current):
- Each generation evaluates **all candidate profiles** on a **batch of `eval-size` samples**.
- For each sample, we take the **best reward across profiles**.
- The main curve uses the **cumulative mean** of those per-sample best rewards (sample-axis plot).

```bash
export CUDA_VISIBLE_DEVICES=0,1
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct
EVAL_ROOT=artifacts/typeagg_all_v2
TYPES=(banks households insurance_companies investment_advisors mutual_funds pension_funds other)

for t in "${TYPES[@]}"; do
  CKPT=$(ls -td outputs/sft_output/${t}/v*/checkpoint-* 2>/dev/null | head -n 1)
  # Example: outputs/sft_output/banks/v0-20260116-151226/checkpoint-64
  if [[ -z "$CKPT" ]]; then
    echo "[skip] no checkpoint for $t"
    continue
  fi

  PYTHONPATH=. python scripts/profile_evolution.py \
    --test-path "${EVAL_ROOT}/grpo/grpo_${t}.jsonl" \
    --investor-type "$t" \
    --base-model "$BASE_MODEL" \
    --lora-path "$CKPT" \
    --llm-guide \
    --llm-only \
    --llm-candidates 3 \
    --generations 3 \
    --population-size 4 \
    --eval-size 40 \
    --k-reasoning 2 \
    --temperature 0.0 \
    --max-new-tokens 256 \
    --out-dir "outputs/profile_evo_exp/${t}" \
    --write-profile-path artifacts/features/type_profile_semantics.json
done
```

#### 7.1) Evolution plots (sample-axis)
Per-type outputs (inside each `--out-dir`):
- `best_value_reward.png`  
  Cumulative mean of **per-sample best reward** (x-axis = samples evaluated).
- `best_value_reward_by_gen.png`  
  Legacy per-generation best reward plot.

To rebuild plots from logs:
```bash
# single type
python scripts/plot_sample_axis_from_debug.py \
  --out-dir outputs/profile_evo_exp/banks \
  --mode best

# all types in one figure
python scripts/plot_sample_axis_from_debug.py \
  --root outputs/profile_evo_exp \
  --mode best
```

To compare before/after rewards across types:
```bash
python scripts/plot_profile_evo_summary.py \
  --root outputs/profile_evo_exp
```

#### 7.2) Merge per-GPU profiles (only if you used multi-GPU evolution)
If you ran evolution with per-GPU profile outputs (e.g. `type_profile_semantics_gpu0.json`),
merge them back into the default profile file **before** rebuilding GRPO/TEST:

```bash
python scripts/merge_profile_outputs.py
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
    -a "outputs/sft_output/${t}" \
    -g 2 \
    -l 512
done
```

#### 9.1) Optional: 3-GPU GRPO training (same types split)
```bash
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
bash scripts/run_grpo_3gpus.sh
```

#### 10) Optional: 3-GPU test evaluation (per-type GRPO checkpoints)
```bash
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
bash scripts/run_test_3gpus.sh
```

## GRPO reward functions

Defaults are defined in `scripts/grpo.sh`:
```
contract_holdings  huber_holdings  profile_numeric_deviation
```
No MSE reward is used by default. Adjust `REWARD_FUNCS` / `REWARD_WEIGHTS` in `scripts/grpo.sh` if needed.
