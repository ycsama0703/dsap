from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.cli.build_history_prompts import build_for_file
from src.cli.build_type_datasets import (
    _build_system_prompt,
    _convert_prompts_to_grpo,
    _convert_prompts_to_sft,
    _load_market_quarterly_safe,
)
from src.cli.build_type_datasets_typeagg import (
    _convert_prompts_to_test_optional_label,
    _enrich_prompts_with_deepseek,
)

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


def discover_types(in_dir: Path) -> List[str]:
    names = []
    for p in sorted(in_dir.glob("*.parquet")):
        if p.name == "all_investors.parquet":
            continue
        names.append(p.stem)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT/GRPO datasets for all type-agg panels")
    ap.add_argument("--in-dir", type=Path, default=Path("data/processed/type_agg"),
                    help="Input directory with type-aggregated parquet files")
    ap.add_argument("--out-root", type=Path, default=Path("artifacts_typeagg_all"),
                    help="Root output directory")
    ap.add_argument("--types", type=str, default="",
                    help="Comma-separated type names; empty = auto-discover from --in-dir")
    ap.add_argument("--sft-limit", type=int, default=1000)
    ap.add_argument("--grpo-limit", type=int, default=1000)
    ap.add_argument("--sft-end", type=str, default="2017-12-31")
    ap.add_argument("--grpo-start", type=str, default="2018-01-01")
    ap.add_argument("--grpo-end", type=str, default="2022-12-31")
    ap.add_argument("--test-start", type=str, default="2023-01-01")
    ap.add_argument("--history-len", type=int, choices=[1, 2], default=1)
    ap.add_argument("--time-bins", type=int, default=10)
    ap.add_argument("--cap-per-pair", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-zero", action="store_true",
                    help="Exclude windows where holding_t == 0")
    ap.add_argument("--random-sample", action="store_true",
                    help="Use simple random sampling instead of stratified sampling.")
    ap.add_argument("--sft-with-think", action="store_true", default=False)
    ap.add_argument("--sft-think-source", choices=["placeholder", "deepseek"], default="placeholder")
    ap.add_argument("--sft-think-strict", action="store_true")
    ap.add_argument("--sft-think-workers", type=int, default=1,
                    help="DeepSeek think parallel workers (default: 1)")
    ap.add_argument("--sft-think-retries", type=int, default=3,
                    help="DeepSeek retry count on connection errors (default: 3)")
    ap.add_argument("--sft-think-backoff", type=float, default=1.5,
                    help="Base backoff seconds for retries (default: 1.5)")
    ap.add_argument("--grpo-no-think-example", action="store_true",
                    help="Disable think/answer placeholder example in GRPO dataset")
    ap.add_argument("--emit-base-min-test", action="store_true",
                    help="Also emit base-min test set (single JSON output, no think).")
    args = ap.parse_args()

    if not args.in_dir.exists():
        raise FileNotFoundError(f"input directory not found: {args.in_dir}")

    if args.types.strip():
        types = [s.strip() for s in args.types.split(",") if s.strip()]
    else:
        types = discover_types(args.in_dir)
        if not types:
            raise FileNotFoundError(f"no parquet files in {args.in_dir}")

    mapping = None
    try:
        if load_mapping is not None:
            mp_path = Path("data/ticker_mapping.csv")
            if mp_path.exists():
                mapping = load_mapping(mp_path)
                print(f"[typeagg-all] loaded mapping: {len(mapping)}")
    except Exception as e:
        print(f"[typeagg-all] mapping load failed: {e}")

    market_df = _load_market_quarterly_safe()

    out_root = args.out_root
    prompts_sft_dir = out_root / "prompts_hist_sft"
    prompts_grpo_dir = out_root / "prompts_hist_grpo"
    prompts_test_dir = out_root / "prompts_hist_test"
    prompts_sft_dir.mkdir(parents=True, exist_ok=True)
    prompts_grpo_dir.mkdir(parents=True, exist_ok=True)
    prompts_test_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "sft").mkdir(parents=True, exist_ok=True)
    (out_root / "grpo").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    total = {"sft": 0, "grpo": 0, "test": 0}

    for t in types:
        in_file = args.in_dir / f"{t}.parquet"
        if not in_file.exists():
            print(f"[typeagg-all] skip type={t}: {in_file} not found")
            continue

        ph_sft = prompts_sft_dir / f"{t}.jsonl"
        ph_grpo = prompts_grpo_dir / f"{t}.jsonl"
        ph_test = prompts_test_dir / f"{t}.jsonl"

        print(f"[typeagg-all] {t}: build SFT prompts (<= {args.sft_end})")
        total["sft"] += build_for_file(
            in_file=in_file,
            out_file=ph_sft,
            per_type_limit=args.sft_limit,
            time_bins=args.time_bins,
            cap_per_pair=args.cap_per_pair,
            seed=args.seed,
            history_len=args.history_len,
            date_start=None,
            date_end=args.sft_end,
            progress_every=50000,
            use_tqdm=False,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero,
            include_permnos=None,
            take_all=False,
            limit_override=args.sft_limit,
            profile_dir=None,
            profile_weights_dir=None,
            random_sample=args.random_sample,
            market_df=market_df,
            require_holding_t1=True,
        )

        print(f"[typeagg-all] {t}: build GRPO prompts ({args.grpo_start}..{args.grpo_end})")
        total["grpo"] += build_for_file(
            in_file=in_file,
            out_file=ph_grpo,
            per_type_limit=args.grpo_limit,
            time_bins=args.time_bins,
            cap_per_pair=args.cap_per_pair,
            seed=args.seed,
            history_len=args.history_len,
            date_start=args.grpo_start,
            date_end=args.grpo_end,
            progress_every=50000,
            use_tqdm=False,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero,
            include_permnos=None,
            take_all=False,
            limit_override=args.grpo_limit,
            profile_dir=None,
            profile_weights_dir=None,
            random_sample=args.random_sample,
            market_df=market_df,
            require_holding_t1=True,
        )

        print(f"[typeagg-all] {t}: build TEST prompts (>= {args.test_start})")
        total["test"] += build_for_file(
            in_file=in_file,
            out_file=ph_test,
            per_type_limit=args.sft_limit,
            time_bins=args.time_bins,
            cap_per_pair=args.cap_per_pair,
            seed=args.seed,
            history_len=args.history_len,
            date_start=args.test_start,
            date_end=None,
            progress_every=50000,
            use_tqdm=False,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero,
            include_permnos=None,
            take_all=True,
            limit_override=None,
            profile_dir=None,
            profile_weights_dir=None,
            random_sample=args.random_sample,
            market_df=market_df,
            require_holding_t1=False,
        )

        system_prompt = _build_system_prompt(t)

        sft_input = ph_sft
        if args.sft_with_think and args.sft_think_source == "deepseek":
            sft_think = out_root / "prompts_hist_sft_with_think" / f"{t}.jsonl"
            _enrich_prompts_with_deepseek(
                ph_sft,
                sft_think,
                curr_only_prompt=True,
                strict=args.sft_think_strict,
                max_workers=args.sft_think_workers,
                max_retries=args.sft_think_retries,
                backoff_sec=args.sft_think_backoff,
            )
            if not sft_think.exists() or sft_think.stat().st_size == 0:
                raise RuntimeError(f"[typeagg-all] DeepSeek output empty: {sft_think}")
            sft_input = sft_think

        sft_out = out_root / "sft" / f"sft_train_{t}.jsonl"
        print(f"[typeagg-all] {t}: convert SFT -> {sft_out}")
        _convert_prompts_to_sft(
            sft_input,
            sft_out,
            system=system_prompt,
            inv_type=t,
            with_think=args.sft_with_think,
            contract_mode="delta",
            decimals=2,
            think_template="",
            label=f"sft_train_{t}",
            progress_every=100,
            curr_only_prompt=True,
        )

        grpo_out = out_root / "grpo" / f"grpo_{t}.jsonl"
        print(f"[typeagg-all] {t}: convert GRPO -> {grpo_out}")
        _convert_prompts_to_grpo(
            ph_grpo,
            grpo_out,
            system=system_prompt,
            inv_type=t,
            no_think_example=args.grpo_no_think_example,
            label=f"grpo_{t}",
            progress_every=100,
            curr_only_prompt=True,
        )

        test_out = out_root / "test" / f"test_{t}_all.jsonl"
        print(f"[typeagg-all] {t}: convert TEST -> {test_out}")
        _convert_prompts_to_test_optional_label(
            ph_test,
            test_out,
            system=system_prompt,
            inv_type=t,
            label=f"test_{t}_all",
            progress_every=100,
            curr_only_prompt=True,
        )
        if args.emit_base_min_test:
            base_min_out = out_root / "test" / f"test_{t}_base_min.jsonl"
            print(f"[typeagg-all] {t}: convert TEST -> base-min ({base_min_out})")
            _convert_prompts_to_test_optional_label(
                ph_test,
                base_min_out,
                system=system_prompt,
                system_suffix=(
                    "Output only <answer>{\"holding_log_delta\": <number>}</answer>. "
                    "No <think> section. No other text."
                ),
                inv_type=t,
                label=f"test_base_min_{t}",
                progress_every=100,
                curr_only_prompt=True,
            )

    print("[typeagg-all] done")
    print(f"[typeagg-all] total prompts: sft={total['sft']} grpo={total['grpo']} test={total['test']}")


if __name__ == "__main__":
    main()
