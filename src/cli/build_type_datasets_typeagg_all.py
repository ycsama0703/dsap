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


def _parse_type_alias(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid type alias '{item}', expected src=dst")
        src, dst = item.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"invalid type alias '{item}', expected src=dst")
        mapping[src] = dst
    return mapping


def _tmp_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}.{tag}{base.suffix}")


def _append_file(src: Path, dst: Path) -> int:
    if not src.exists() or src.stat().st_size == 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as fsrc, dst.open("a", encoding="utf-8") as fdst:
        for line in fsrc:
            fdst.write(line)
            n += 1
    return n


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT/GRPO datasets for all type-agg panels")
    ap.add_argument("--in-dir", type=Path, default=Path("data/processed/type_agg"),
                    help="Input directory with type-aggregated parquet files")
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/typeagg_all"),
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
    ap.add_argument("--sft-profile-mode", choices=["real", "neutral", "none"], default="neutral",
                    help="Profile context mode for SFT prompts (default: neutral)")
    ap.add_argument("--grpo-no-think-example", action="store_true",
                    help="Disable think/answer placeholder example in GRPO dataset")
    ap.add_argument("--emit-base-min-test", action="store_true",
                    help="Also emit base-min test set (single JSON output, no think).")
    ap.add_argument("--type-alias", type=str, default="",
                    help="Comma-separated rename mapping, e.g. other=households")
    args = ap.parse_args()

    if not args.in_dir.exists():
        raise FileNotFoundError(f"input directory not found: {args.in_dir}")

    if args.types.strip():
        types = [s.strip() for s in args.types.split(",") if s.strip()]
    else:
        types = discover_types(args.in_dir)
        if not types:
            raise FileNotFoundError(f"no parquet files in {args.in_dir}")

    alias_map = _parse_type_alias(args.type_alias) if args.type_alias.strip() else {}

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
    written_types: set[str] = set()

    for t in types:
        in_file = args.in_dir / f"{t}.parquet"
        if not in_file.exists():
            print(f"[typeagg-all] skip type={t}: {in_file} not found")
            continue

        out_type = alias_map.get(t, t)
        append = out_type in written_types
        if out_type != t:
            print(f"[typeagg-all] alias {t} -> {out_type}")

        ph_sft = prompts_sft_dir / f"{out_type}.jsonl"
        ph_grpo = prompts_grpo_dir / f"{out_type}.jsonl"
        ph_test = prompts_test_dir / f"{out_type}.jsonl"
        ph_sft_work = _tmp_path(ph_sft, t) if append else ph_sft
        ph_grpo_work = _tmp_path(ph_grpo, t) if append else ph_grpo
        ph_test_work = _tmp_path(ph_test, t) if append else ph_test

        print(f"[typeagg-all] {t}: build SFT prompts (<= {args.sft_end})")
        total["sft"] += build_for_file(
            in_file=in_file,
            out_file=ph_sft_work,
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
            out_file=ph_grpo_work,
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
            out_file=ph_test_work,
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

        system_prompt = _build_system_prompt(out_type)

        sft_input = ph_sft_work if append else ph_sft
        sft_think_work = None
        if args.sft_with_think and args.sft_think_source == "deepseek":
            sft_think = out_root / "prompts_hist_sft_with_think" / f"{out_type}.jsonl"
            if append:
                sft_think = _tmp_path(sft_think, t)
            _enrich_prompts_with_deepseek(
                ph_sft if not append else ph_sft_work,
                sft_think,
                curr_only_prompt=True,
                strict=args.sft_think_strict,
                profile_mode=args.sft_profile_mode,
                max_workers=args.sft_think_workers,
                max_retries=args.sft_think_retries,
                backoff_sec=args.sft_think_backoff,
            )
            if not sft_think.exists() or sft_think.stat().st_size == 0:
                raise RuntimeError(f"[typeagg-all] DeepSeek output empty: {sft_think}")
            sft_input = sft_think
            sft_think_work = sft_think

        sft_out = out_root / "sft" / f"sft_train_{out_type}.jsonl"
        sft_out_work = _tmp_path(sft_out, t) if append else sft_out
        print(f"[typeagg-all] {t}: convert SFT -> {sft_out}")
        _convert_prompts_to_sft(
            sft_input,
            sft_out_work,
            system=system_prompt,
            inv_type=out_type,
            with_think=args.sft_with_think,
            contract_mode="delta",
            decimals=2,
            think_template="",
            profile_mode=args.sft_profile_mode,
            label=f"sft_train_{out_type}",
            progress_every=100,
            curr_only_prompt=True,
        )
        if append:
            _append_file(sft_out_work, sft_out)
            _safe_unlink(sft_out_work)
        if sft_think_work is not None and append:
            _safe_unlink(sft_think_work)

        grpo_out = out_root / "grpo" / f"grpo_{out_type}.jsonl"
        grpo_out_work = _tmp_path(grpo_out, t) if append else grpo_out
        print(f"[typeagg-all] {t}: convert GRPO -> {grpo_out}")
        _convert_prompts_to_grpo(
            ph_grpo if not append else ph_grpo_work,
            grpo_out_work,
            system=system_prompt,
            inv_type=out_type,
            no_think_example=args.grpo_no_think_example,
            label=f"grpo_{out_type}",
            progress_every=100,
            curr_only_prompt=True,
        )
        if append:
            _append_file(grpo_out_work, grpo_out)
            _safe_unlink(grpo_out_work)

        test_out = out_root / "test" / f"test_{out_type}_all.jsonl"
        test_out_work = _tmp_path(test_out, t) if append else test_out
        print(f"[typeagg-all] {t}: convert TEST -> {test_out}")
        _convert_prompts_to_test_optional_label(
            ph_test if not append else ph_test_work,
            test_out_work,
            system=system_prompt,
            inv_type=out_type,
            label=f"test_{out_type}_all",
            progress_every=100,
            curr_only_prompt=True,
        )
        if append:
            _append_file(test_out_work, test_out)
            _safe_unlink(test_out_work)
        if args.emit_base_min_test:
            base_min_out = out_root / "test" / f"test_{out_type}_base_min.jsonl"
            base_min_out_work = _tmp_path(base_min_out, t) if append else base_min_out
            print(f"[typeagg-all] {t}: convert TEST -> base-min ({base_min_out})")
            _convert_prompts_to_test_optional_label(
                ph_test if not append else ph_test_work,
                base_min_out_work,
                system=system_prompt,
                system_suffix=(
                    "Output only <answer>{\"holding_log_delta\": <number>}</answer>. "
                    "No <think> section. No other text."
                ),
                inv_type=out_type,
                label=f"test_base_min_{out_type}",
                progress_every=100,
                curr_only_prompt=True,
            )
            if append:
                _append_file(base_min_out_work, base_min_out)
                _safe_unlink(base_min_out_work)

        if append:
            _append_file(ph_sft_work, ph_sft)
            _append_file(ph_grpo_work, ph_grpo)
            _append_file(ph_test_work, ph_test)
            _safe_unlink(ph_sft_work)
            _safe_unlink(ph_grpo_work)
            _safe_unlink(ph_test_work)

        if out_type not in written_types:
            written_types.add(out_type)

    print("[typeagg-all] done")
    print(f"[typeagg-all] total prompts: sft={total['sft']} grpo={total['grpo']} test={total['test']}")


if __name__ == "__main__":
    main()
