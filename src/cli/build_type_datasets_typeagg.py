from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd

from src.cli.build_history_prompts import build_for_file
from src.cli.build_type_datasets import (
    THINK_PLACEHOLDER,
    _build_structured_prompt,
    _build_system_prompt,
    _convert_prompts_to_grpo,
    _convert_prompts_to_sft,
    _get_profile_strength,
    _load_market_quarterly_safe,
)

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


def _estimate_total_records(fp: Path) -> int | None:
    try:
        with fp.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def _create_progress_bar(label: str, total: int | None):
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm(total=total, desc=label)


def _parse_permno_filter(raw: str | None, path_str: str | None) -> set[int] | None:
    vals: set[int] = set()

    def _consume(text: str):
        for token in re.split(r"[\s,]+", text.strip()):
            if not token:
                continue
            try:
                vals.add(int(token))
            except Exception:
                continue

    if raw:
        _consume(raw)
    if path_str:
        try:
            p = Path(path_str)
            if p.exists():
                with p.open() as f:
                    for line in f:
                        _consume(line)
        except Exception:
            pass
    return vals or None


def _convert_prompts_to_test_optional_label(
    inp: Path,
    outp: Path,
    *,
    system: str,
    system_suffix: str | None = None,
    inv_type: str | None = None,
    label: str = "test",
    progress_every: int = 100,
    curr_only_prompt: bool = True,
) -> int:
    sem_path = Path("artifacts/features/profile_semantics_llm.json")
    sem_map: dict[tuple[str, int], dict] = {}
    sem_type_path = Path("artifacts/features/type_profile_semantics.json")
    sem_type_map: dict[str, dict] = {}
    if sem_path.exists():
        try:
            data = json.loads(sem_path.read_text(encoding="utf-8"))
            for item in data:
                pid = item.get("profile_id")
                if not pid or "_p" not in pid:
                    continue
                tname, kstr = pid.rsplit("_p", 1)
                try:
                    k = int(kstr)
                except Exception:
                    continue
                sem_map[(tname, k)] = item
        except Exception as e:
            print(f"[warn] failed to load profile semantics: {e}")
    if sem_type_path.exists():
        try:
            data = json.loads(sem_type_path.read_text(encoding="utf-8"))
            for item in data:
                t = item.get("investor_type")
                if not t:
                    continue
                sem_type_map[t] = item
        except Exception as e:
            print(f"[warn] failed to load type profile semantics: {e}")

    outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = _build_structured_prompt(rec, curr_only=curr_only_prompt)
            if not prompt:
                prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue

            mgrno_val = rec.get("mgrno")
            base_system = _build_system_prompt(inv_type, mgrno_val) if inv_type else system
            if system_suffix:
                system_content = f"{base_system}\n\n{system_suffix}"
            else:
                system_content = base_system

            prof_k = rec.get("label_profile_k") or rec.get("profile_k")
            if prof_k is None:
                prof_k = 0
            obj_w = rec.get("objective_weights")
            semantics = {}
            if prof_k is not None and inv_type is not None:
                semantics = sem_map.get((inv_type, int(prof_k)), {})
                obj = semantics.get("objective_weights") or {}
                if obj:
                    obj_w = {
                        "alpha_w": obj.get("alpha"),
                        "risk_w": obj.get("risk"),
                        "tc_w": obj.get("tc"),
                    }
            if (not semantics) and inv_type is not None:
                sem_fallback = sem_type_map.get(inv_type)
                if sem_fallback:
                    semantics = sem_fallback
                    obj = semantics.get("objective_weights") or {}
                    if obj:
                        obj_w = {
                            "alpha_w": obj.get("risk_aversion"),
                            "risk_w": obj.get("herd_behavior"),
                            "tc_w": obj.get("profit_driven"),
                        }
                        prof_k = 0
            if prof_k is not None:
                ow_sem = obj_w or {}
                ow_norm = {
                    "risk_aversion": ow_sem.get("risk_aversion") if "risk_aversion" in ow_sem else ow_sem.get("alpha_w") or ow_sem.get("alpha"),
                    "herd_behavior": ow_sem.get("herd_behavior") if "herd_behavior" in ow_sem else ow_sem.get("risk_w") or ow_sem.get("risk"),
                    "profit_driven": ow_sem.get("profit_driven") if "profit_driven" in ow_sem else ow_sem.get("tc_w") or ow_sem.get("tc"),
                }
                ow_norm = {k: v for k, v in ow_norm.items() if v is not None}
                ctx = {}
                if ow_norm:
                    ctx["objective_weights"] = ow_norm
                if semantics:
                    ctx["philosophy"] = semantics.get("philosophy", {})
                    ctx["constraints"] = semantics.get("constraints", {})
                    if "summary" in semantics:
                        ctx["summary"] = semantics.get("summary")
                if "profile_strength" not in ctx:
                    strength = _get_profile_strength(inv_type, mode="real")
                    if strength:
                        ctx["profile_strength"] = strength
                if ctx:
                    prompt = "<profile_context>\n" + json.dumps(ctx, ensure_ascii=False) + "\n</profile_context>\n\n" + prompt

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            if curr_only_prompt and "history_rows" in rec and isinstance(rec["history_rows"], dict):
                rec = dict(rec)
                rec["history_rows"] = {"t": rec["history_rows"].get("t")}

            ht = rec.get("holding_t")
            tp1 = rec.get("label_tp1") or rec.get("label")
            hld = rec.get("holding_log_delta")
            if hld is None and ht is not None and tp1 is not None:
                try:
                    hld = math.log((float(tp1) + 1e-6) / (float(ht) + 1e-6))
                except Exception:
                    hld = None

            out = {
                "messages": messages,
                "holding_log_delta": hld,
                "label_delta": hld,
                "label_tp1": tp1,
                "holding_t": ht,
                "shares": rec.get("shares"),
                "mgrno": rec.get("mgrno"),
                "permno": rec.get("permno"),
                "date": rec.get("date"),
                "ticker": rec.get("ticker"),
                "company": rec.get("company"),
                "history_rows": rec.get("history_rows"),
                "vix_q_prev": rec.get("vix_q_prev"),
                "ln_market_volume_q_prev": rec.get("ln_market_volume_q_prev"),
                "stock_vol_q_prev": rec.get("stock_vol_q_prev"),
                "stock_ln_volume_q_prev": rec.get("stock_ln_volume_q_prev"),
                "profile_semantics": semantics,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[typeagg] {label}: converted {n}")
    print(f"[typeagg] {label}: done (total={n})")
    return n


def _enrich_prompts_with_deepseek(
    inp: Path,
    outp: Path,
    *,
    curr_only_prompt: bool,
    strict: bool,
    profile_mode: str = "real",
    max_workers: int = 1,
    max_retries: int = 0,
    backoff_sec: float = 1.0,
) -> int:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("[typeagg] deepseek think requested but no DEEPSEEK_API_KEY/OPENAI_API_KEY set")

    client = OpenAI(api_key=api_key, base_url=api_base)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = outp.with_suffix(outp.suffix + ".tmp")

    n = 0
    total = _estimate_total_records(inp)
    pbar = _create_progress_bar("deepseek-think", total)
    max_workers = max(1, int(max_workers))
    max_inflight = max(4, max_workers * 4)

    def _build_think(line: str, idx: int) -> tuple[int, dict | None]:
        rec = json.loads(line)
        prompt = _build_structured_prompt(rec, curr_only=curr_only_prompt)
        if not prompt:
            prompt = rec.get("prompt") or rec.get("query")
        if not prompt:
            return idx, None

        neutral_hint = ""
        if (profile_mode or "").lower() != "real":
            neutral_hint = "Assume a neutral investor profile; do not reference profile-specific preferences.\n"
        ds_prompt = (
            "You are a financial reasoning assistant.\n\n"
            f"{neutral_hint}"
            "Given the current data below, write a concise reasoning (<=3 sentences) inside "
            "<think>...</think> explaining the expected direction/magnitude of the holding change. "
            "Do NOT mention any true labels.\n\n---\n"
            f"{prompt}"
        )
        think_text = None
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a financial reasoning assistant."},
                        {"role": "user", "content": ds_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                think_text = resp.choices[0].message.content.strip()
                if not think_text.startswith("<think>"):
                    think_text = f"<think>{think_text}</think>"
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    sleep_for = max(0.0, backoff_sec) * (2 ** attempt)
                    if sleep_for:
                        time.sleep(sleep_for)
                    continue
                if strict:
                    raise RuntimeError(f"[typeagg] deepseek think failed: {e}") from e
                print(f"[WARN] DeepSeek generation failed: {e}")
                think_text = THINK_PLACEHOLDER

        if think_text is None and last_err is not None:
            if strict:
                raise RuntimeError(f"[typeagg] deepseek think failed: {last_err}") from last_err
            think_text = THINK_PLACEHOLDER

        rec["think"] = think_text
        return idx, rec

    with inp.open("r", encoding="utf-8") as f, tmp_path.open("w", encoding="utf-8") as fout:
        if max_workers == 1:
            for idx, line in enumerate(f):
                _, rec = _build_think(line, idx)
                if rec is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if pbar is not None:
                    pbar.update(1)
                elif n % 50 == 0:
                    print(f"[typeagg] deepseek think: {n} done")
        else:
            pending = {}
            buffer: dict[int, dict | None] = {}
            next_idx = 0

            def _drain_done(done_set):
                nonlocal n, next_idx
                for fut in done_set:
                    idx = pending.pop(fut)
                    rec = fut.result()
                    buffer[idx] = rec[1]
                    if pbar is not None:
                        pbar.update(1)
                while next_idx in buffer:
                    rec = buffer.pop(next_idx)
                    if rec is not None:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                    next_idx += 1

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for idx, line in enumerate(f):
                    fut = ex.submit(_build_think, line, idx)
                    pending[fut] = idx
                    if len(pending) >= max_inflight:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        _drain_done(done)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    _drain_done(done)
    if pbar is not None:
        pbar.close()
    if n == 0:
        raise RuntimeError(f"[typeagg] deepseek think produced empty file: {tmp_path}")
    os.replace(tmp_path, outp)
    print(f"[typeagg] deepseek think enriched: {n} -> {outp}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT/GRPO/Test datasets for type-aggregated panels")
    ap.add_argument("--type", required=True, help="Firm type stem, e.g., 'banks'")
    ap.add_argument("--in-dir", type=str, default="data/processed/type_agg",
                    help="Input directory with type-aggregated parquet files")
    ap.add_argument("--out-root", type=str, default="artifacts/typeagg",
                    help="Root output directory (separate from legacy artifacts)")
    ap.add_argument("--per-type-limit", type=int, default=1000)
    ap.add_argument("--sft-limit", type=int, default=None)
    ap.add_argument("--grpo-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)

    # date splits
    ap.add_argument("--sft-end", type=str, default="2017-12-31")
    ap.add_argument("--grpo-start", type=str, default="2018-01-01")
    ap.add_argument("--grpo-end", type=str, default="2022-12-31")
    ap.add_argument("--test-start", type=str, default="2023-01-01")

    ap.add_argument("--history-len", type=int, choices=[1, 2], default=1,
                    help="Number of consecutive quarters in prompt (default: 1 = current-only)")
    ap.add_argument("--exclude-zero", action="store_true",
                    help="Exclude windows where t-time holding_t == 0")
    ap.add_argument("--random-sample", action="store_true",
                    help="Use simple random sampling instead of stratified sampling.")

    ap.add_argument("--include-permnos", type=str, default="",
                    help="Comma/space separated list of permnos to include. Empty = all.")
    ap.add_argument("--include-permnos-file", type=str, default="",
                    help="Optional file with permnos (one per line) to include.")
    ap.add_argument("--test-permnos", type=str, default="",
                    help="Comma/space separated list of permnos for TEST split. Empty = use include-permnos.")
    ap.add_argument("--test-permnos-file", type=str, default="",
                    help="Optional file with permnos (one per line) for TEST split. Empty = use include-permnos.")
    ap.add_argument("--test-take-all", action="store_true",
                    help="If set, take all available test windows for the specified permnos (no sampling/limit).")

    ap.add_argument("--sft-with-think", action="store_true", default=False,
                    help="Include <think> block for SFT outputs (default: disabled)")
    ap.add_argument("--no-sft-think", dest="sft_with_think", action="store_false",
                    help="Disable <think> block for SFT outputs")
    ap.add_argument("--sft-think-source", choices=["placeholder", "deepseek"], default="placeholder",
                    help="Think generation source when enabled (default: placeholder)")
    ap.add_argument("--sft-think-workers", type=int, default=1,
                    help="DeepSeek think parallel workers (default: 1)")
    ap.add_argument("--sft-think-retries", type=int, default=3,
                    help="DeepSeek retry count on connection errors (default: 3)")
    ap.add_argument("--sft-think-backoff", type=float, default=1.5,
                    help="Base backoff seconds for retries (default: 1.5)")
    ap.add_argument("--sft-think-strict", action="store_true",
                    help="Fail if DeepSeek think generation errors")
    ap.add_argument("--sft-contract-mode", choices=["absolute", "delta"], default="delta",
                    help="Target for SFT outputs (default: delta)")
    ap.add_argument("--sft-decimals", type=int, default=2,
                    help="Decimal places for SFT/TEST labels (default: 2)")
    ap.add_argument("--sft-profile-mode", choices=["real", "neutral", "none"], default="neutral",
                    help="Profile context mode for SFT prompts (default: neutral)")

    ap.add_argument("--grpo-no-think-example", action="store_true",
                    help="Disable think/answer placeholder example in GRPO dataset")
    ap.add_argument("--emit-base-min-test", action="store_true",
                    help="Also emit base-min test set (single JSON output, no think).")

    ap.add_argument("--profile-dir", type=str, default=None,
                    help="Directory with *_iq_profile.(csv|parquet) per type (optional)")
    ap.add_argument("--profile-weights-dir", type=str, default=None,
                    help="Directory with *_profile_objective_weights.(csv|parquet) (optional)")
    args = ap.parse_args()

    t = args.type
    in_file = Path(args.in_dir) / f"{t}.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"input file not found: {in_file}")

    out_root = Path(args.out_root)
    prompts_sft = out_root / "prompts_hist_sft" / f"{t}.jsonl"
    prompts_grpo = out_root / "prompts_hist_grpo" / f"{t}.jsonl"
    prompts_test = out_root / "prompts_hist_test" / f"{t}.jsonl"
    sft_out = out_root / "sft" / f"sft_train_{t}.jsonl"
    grpo_out = out_root / "grpo" / f"grpo_{t}.jsonl"
    test_out = out_root / "test" / f"test_{t}_all.jsonl"

    mapping = None
    try:
        if load_mapping is not None:
            mp_path = Path("data/ticker_mapping.csv")
            if mp_path.exists():
                mapping = load_mapping(mp_path)
                print(f"[typeagg] loaded mapping: {len(mapping)}")
    except Exception as e:
        print(f"[typeagg] mapping load failed: {e}")

    market_df = _load_market_quarterly_safe()

    permno_filter = _parse_permno_filter(args.include_permnos, args.include_permnos_file)
    test_permno_filter = _parse_permno_filter(args.test_permnos, args.test_permnos_file)

    take_all_mode = False
    take_all_mode_test = args.test_take_all or bool(test_permno_filter)
    test_limit_override = None if take_all_mode_test else args.test_limit

    print(f"[typeagg] building SFT prompts for {t} (<= {args.sft_end})")
    build_for_file(
        in_file=in_file,
        out_file=prompts_sft,
        per_type_limit=args.sft_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=None,
        date_end=args.sft_end,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=permno_filter,
        take_all=take_all_mode,
        limit_override=args.sft_limit,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
        require_holding_t1=True,
    )

    print(f"[typeagg] building GRPO prompts for {t} ({args.grpo_start}..{args.grpo_end})")
    build_for_file(
        in_file=in_file,
        out_file=prompts_grpo,
        per_type_limit=args.grpo_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=args.grpo_start,
        date_end=args.grpo_end,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=permno_filter,
        take_all=take_all_mode,
        limit_override=args.grpo_limit,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
        require_holding_t1=True,
    )

    print(f"[typeagg] building TEST prompts for {t} (>= {args.test_start})")
    build_for_file(
        in_file=in_file,
        out_file=prompts_test,
        per_type_limit=args.test_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=args.test_start,
        date_end=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=test_permno_filter or permno_filter,
        take_all=take_all_mode_test,
        limit_override=test_limit_override,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
        require_holding_t1=False,
    )

    system_prompt = _build_system_prompt(t)

    print(f"[typeagg] convert SFT -> {sft_out}")
    sft_input = prompts_sft
    if args.sft_with_think and args.sft_think_source == "deepseek":
        prompts_sft_think = out_root / "prompts_hist_sft_with_think" / f"{t}.jsonl"
        _enrich_prompts_with_deepseek(
            prompts_sft,
            prompts_sft_think,
            curr_only_prompt=True,
            strict=args.sft_think_strict,
            profile_mode=args.sft_profile_mode,
            max_workers=args.sft_think_workers,
            max_retries=args.sft_think_retries,
            backoff_sec=args.sft_think_backoff,
        )
        sft_input = prompts_sft_think
    sft_tmp = sft_out.with_suffix(".tmp.jsonl")
    think_template = THINK_PLACEHOLDER if args.sft_with_think and args.sft_think_source == "placeholder" else ""
    _convert_prompts_to_sft(
        sft_input,
        sft_tmp,
        system=system_prompt,
        inv_type=t,
        with_think=args.sft_with_think,
        contract_mode=args.sft_contract_mode,
        decimals=args.sft_decimals,
        think_template=think_template,
        profile_mode=args.sft_profile_mode,
        label=f"sft_train_{t}",
        progress_every=100,
        curr_only_prompt=True,
    )
    if not sft_tmp.exists() or sft_tmp.stat().st_size == 0:
        raise RuntimeError(f"[typeagg] SFT conversion produced empty file: {sft_tmp}")
    os.replace(sft_tmp, sft_out)

    print(f"[typeagg] convert GRPO -> {grpo_out}")
    _convert_prompts_to_grpo(
        prompts_grpo,
        grpo_out,
        system=system_prompt,
        inv_type=t,
        no_think_example=args.grpo_no_think_example,
        label=f"grpo_{t}",
        progress_every=100,
        curr_only_prompt=True,
    )

    print(f"[typeagg] convert TEST -> {test_out} (labels optional)")
    _convert_prompts_to_test_optional_label(
        prompts_test,
        test_out,
        system=system_prompt,
        inv_type=t,
        label=f"test_{t}_all",
        progress_every=100,
        curr_only_prompt=True,
    )
    if args.emit_base_min_test:
        base_min_out = out_root / "test" / f"test_{t}_base_min.jsonl"
        print(f"[typeagg] convert TEST -> base-min ({base_min_out})")
        _convert_prompts_to_test_optional_label(
            prompts_test,
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

    print("[typeagg] done.")


if __name__ == "__main__":
    main()
