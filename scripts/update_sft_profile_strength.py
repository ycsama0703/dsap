#!/usr/bin/env python3
"""
Update <profile_context> JSON inside SFT jsonl files, without touching think/answer.

Default behavior:
- Adds "profile_strength" with neutral global means computed from stats CSV.
- Does NOT change any other fields.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_global_means(stats_path: Path) -> Dict[str, float]:
    import pandas as pd

    df = pd.read_csv(stats_path)
    if df.empty:
        return {"x_risk_mean": 0.0, "x_herd_mean": 0.0, "x_profit_mean": 0.0}
    return {
        "x_risk_mean": float(df["x_risk_mean"].mean()),
        "x_herd_mean": float(df["x_herd_mean"].mean()),
        "x_profit_mean": float(df["x_profit_mean"].mean()),
    }


def compute_overall_mean(x_risk: float, x_herd: float, x_profit: float) -> float:
    return float((abs(x_risk) + abs(x_herd) + abs(x_profit)) / 3.0)


def extract_profile_context(content: str) -> Tuple[str, Dict[str, Any], str] | None:
    start_tag = "<profile_context>"
    end_tag = "</profile_context>"
    start = content.find(start_tag)
    end = content.find(end_tag)
    if start == -1 or end == -1 or end <= start:
        return None
    before = content[: start + len(start_tag)]
    after = content[end:]
    raw = content[start + len(start_tag) : end].strip()
    if not raw:
        return None
    try:
        prof = json.loads(raw)
    except Exception:
        return None
    return before, prof, after


def write_profile_context(before: str, prof: Dict[str, Any], after: str) -> str:
    # Keep ASCII unless input contained non-ASCII.
    prof_json = json.dumps(prof, ensure_ascii=True)
    return f"{before}\n{prof_json}\n{after}"


def update_profile_strength(
    prof: Dict[str, Any],
    mode: str,
    neutral_means: Dict[str, float],
) -> Dict[str, Any]:
    if mode == "neutral-null":
        strength = {
            "mode": "neutral_null",
            "x_risk_mean": None,
            "x_herd_mean": None,
            "x_profit_mean": None,
            "overall_mean": None,
        }
    else:
        # neutral-mean (default)
        x_r = neutral_means["x_risk_mean"]
        x_h = neutral_means["x_herd_mean"]
        x_p = neutral_means["x_profit_mean"]
        strength = {
            "mode": "neutral_mean",
            "x_risk_mean": x_r,
            "x_herd_mean": x_h,
            "x_profit_mean": x_p,
            "overall_mean": compute_overall_mean(x_r, x_h, x_p),
        }

    prof["profile_strength"] = strength
    return prof


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def process_file(
    in_path: Path,
    out_path: Path,
    mode: str,
    neutral_means: Dict[str, float],
) -> int:
    updated = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            for msg in msgs:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                extracted = extract_profile_context(content)
                if not extracted:
                    continue
                before, prof, after = extracted
                prof = update_profile_strength(prof, mode, neutral_means)
                msg["content"] = write_profile_context(before, prof, after)
                updated += 1
                break
            f_out.write(json.dumps(obj, ensure_ascii=True) + "\n")
    return updated


def main() -> None:
    ap = argparse.ArgumentParser(description="Update profile_context in SFT jsonl files.")
    ap.add_argument("--sft-dir", required=True, help="Directory with sft_train_*.jsonl")
    ap.add_argument("--stats", default="data/grpo_profile_stats.csv")
    ap.add_argument(
        "--mode",
        choices=["neutral-mean", "neutral-null"],
        default="neutral-mean",
        help="Neutral strength strategy for SFT.",
    )
    ap.add_argument("--suffix", default=".profile_strength.jsonl")
    ap.add_argument("--in-place", action="store_true", help="Overwrite files in place.")
    args = ap.parse_args()

    sft_dir = Path(args.sft_dir)
    stats_path = Path(args.stats)
    neutral_means = load_global_means(stats_path) if stats_path.exists() else {
        "x_risk_mean": 0.0,
        "x_herd_mean": 0.0,
        "x_profit_mean": 0.0,
    }

    files = sorted(sft_dir.glob("sft_train_*.jsonl"))
    if not files:
        raise SystemExit(f"No sft_train_*.jsonl found in {sft_dir}")

    for path in files:
        out_path = path if args.in_place else path.with_suffix(path.suffix + args.suffix)
        updated = process_file(path, out_path, args.mode, neutral_means)
        print(f"[update] {path.name} -> {out_path.name} (updated {updated} rows)")


if __name__ == "__main__":
    main()
