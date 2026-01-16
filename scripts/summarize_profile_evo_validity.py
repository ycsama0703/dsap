#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize valid/invalid sample counts from profile evolution debug logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize valid samples per generation.")
    ap.add_argument("--out-dir", type=str, required=True, help="outputs/profile_evo/<type>")
    ap.add_argument("--stage", type=str, default="population", choices=["population", "llm_candidates", "both"])
    ap.add_argument("--details", action="store_true", help="Print per-candidate counts.")
    return ap.parse_args()


def iter_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def count_valid(record: Dict[str, Any]) -> Tuple[int, int]:
    if record.get("valid_samples") is not None and record.get("invalid_samples") is not None:
        return int(record["valid_samples"]), int(record["invalid_samples"])
    details = record.get("details") or []
    valid = 0
    for d in details:
        if d.get("valid") is True:
            valid += 1
        elif d.get("valid") is False:
            continue
        else:
            if d.get("pred") is not None and d.get("y_true") is not None:
                valid += 1
    invalid = max(0, len(details) - valid)
    return valid, invalid


def summarize(records: List[dict]) -> Dict[str, Any]:
    if not records:
        return {}
    valid_counts = []
    total_counts = []
    for r in records:
        v, inv = count_valid(r)
        valid_counts.append(v)
        total_counts.append(v + inv)
    return {
        "candidates": len(records),
        "valid_min": min(valid_counts),
        "valid_max": max(valid_counts),
        "valid_mean": mean(valid_counts),
        "valid_median": median(valid_counts),
        "total_mean": mean(total_counts),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    debug_root = out_dir / "debug"
    if not debug_root.exists():
        raise SystemExit(f"[error] debug folder not found: {debug_root}")

    stages = ["population", "llm_candidates"] if args.stage == "both" else [args.stage]
    gen_dirs = sorted([p for p in debug_root.glob("gen_*") if p.is_dir()])

    for gen_dir in gen_dirs:
        gen = gen_dir.name.replace("gen_", "")
        for stage in stages:
            path = gen_dir / f"{stage}.jsonl"
            records = iter_jsonl(path)
            if not records:
                continue
            summary = summarize(records)
            print(
                f"[gen {gen}] stage={stage} "
                f"candidates={summary['candidates']} "
                f"valid_mean={summary['valid_mean']:.2f} "
                f"valid_min={summary['valid_min']} "
                f"valid_max={summary['valid_max']} "
                f"total_mean={summary['total_mean']:.2f}"
            )
            if args.details:
                for r in records:
                    v, inv = count_valid(r)
                    idx = r.get("candidate_idx")
                    mean_reward = r.get("mean_reward")
                    print(
                        f"  - candidate {idx}: valid={v} invalid={inv} mean_reward={mean_reward}"
                    )


if __name__ == "__main__":
    main()
