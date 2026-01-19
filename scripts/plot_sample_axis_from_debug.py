#!/usr/bin/env python3
"""
Rebuild sample-axis plot from debug/ gen_* logs and progress.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def norm_weights(w: Dict[str, float], ndigits: int = 6) -> Tuple[Tuple[str, float], ...]:
    return tuple((k, round(float(v), ndigits)) for k, v in sorted(w.items()))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_records(gen_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for name in ("population.jsonl", "llm_candidates.jsonl"):
        records.extend(iter_jsonl(gen_dir / name))
    return records


def best_match_record(
    records: List[Dict[str, Any]], target_weights: Dict[str, float]
) -> Dict[str, Any] | None:
    if not records:
        return None
    target = norm_weights(target_weights)
    by_weight: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
    for r in records:
        w = r.get("weights") or {}
        key = norm_weights(w)
        if key not in by_weight:
            by_weight[key] = r
    if target in by_weight:
        return by_weight[target]

    def l1_dist(w: Dict[str, float]) -> float:
        s = 0.0
        for k in ("risk_aversion", "herd_behavior", "profit_driven"):
            try:
                s += abs(float(w.get(k, 0.0)) - float(target_weights.get(k, 0.0)))
            except Exception:
                s += 1e9
        return s

    return min(records, key=lambda r: l1_dist(r.get("weights") or {}))


def rewards_from_details(details: List[Dict[str, Any]]) -> List[float]:
    out = []
    for d in details:
        r = d.get("reward")
        if r is None:
            continue
        try:
            out.append(float(r))
        except Exception:
            continue
    return out


def compute_cum_avg(out_dir: Path) -> Tuple[List[float], int]:
    progress_path = out_dir / "progress.jsonl"
    if not progress_path.exists():
        return [], 0

    progress = [json.loads(x) for x in progress_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    progress = sorted(progress, key=lambda r: r.get("generation", 0))

    gen_rewards: List[List[float]] = []
    missing = 0
    for entry in progress:
        gen = int(entry.get("generation", 0))
        best_profile = entry.get("best_profile") or {}
        gen_dir = out_dir / "debug" / f"gen_{gen}"
        records = load_records(gen_dir)
        rec = best_match_record(records, best_profile)
        if not rec:
            missing += 1
            gen_rewards.append([])
            continue
        rewards = rewards_from_details(rec.get("details") or [])
        gen_rewards.append(rewards)

    flat_rewards = [r for g in gen_rewards for r in g]
    if not flat_rewards:
        return [], missing

    cum_avg = []
    s = 0.0
    for i, r in enumerate(flat_rewards, 1):
        s += r
        cum_avg.append(s / i)
    return cum_avg, missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot sample-axis cumulative mean reward.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir", help="Type output dir with debug/ and progress.jsonl")
    group.add_argument("--root", help="Root dir containing per-type output dirs")
    ap.add_argument("--out-path", default=None, help="Output image path")
    ap.add_argument("--types", nargs="*", default=None, help="Optional list of types to include")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise SystemExit("matplotlib not available.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        cum_avg, missing = compute_cum_avg(out_dir)
        if not cum_avg:
            raise SystemExit("No rewards found from debug logs.")

        plt.figure()
        plt.plot(range(1, len(cum_avg) + 1), cum_avg, marker="o")
        plt.xlabel("Samples Evaluated")
        plt.ylabel("Cumulative Mean Reward")
        plt.title("Cumulative Mean Reward vs Samples (best profile)")
        plt.grid(True)

        out_path = Path(args.out_path) if args.out_path else out_dir / "best_value_reward.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()

        if missing:
            print(f"[warn] missing best_profile records for {missing} generation(s)")
        print(f"[ok] wrote {out_path}")
        return

    root = Path(args.root)
    if args.types:
        type_dirs = [root / t for t in args.types]
    else:
        type_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    plt.figure()
    total_missing = 0
    plotted = 0
    for tdir in type_dirs:
        cum_avg, missing = compute_cum_avg(tdir)
        if not cum_avg:
            continue
        total_missing += missing
        plotted += 1
        plt.plot(range(1, len(cum_avg) + 1), cum_avg, label=tdir.name)

    if plotted == 0:
        raise SystemExit(f"No rewards found under {root}")

    plt.xlabel("Samples Evaluated")
    plt.ylabel("Cumulative Mean Reward")
    plt.title("Cumulative Mean Reward vs Samples (best profile)")
    plt.grid(True)
    plt.legend()

    out_path = Path(args.out_path) if args.out_path else root / "best_value_reward_all_types.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    if total_missing:
        print(f"[warn] missing best_profile records for {total_missing} generation(s)")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
