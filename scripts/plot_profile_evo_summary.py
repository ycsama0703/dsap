#!/usr/bin/env python3
"""
Plot before/after reward bars per type from profile_evolution_results.json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_rewards(path: Path) -> Tuple[float, float] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rewards = data.get("best_rewards") or []
    if not rewards:
        return None
    try:
        before = float(rewards[0])
        after = float(rewards[-1])
    except Exception:
        return None
    return before, after


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot before/after rewards per type.")
    ap.add_argument("--root", required=True, help="Root dir with per-type outputs")
    ap.add_argument("--out-path", default=None, help="Output image path")
    ap.add_argument("--types", nargs="*", default=None, help="Optional list of types to include")
    args = ap.parse_args()

    root = Path(args.root)
    if args.types:
        type_dirs = [root / t for t in args.types]
    else:
        type_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    rows: List[Tuple[str, float, float]] = []
    for tdir in type_dirs:
        res_path = tdir / "profile_evolution_results.json"
        if not res_path.exists():
            continue
        rewards = load_rewards(res_path)
        if not rewards:
            continue
        before, after = rewards
        rows.append((tdir.name, before, after))

    if not rows:
        raise SystemExit(f"No results found under {root}")

    labels = [r[0] for r in rows]
    before_vals = [r[1] for r in rows]
    after_vals = [r[2] for r in rows]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], before_vals, width, label="before")
    plt.bar([i + width / 2 for i in x], after_vals, width, label="after")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Reward")
    plt.title("Before vs After Reward by Type")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out_path) if args.out_path else root / "profile_evo_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
