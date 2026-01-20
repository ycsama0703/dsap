#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def load_profiles(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    vals = {k: float(v) for k, v in weights.items()}
    s = sum(vals.values())
    if s <= 0:
        return vals
    return {k: v / s for k, v in vals.items()}


def find_best_weights(result_path: Path) -> Dict[str, float] | None:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    weights = None
    if isinstance(data.get("best_profiles"), list) and data["best_profiles"]:
        weights = data["best_profiles"][-1]
    elif isinstance(data.get("best_profile"), dict):
        weights = data["best_profile"]
    if not isinstance(weights, dict):
        return None
    return normalize_weights(weights)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Update profile semantics from evolution outputs.")
    ap.add_argument("--evo-root", type=str, default="outputs/profile_evo_exp_v7")
    ap.add_argument("--profile-init", type=str, default="artifacts/features/type_profile_semantics_init.json")
    ap.add_argument("--profile-out", type=str, default="artifacts/features/type_profile_semantics.json")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    evo_root = Path(args.evo_root)
    init_path = Path(args.profile_init)
    out_path = Path(args.profile_out)

    init_profiles = load_profiles(init_path)
    init_map = {p.get("investor_type"): p for p in init_profiles if p.get("investor_type")}

    if out_path.exists():
        current_profiles = load_profiles(out_path)
    else:
        current_profiles = []
    current_map = {p.get("investor_type"): p for p in current_profiles if p.get("investor_type")}

    updated = []
    for type_dir in sorted(p for p in evo_root.iterdir() if p.is_dir()):
        inv_type = type_dir.name
        result_path = type_dir / "profile_evolution_results.json"
        if not result_path.exists():
            print(f"[skip] missing results: {result_path}")
            continue
        weights = find_best_weights(result_path)
        if weights is None:
            print(f"[skip] no weights in: {result_path}")
            continue
        if inv_type in current_map:
            entry = current_map[inv_type]
        elif inv_type in init_map:
            entry = deepcopy(init_map[inv_type])
            current_profiles.append(entry)
        else:
            print(f"[skip] type not found in init/current: {inv_type}")
            continue
        entry["objective_weights"] = weights
        updated.append(inv_type)

    if args.dry_run:
        print(f"[dry-run] would update: {', '.join(updated)}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(current_profiles, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[done] updated {len(updated)} types -> {out_path}")


if __name__ == "__main__":
    main()
