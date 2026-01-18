#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-GPU evolved profile JSONs into a single profile file.

Default inputs:
  artifacts/features/type_profile_semantics_gpu0.json
  artifacts/features/type_profile_semantics_gpu1.json
  artifacts/features/type_profile_semantics_gpu2.json

Default output:
  artifacts/features/type_profile_semantics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_profiles(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def to_map(profiles: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in profiles:
        t = p.get("investor_type")
        if t:
            out[t] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-GPU profile outputs.")
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "artifacts/features/type_profile_semantics_gpu0.json",
            "artifacts/features/type_profile_semantics_gpu1.json",
            "artifacts/features/type_profile_semantics_gpu2.json",
        ],
        help="Ordered list of profile JSONs to merge (later wins).",
    )
    ap.add_argument(
        "--output",
        default="artifacts/features/type_profile_semantics.json",
        help="Output profile JSON path.",
    )
    args = ap.parse_args()

    output_path = Path(args.output)
    base_profiles = load_profiles(output_path)
    if not base_profiles:
        for src in args.inputs:
            base_profiles = load_profiles(Path(src))
            if base_profiles:
                break
    merged = to_map(base_profiles)

    updated_types = set()
    for src in args.inputs:
        src_path = Path(src)
        profiles = load_profiles(src_path)
        if not profiles:
            continue
        for p in profiles:
            t = p.get("investor_type")
            if not t:
                continue
            merged[t] = p
            updated_types.add(t)

    out_list = [merged[k] for k in sorted(merged.keys())]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_list, indent=2, ensure_ascii=True))
    print(f"[merge] wrote {output_path} (updated {len(updated_types)} types)")


if __name__ == "__main__":
    main()
