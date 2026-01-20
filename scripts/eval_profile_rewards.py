#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_profile_evolution_module(repo_root: Path):
    mod_path = repo_root / "scripts" / "profile_evolution.py"
    spec = importlib.util.spec_from_file_location("profile_evolution", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_init_profiles(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for item in data:
        inv = item.get("investor_type")
        weights = item.get("objective_weights")
        if isinstance(inv, str) and isinstance(weights, dict):
            out[inv] = dict(weights)
    return out


def load_final_profile(path: Path) -> Dict[str, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    best_profiles = data.get("best_profiles")
    if isinstance(best_profiles, list) and best_profiles:
        last = best_profiles[-1]
        if isinstance(last, dict):
            return dict(last)
    return None


def select_eval_subset(
    chats: List[List[Dict[str, str]]],
    y_true: List[float],
    seed: int,
    eval_size: int,
) -> Tuple[List[List[Dict[str, str]]], List[float]]:
    indices = list(range(len(chats)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    total = min(len(indices), eval_size)
    indices = indices[:total]
    return [chats[i] for i in indices], [y_true[i] for i in indices]


def find_latest_checkpoint(lora_root: Path, inv_type: str) -> str | None:
    base = lora_root / inv_type
    candidates = sorted(base.glob("v*/checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate init vs evolved profiles on GRPO eval set.")
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--eval-root", type=str, default="artifacts/typeagg_all_v4")
    ap.add_argument("--profile-init", type=str, default="artifacts/features/type_profile_semantics_init.json")
    ap.add_argument("--profile-evo-root", type=str, default="outputs/profile_evo_exp_v7")
    ap.add_argument("--lora-root", type=str, default="outputs/sft_output")
    ap.add_argument("--eval-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-reasoning", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--types", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="outputs/profile_evo_exp_v7/profile_reward_compare.csv")
    ap.add_argument("--out-plot", type=str, default="outputs/profile_evo_exp_v7/profile_reward_compare.png")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.backends.hf_infer import build_eval_inputs, load_model_and_tokenizer  # noqa: E402

    pe = load_profile_evolution_module(repo_root)

    init_profiles = load_init_profiles(Path(args.profile_init))
    profile_root = Path(args.profile_evo_root)
    eval_root = Path(args.eval_root)
    lora_root = Path(args.lora_root)

    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
    else:
        wanted = None

    type_dirs = [p for p in profile_root.iterdir() if p.is_dir()]
    types = sorted([p.name for p in type_dirs if wanted is None or p.name in wanted])
    if not types:
        raise RuntimeError(f"no types found under: {profile_root}")

    rows = []
    for inv_type in types:
        init_weights = init_profiles.get(inv_type)
        if init_weights is None:
            print(f"[skip] init profile missing for {inv_type}")
            continue
        final_path = profile_root / inv_type / "profile_evolution_results.json"
        final_weights = load_final_profile(final_path)
        if final_weights is None:
            print(f"[skip] final profile missing for {inv_type}: {final_path}")
            continue
        if hasattr(pe, "normalize_weights"):
            init_weights = pe.normalize_weights(init_weights)
            final_weights = pe.normalize_weights(final_weights)
        test_path = eval_root / "grpo" / f"grpo_{inv_type}.jsonl"
        if not test_path.exists():
            print(f"[skip] eval data missing for {inv_type}: {test_path}")
            continue
        ckpt = find_latest_checkpoint(lora_root, inv_type)
        if ckpt is None:
            print(f"[skip] checkpoint missing for {inv_type} under {lora_root}")
            continue

        print(f"[load] {inv_type} ckpt={ckpt}")
        tok, model = load_model_and_tokenizer(args.base_model, ckpt, torch_dtype=args.torch_dtype)

        chats, y_true, *_ = build_eval_inputs(str(test_path))
        eval_chats, eval_y = select_eval_subset(chats, y_true, args.seed, args.eval_size)

        eval_args = SimpleNamespace(
            k_reasoning=args.k_reasoning,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            progress=False,
        )

        init_scores = pe.evaluate_profile_scores(init_weights, eval_chats, eval_y, tok, model, eval_args)
        final_scores = pe.evaluate_profile_scores(final_weights, eval_chats, eval_y, tok, model, eval_args)
        init_mean = float(np.mean(init_scores)) if init_scores else -1e9
        final_mean = float(np.mean(final_scores)) if final_scores else -1e9

        rows.append(
            {
                "type": inv_type,
                "reward_init": init_mean,
                "reward_final": final_mean,
                "valid_init": len(init_scores),
                "valid_final": len(final_scores),
                "checkpoint": ckpt,
            }
        )

    if not rows:
        raise RuntimeError("no rows computed; check inputs")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("type,reward_init,reward_final,valid_init,valid_final,checkpoint\n")
        for r in rows:
            f.write(
                f"{r['type']},{r['reward_init']},{r['reward_final']},"
                f"{r['valid_init']},{r['valid_final']},{r['checkpoint']}\n"
            )

    labels = [r["type"] for r in rows]
    init_vals = [r["reward_init"] for r in rows]
    final_vals = [r["reward_final"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig_w = max(6.0, len(labels) * 1.2)
    plt.figure(figsize=(fig_w, 4.5))
    plt.bar(x - width / 2, init_vals, width, label="init")
    plt.bar(x + width / 2, final_vals, width, label="final")
    plt.ylabel("Mean Value Reward")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"[done] csv: {out_csv}")
    print(f"[done] plot: {out_plot}")


if __name__ == "__main__":
    main()
