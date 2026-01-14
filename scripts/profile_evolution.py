#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal profile evolution experiment:
- Keeps SFT/GRPO unchanged.
- Evolves objective_weights via outcome-guided accuracy only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    extract_pred,
    infer_chat_batch,
    load_model_and_tokenizer,
)

PROFILE_CONTEXT_RE = re.compile(r"<profile_context>\s*(\{.*?\})\s*</profile_context>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal profile evolution (objective_weights only).")
    ap.add_argument("--test-path", type=str, required=True)
    ap.add_argument("--profile-path", type=str, default="artifacts/features/type_profile_semantics.json")
    ap.add_argument("--investor-type", type=str, default="banks")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora-path", type=str, default=None)
    ap.add_argument("--eval-size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--population-size", type=int, default=8)
    ap.add_argument("--parents", type=int, default=2)
    ap.add_argument("--children-per-parent", type=int, default=3)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--k-reasoning", type=int, default=4)
    ap.add_argument("--mutation-sigma", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--write-profile-path", type=str, default=None,
                    help="If set, write best weights into this profile JSON (in-place update).")
    return ap.parse_args()


def normalize_weights(w: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
    s = sum(w.values())
    for k in w:
        w[k] = w[k] / (s + eps)
    return w


def mutate_objective_weights(w: Dict[str, float], sigma: float, eps: float = 1e-6) -> Dict[str, float]:
    keys = list(w.keys())
    vec = np.array([w[k] for k in keys], dtype=float)
    vec = vec + np.random.normal(0, sigma, size=len(vec))
    vec = np.clip(vec, eps, None)
    vec = vec / vec.sum()
    return {k: float(v) for k, v in zip(keys, vec)}


def init_population(seed: Dict[str, float], n: int, sigma: float) -> List[Dict[str, float]]:
    population = []
    for _ in range(n):
        population.append(mutate_objective_weights(seed, sigma=sigma))
    return population


def update_profile_context(user_text: str, objective_weights: Dict[str, float]) -> str:
    m = PROFILE_CONTEXT_RE.search(user_text or "")
    if not m:
        return user_text
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return user_text
    obj["objective_weights"] = objective_weights
    new_json = json.dumps(obj, ensure_ascii=True)
    return f"{user_text[:m.start(1)]}{new_json}{user_text[m.end(1):]}"


def build_weighted_chats(chats: List[List[Dict[str, str]]], weights: Dict[str, float]) -> List[List[Dict[str, str]]]:
    out = []
    for msgs in chats:
        sys_msg = msgs[0]
        usr_msg = msgs[1]
        new_user = {
            "role": "user",
            "content": update_profile_context(usr_msg["content"], weights),
        }
        out.append([{"role": "system", "content": sys_msg["content"]}, new_user])
    return out


def chunked(items: List[int], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def infer_in_batches(
    tokenizer,
    model,
    chats: List[List[Dict[str, str]]],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    indices = list(range(len(chats)))
    for batch in chunked(indices, batch_size):
        batch_msgs = [chats[i] for i in batch]
        outputs.extend(
            infer_chat_batch(
                tokenizer,
                model,
                batch_msgs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                force_think=False,
            )
        )
    return outputs


def value_reward(pred: float | None, y_true: float | None) -> float:
    if pred is None or y_true is None:
        return -1e9
    return -abs(pred - y_true)


def evaluate_profile(
    weights: Dict[str, float],
    eval_chats: List[List[Dict[str, str]]],
    eval_y: List[float],
    tokenizer,
    model,
    args: argparse.Namespace,
) -> float:
    weighted_chats = build_weighted_chats(eval_chats, weights)
    best_scores = [None] * len(eval_y)
    for _ in range(args.k_reasoning):
        completions = infer_in_batches(
            tokenizer,
            model,
            weighted_chats,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        preds = [extract_pred(c) for c in completions]
        for i, (pred, yt) in enumerate(zip(preds, eval_y)):
            r = value_reward(pred, yt)
            if best_scores[i] is None or r > best_scores[i]:
                best_scores[i] = r
    scores = [s if s is not None else -1e9 for s in best_scores]
    return float(np.mean(scores))


def evolve_population(
    population: List[Dict[str, float]],
    fitness_map: Dict[int, float],
    parents: int,
    children_per_parent: int,
    mutation_sigma: float,
    pop_size: int,
) -> List[Dict[str, float]]:
    ranked = sorted(population, key=lambda p: fitness_map[id(p)], reverse=True)
    top_parents = ranked[:parents]
    children = []
    for p in top_parents:
        for _ in range(children_per_parent):
            children.append(mutate_objective_weights(p, sigma=mutation_sigma))
    new_population = top_parents + children
    return new_population[:pop_size]


def maybe_save_outputs(out_dir: str, best_rewards: List[float], best_profiles: List[Dict[str, float]]) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_rewards": best_rewards,
        "best_profiles": best_profiles,
    }
    (out_path / "profile_evolution_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True)
    )
    try:
        import matplotlib.pyplot as plt  # type: ignore

        gens = list(range(len(best_rewards)))
        plt.figure()
        plt.plot(gens, best_rewards, marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Best Value Reward")
        plt.title("Best Value Reward vs Generation")
        plt.grid(True)
        plt.savefig(out_path / "best_value_reward.png", dpi=150)
        plt.close()

        ra = [p["risk_aversion"] for p in best_profiles]
        hb = [p["herd_behavior"] for p in best_profiles]
        pd = [p["profit_driven"] for p in best_profiles]
        plt.figure()
        plt.plot(gens, ra, label="risk_aversion")
        plt.plot(gens, hb, label="herd_behavior")
        plt.plot(gens, pd, label="profit_driven")
        plt.xlabel("Generation")
        plt.ylabel("Weight Value")
        plt.title("Objective Weight Evolution")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_path / "objective_weight_evolution.png", dpi=150)
        plt.close()
    except Exception:
        print("[profile-evo] matplotlib not available, skip plots")


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    profiles = json.loads(Path(args.profile_path).read_text(encoding="utf-8"))
    seed_profile = next((p for p in profiles if p.get("investor_type") == args.investor_type), None)
    if seed_profile is None:
        raise ValueError(f"investor_type not found in profile file: {args.investor_type}")
    seed_weights = normalize_weights(dict(seed_profile["objective_weights"]))

    chats, y_true, _, _, _, _, _ = build_eval_inputs(args.test_path)
    if not chats:
        raise RuntimeError("no eval samples found in test set")
    indices = list(range(len(chats)))
    random.shuffle(indices)
    eval_n = min(args.eval_size, len(indices))
    indices = indices[:eval_n]
    eval_chats = [chats[i] for i in indices]
    eval_y = [y_true[i] for i in indices]

    tokenizer, model = load_model_and_tokenizer(
        args.base_model, args.lora_path, torch_dtype=args.torch_dtype
    )
    model.eval()

    population = init_population(seed_weights, n=args.population_size, sigma=0.03)
    best_rewards: List[float] = []
    best_profiles: List[Dict[str, float]] = []

    for gen in range(args.generations):
        fitness_map: Dict[int, float] = {}
        for p in population:
            fitness_map[id(p)] = evaluate_profile(p, eval_chats, eval_y, tokenizer, model, args)

        best_profile = max(population, key=lambda p: fitness_map[id(p)])
        best_reward = fitness_map[id(best_profile)]
        best_rewards.append(best_reward)
        best_profiles.append(deepcopy(best_profile))

        print(f"[gen {gen}] best_reward={best_reward:.6f} best_weights={best_profile}")

        population = evolve_population(
            population,
            fitness_map,
            parents=args.parents,
            children_per_parent=args.children_per_parent,
            mutation_sigma=args.mutation_sigma,
            pop_size=args.population_size,
        )

    if args.out_dir:
        maybe_save_outputs(args.out_dir, best_rewards, best_profiles)
    if args.write_profile_path:
        out_path = Path(args.write_profile_path)
        data = json.loads(Path(args.profile_path).read_text(encoding="utf-8"))
        updated = False
        for item in data:
            if item.get("investor_type") == args.investor_type:
                item["objective_weights"] = best_profiles[-1]
                updated = True
                break
        if not updated:
            raise RuntimeError(f"investor_type not found for write: {args.investor_type}")
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=True))
        print(f"[profile-evo] wrote updated profile to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
