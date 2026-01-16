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
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    extract_pred,
    infer_chat_batch,
    load_model_and_tokenizer,
)

PROFILE_CONTEXT_RE = re.compile(r"<profile_context>\s*(\{.*?\})\s*</profile_context>", re.DOTALL)
LLM_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def maybe_tqdm(iterable, total=None, desc: str = ""):
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


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
    ap.add_argument("--profile-stats", type=str, default="data/grpo_profile_stats.csv")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--write-profile-path", type=str, default=None,
                    help="If set, write best weights into this profile JSON (in-place update).")
    ap.add_argument("--llm-guide", action="store_true", help="Use LLM to propose candidate weight sets.")
    ap.add_argument("--llm-candidates", type=int, default=6)
    ap.add_argument("--llm-model", type=str, default=None, help="LLM model name (default: DEEPSEEK_MODEL env or deepseek-chat).")
    ap.add_argument("--llm-temperature", type=float, default=0.2)
    ap.add_argument("--llm-max-tokens", type=int, default=512)
    ap.add_argument("--llm-retries", type=int, default=2)
    ap.add_argument("--llm-backoff", type=float, default=1.5)
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars if available.")
    ap.set_defaults(progress=True)
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
    progress: bool = False,
    desc: str = "infer",
) -> List[str]:
    outputs: List[str] = []
    indices = list(range(len(chats)))
    batches = list(chunked(indices, batch_size))
    iterator = batches
    if progress:
        iterator = maybe_tqdm(batches, total=len(batches), desc=desc)
    for batch in iterator:
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


def evaluate_profile_scores(
    weights: Dict[str, float],
    eval_chats: List[List[Dict[str, str]]],
    eval_y: List[float],
    tokenizer,
    model,
    args: argparse.Namespace,
    progress_tag: str = "",
) -> List[float]:
    weighted_chats = build_weighted_chats(eval_chats, weights)
    best_scores = [None] * len(eval_y)
    for k_idx in range(args.k_reasoning):
        completions = infer_in_batches(
            tokenizer,
            model,
            weighted_chats,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            progress=args.progress,
            desc=f"{progress_tag} k{k_idx+1}/{args.k_reasoning}",
        )
        preds = [extract_pred(c) for c in completions]
        for i, (pred, yt) in enumerate(zip(preds, eval_y)):
            r = value_reward(pred, yt)
            if best_scores[i] is None or r > best_scores[i]:
                best_scores[i] = r
    return [s if s is not None else -1e9 for s in best_scores]


def evaluate_profile(
    weights: Dict[str, float],
    eval_chats: List[List[Dict[str, str]]],
    eval_y: List[float],
    tokenizer,
    model,
    args: argparse.Namespace,
) -> float:
    scores = evaluate_profile_scores(weights, eval_chats, eval_y, tokenizer, model, args)
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


def maybe_save_outputs(out_dir: str, best_rewards: List[float], best_profiles: List[Dict[str, float]], llm_logs: List[dict]) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_rewards": best_rewards,
        "best_profiles": best_profiles,
        "llm_logs": llm_logs,
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


def append_progress(out_dir: str, record: dict) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prog_path = out_path / "progress.jsonl"
    with prog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_profile_stats(stats_path: Path) -> dict:
    try:
        import pandas as pd
    except Exception:
        return {}
    if not stats_path.exists():
        return {}
    df = pd.read_csv(stats_path)
    if df.empty or "type" not in df.columns:
        return {}
    df = df.set_index("type")
    return df.to_dict(orient="index")


def safe_std_from_var(v: Any) -> float:
    try:
        vv = float(v)
        if not np.isfinite(vv) or vv <= 0:
            return 0.0
        return float(np.sqrt(vv))
    except Exception:
        return 0.0


def build_llm_prompt(current_weights: Dict[str, float], eval_summary: dict, k: int) -> str:
    return (
        "Current objective_weights:\n"
        f"{json.dumps(current_weights, ensure_ascii=True)}\n\n"
        "Eval summary (GRPO-period holdout):\n"
        f"- value_reward_mean: {eval_summary.get('value_reward_mean')}\n"
        f"- value_reward_std: {eval_summary.get('value_reward_std')}\n"
        f"- x_risk_mean/std: {eval_summary.get('x_risk_mean')}/{eval_summary.get('x_risk_std')}\n"
        f"- x_herd_mean/std: {eval_summary.get('x_herd_mean')}/{eval_summary.get('x_herd_std')}\n"
        f"- x_profit_mean/std: {eval_summary.get('x_profit_mean')}/{eval_summary.get('x_profit_std')}\n\n"
        "Task:\n"
        "1) Briefly summarize the likely failure mode (1-2 sentences max).\n"
        "2) Provide 2-3 short evidence points supporting the adjustment direction.\n"
        f"3) Generate {k} diverse candidate objective_weights.\n\n"
        "Constraints:\n"
        "- Only adjust risk_aversion, herd_behavior, profit_driven.\n"
        "- Each candidate must be non-negative and sum to 1.\n"
        "- Candidates should be diverse (avoid near-duplicates).\n"
        "- Keep rationale concise (no chain-of-thought).\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "diagnosis": "short failure mode (<=2 sentences)",\n'
        '  "evidence": ["bullet 1", "bullet 2", "bullet 3 (optional)"],\n'
        '  "candidates": [\n'
        '    {"weights": {"risk_aversion": w1, "herd_behavior": w2, "profit_driven": w3}, "note": "short reason tag"}\n'
        "  ]\n"
        "}\n"
    )


def call_llm(prompt: str, args: argparse.Namespace) -> dict | None:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model_name = args.llm_model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(args.llm_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a profile evolution controller. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(content)
            except Exception:
                m = LLM_JSON_RE.search(content)
                if m:
                    return json.loads(m.group(0))
        except Exception:
            if attempt < args.llm_retries:
                time.sleep(args.llm_backoff * (attempt + 1))
                continue
            return None
    return None


def normalize_candidate(weights: dict) -> Dict[str, float] | None:
    if not isinstance(weights, dict):
        return None
    keys = ("risk_aversion", "herd_behavior", "profit_driven")
    vals = {}
    for k in keys:
        if k not in weights:
            return None
        try:
            vals[k] = float(weights[k])
        except Exception:
            return None
    for k in keys:
        if vals[k] < 0:
            vals[k] = 0.0
    return normalize_weights(vals)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    banner = "=" * 72
    print(banner)
    print("[profile-evo] config")
    print(f"  eval_size={args.eval_size} population_size={args.population_size} k_reasoning={args.k_reasoning}")
    print(f"  generations={args.generations} llm_guide={args.llm_guide} llm_candidates={args.llm_candidates}")
    print(f"  base_model={args.base_model} lora_path={args.lora_path}")
    print(banner)

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

    stats_map = load_profile_stats(Path(args.profile_stats))
    type_stats = stats_map.get(args.investor_type, {})

    population = init_population(seed_weights, n=args.population_size, sigma=0.03)
    best_rewards: List[float] = []
    best_profiles: List[Dict[str, float]] = []
    llm_logs: List[dict] = []

    for gen in range(args.generations):
        print("=" * 60)
        print(f"[gen {gen}] start  (pop={args.population_size}, eval={args.eval_size}, k={args.k_reasoning})")
        print("-" * 60)
        fitness_map: Dict[int, float] = {}
        pop_iter = population
        if args.progress:
            pop_iter = maybe_tqdm(population, total=len(population), desc=f"gen {gen} pop_eval")
        for p in pop_iter:
            fitness_map[id(p)] = evaluate_profile(p, eval_chats, eval_y, tokenizer, model, args)

        best_profile = max(population, key=lambda p: fitness_map[id(p)])
        best_reward = fitness_map[id(best_profile)]

        llm_entry: dict = {"generation": gen}
        candidates: List[Dict[str, float]] = []
        if args.llm_guide:
            print(f"[gen {gen}] llm propose -> {args.llm_candidates} candidates")
            best_scores = evaluate_profile_scores(
                best_profile,
                eval_chats,
                eval_y,
                tokenizer,
                model,
                args,
                progress_tag=f"gen {gen} best",
            )
            eval_summary = {
                "value_reward_mean": float(np.mean(best_scores)),
                "value_reward_std": float(np.std(best_scores)),
                "x_risk_mean": type_stats.get("x_risk_mean"),
                "x_risk_std": safe_std_from_var(type_stats.get("x_risk_var")),
                "x_herd_mean": type_stats.get("x_herd_mean"),
                "x_herd_std": safe_std_from_var(type_stats.get("x_herd_var")),
                "x_profit_mean": type_stats.get("x_profit_mean"),
                "x_profit_std": safe_std_from_var(type_stats.get("x_profit_var")),
            }
            prompt = build_llm_prompt(best_profile, eval_summary, args.llm_candidates)
            resp = call_llm(prompt, args)
            if resp:
                llm_entry["diagnosis"] = resp.get("diagnosis")
                llm_entry["evidence"] = resp.get("evidence")
                raw_candidates = resp.get("candidates") or []
                for c in raw_candidates:
                    weights = normalize_candidate(c.get("weights") if isinstance(c, dict) else None)
                    if weights:
                        candidates.append(weights)
                llm_entry["candidate_count"] = len(candidates)

        if candidates:
            cand_iter = candidates
            if args.progress:
                cand_iter = maybe_tqdm(candidates, total=len(candidates), desc=f"gen {gen} cand_eval")
            for cand in cand_iter:
                fitness_map[id(cand)] = evaluate_profile(
                    cand,
                    eval_chats,
                    eval_y,
                    tokenizer,
                    model,
                    args,
                )
            population_all = population + candidates
        else:
            population_all = population

        best_profile = max(population_all, key=lambda p: fitness_map[id(p)])
        best_reward = fitness_map[id(best_profile)]
        best_rewards.append(best_reward)
        best_profiles.append(deepcopy(best_profile))
        llm_entry["best_reward"] = best_reward
        llm_entry["best_profile"] = best_profile
        llm_logs.append(llm_entry)

        print(f"[gen {gen}] best_reward={best_reward:.6f} best_weights={best_profile}")
        if args.out_dir:
            append_progress(args.out_dir, llm_entry)
        print("=" * 60)

        population = evolve_population(
            population_all,
            fitness_map,
            parents=args.parents,
            children_per_parent=args.children_per_parent,
            mutation_sigma=args.mutation_sigma,
            pop_size=args.population_size,
        )

    if args.out_dir:
        maybe_save_outputs(args.out_dir, best_rewards, best_profiles, llm_logs)
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
