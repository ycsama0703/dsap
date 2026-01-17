#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal profile evolution experiment:
- Keeps SFT/GRPO unchanged.
- Evolves objective_weights via outcome-guided accuracy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    infer_chat_batch,
    load_model_and_tokenizer,
)

PROFILE_CONTEXT_RE = re.compile(r"<profile_context>\s*(\{.*?\})\s*</profile_context>", re.DOTALL)
LLM_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


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
    ap.add_argument("--eval-size", type=int, default=5)
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
    ap.add_argument("--llm-sample-best", type=int, default=2, help="How many best (lowest error) samples to show LLM.")
    ap.add_argument("--llm-sample-worst", type=int, default=2, help="How many worst (highest error) samples to show LLM.")
    ap.add_argument("--llm-sample-max-chars", type=int, default=600, help="Max chars of raw output per sample.")
    ap.add_argument("--llm-only", action="store_true", help="Disable random mutation; evolve only via LLM candidates.")
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


def value_reward(pred: float, y_true: float) -> float:
    return -abs(pred - y_true)


def evaluate_profile_scores(
    weights: Dict[str, float],
    eval_chats: List[List[Dict[str, str]]],
    eval_y: List[float],
    tokenizer,
    model,
    args: argparse.Namespace,
    progress_tag: str = "",
    eval_meta: List[Dict[str, Any]] | None = None,
    return_details: bool = False,
) -> List[float] | Tuple[List[float], List[Dict[str, Any]]]:
    weighted_chats = build_weighted_chats(eval_chats, weights)
    best_scores = [None] * len(eval_y)
    best_preds: List[float | None] = [None] * len(eval_y)
    best_completions: List[str | None] = [None] * len(eval_y)
    best_k: List[int | None] = [None] * len(eval_y)
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
        preds = [strict_extract_pred(c) for c in completions]
        for i, (pred, yt) in enumerate(zip(preds, eval_y)):
            if pred is None or yt is None:
                continue
            r = value_reward(pred, yt)
            if best_scores[i] is None or r > best_scores[i]:
                best_scores[i] = r
                best_preds[i] = pred
                best_completions[i] = completions[i]
                best_k[i] = k_idx + 1
    scores = [s for s in best_scores if s is not None]
    if not return_details:
        return scores
    details: List[Dict[str, Any]] = []
    for i, score in enumerate(best_scores):
        meta = eval_meta[i] if eval_meta is not None else {}
        pred = best_preds[i]
        yt = eval_y[i]
        valid = pred is not None and yt is not None and score is not None
        abs_err = None if not valid else abs(pred - yt)
        details.append(
            {
                **meta,
                "y_true": yt,
                "pred": pred,
                "abs_error": abs_err,
                "reward": score if valid else None,
                "best_k": best_k[i],
                "completion": best_completions[i],
                "valid": valid,
            }
        )
    return scores, details


def evaluate_profile(
    weights: Dict[str, float],
    eval_chats: List[List[Dict[str, str]]],
    eval_y: List[float],
    tokenizer,
    model,
    args: argparse.Namespace,
) -> float:
    scores = evaluate_profile_scores(weights, eval_chats, eval_y, tokenizer, model, args)
    return float(np.mean(scores)) if scores else -1e9


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


def extract_think(text: str | None) -> str | None:
    if not text:
        return None
    m = THINK_RE.search(text)
    if not m:
        return None
    content = m.group(1).strip()
    return content or None


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    content = m.group(1).strip()
    return content or None


def strict_extract_pred(text: str | None) -> float | None:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    content = m.group(1).strip()
    try:
        obj = json.loads(content)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("holding_log_delta") is None:
        return None
    try:
        val = float(obj["holding_log_delta"])
    except Exception:
        return None
    if not np.isfinite(val) or not (-10 < val < 10):
        return None
    return val


def extract_profile_strength(user_text: str | None) -> Dict[str, Any]:
    if not user_text:
        return {}
    m = PROFILE_CONTEXT_RE.search(user_text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return {}
    ps = obj.get("profile_strength")
    return ps if isinstance(ps, dict) else {}


def truncate_text(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def select_llm_examples(details: List[Dict[str, Any]], n_best: int, n_worst: int) -> Dict[str, List[Dict[str, Any]]]:
    scored = [d for d in details if d.get("abs_error") is not None]
    if not scored:
        return {"worst": [], "best": []}
    scored = sorted(scored, key=lambda d: d["abs_error"])
    best = scored[: max(n_best, 0)]
    worst = scored[-max(n_worst, 0) :] if n_worst > 0 else []
    seen = set()
    def _uniq(items):
        out = []
        for d in items:
            key = (d.get("id"), d.get("permno"), d.get("date"))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out
    return {"worst": _uniq(worst), "best": _uniq(best)}


def count_valid_details(details: List[Dict[str, Any]]) -> int:
    return sum(1 for d in details if d.get("valid"))


def write_candidate_samples_csv(
    out_dir: str,
    gen: int,
    stage: str,
    candidate_idx: int,
    weights: Dict[str, float],
    details: List[Dict[str, Any]],
) -> None:
    debug_dir = Path(out_dir) / "debug" / f"gen_{gen}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{stage}_samples.csv"
    write_header = not out_path.exists()
    fieldnames = [
        "generation",
        "stage",
        "candidate_idx",
        "risk_aversion",
        "herd_behavior",
        "profit_driven",
        "ticker",
        "company",
        "id",
        "quarter",
        "date",
        "permno",
        "holding_t",
        "y_true",
        "pred",
        "abs_error",
        "reward",
        "valid",
        "best_k",
        "profile_strength_mode",
        "x_risk_mean",
        "x_herd_mean",
        "x_profit_mean",
        "overall_mean",
        "raw_output",
        "reason",
        "answer_body",
    ]
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for d in details:
            completion = d.get("completion")
            writer.writerow(
                {
                    "generation": gen,
                    "stage": stage,
                    "candidate_idx": candidate_idx,
                    "risk_aversion": weights.get("risk_aversion"),
                    "herd_behavior": weights.get("herd_behavior"),
                    "profit_driven": weights.get("profit_driven"),
                    "ticker": d.get("ticker"),
                    "company": d.get("company"),
                    "id": d.get("id"),
                    "quarter": d.get("quarter"),
                    "date": d.get("date"),
                    "permno": d.get("permno"),
                    "holding_t": d.get("holding_t"),
                    "y_true": d.get("y_true"),
                    "pred": d.get("pred"),
                    "abs_error": d.get("abs_error"),
                    "reward": d.get("reward"),
                    "valid": d.get("valid"),
                    "best_k": d.get("best_k"),
                    "profile_strength_mode": d.get("profile_strength_mode"),
                    "x_risk_mean": d.get("x_risk_mean"),
                    "x_herd_mean": d.get("x_herd_mean"),
                    "x_profit_mean": d.get("x_profit_mean"),
                    "overall_mean": d.get("overall_mean"),
                    "raw_output": completion,
                    "reason": extract_think(completion),
                    "answer_body": extract_answer(completion),
                }
            )


def write_candidate_debug(
    out_dir: str,
    gen: int,
    stage: str,
    candidate_idx: int,
    weights: Dict[str, float],
    scores: List[float],
    details: List[Dict[str, Any]],
    note: str | None = None,
) -> None:
    debug_dir = Path(out_dir) / "debug" / f"gen_{gen}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{stage}.jsonl"
    valid_abs = [d.get("abs_error") for d in details if d.get("abs_error") is not None]
    valid_count = sum(1 for d in details if d.get("valid"))
    record = {
        "generation": gen,
        "stage": stage,
        "candidate_idx": candidate_idx,
        "weights": weights,
        "mean_reward": float(np.mean(scores)) if scores else None,
        "std_reward": float(np.std(scores)) if scores else None,
        "mean_abs_error": float(np.mean(valid_abs)) if valid_abs else None,
        "valid_samples": valid_count,
        "invalid_samples": len(details) - valid_count,
        "note": note,
        "details": details,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    write_candidate_samples_csv(out_dir, gen, stage, candidate_idx, weights, details)


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


def load_permno_mapping(path: Path) -> Dict[int, Dict[str, str]]:
    try:
        import pandas as pd
    except Exception:
        return {}
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "PERMNO" not in df.columns:
        return {}
    df = df.dropna(subset=["PERMNO"])
    df["PERMNO"] = df["PERMNO"].astype(int)
    df = df.sort_values(by="PERMNO").drop_duplicates(subset=["PERMNO"], keep="last")
    mapping: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        mapping[int(row["PERMNO"])] = {
            "ticker": str(row["TICKER"]) if "TICKER" in row and not isinstance(row["TICKER"], float) else "",
            "company": str(row["COMNAM"]) if "COMNAM" in row and not isinstance(row["COMNAM"], float) else "",
        }
    return mapping


def safe_std_from_var(v: Any) -> float:
    try:
        vv = float(v)
        if not np.isfinite(vv) or vv <= 0:
            return 0.0
        return float(np.sqrt(vv))
    except Exception:
        return 0.0


def build_llm_prompt(current_weights: Dict[str, float], eval_summary: dict, k: int, examples: dict | None = None) -> str:
    examples = examples or {}
    def _fmt(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "[]"
        payload = []
        for d in items:
            raw_out = d.get("completion")
            payload.append(
                {
                    "y_true": d.get("y_true"),
                    "pred": d.get("pred"),
                    "reason": extract_think(raw_out),
                }
            )
        return json.dumps(payload, ensure_ascii=True)

    return (
        "Current objective_weights:\n"
        f"{json.dumps(current_weights, ensure_ascii=True)}\n\n"
        "Eval summary (GRPO-period holdout):\n"
        f"- value_reward_mean: {eval_summary.get('value_reward_mean')}\n"
        f"- value_reward_std: {eval_summary.get('value_reward_std')}\n"
        f"- x_risk_mean/std: {eval_summary.get('x_risk_mean')}/{eval_summary.get('x_risk_std')}\n"
        f"- x_herd_mean/std: {eval_summary.get('x_herd_mean')}/{eval_summary.get('x_herd_std')}\n"
        f"- x_profit_mean/std: {eval_summary.get('x_profit_mean')}/{eval_summary.get('x_profit_std')}\n\n"
        "Representative samples (worst/best):\n"
        f"- worst_samples: {_fmt(examples.get('worst', []))}\n"
        f"- best_samples: {_fmt(examples.get('best', []))}\n\n"
        "Task:\n"
        "1) Briefly summarize the likely failure mode in terms of objective_weights being too high/low\n"
        "   (1-2 sentences max, must mention risk_aversion/herd_behavior/profit_driven explicitly).\n"
        "2) Provide 2-3 short evidence points supporting how those weight shifts reduce prediction bias.\n"
        f"3) Generate {k} diverse candidate objective_weights consistent with that diagnosis.\n\n"
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
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "_raw" not in parsed:
                    parsed["_raw"] = content
                return parsed
            except Exception:
                m = LLM_JSON_RE.search(content)
                if m:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict) and "_raw" not in parsed:
                        parsed["_raw"] = content
                    return parsed
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

    chats, y_true, quarters, ids, holding_ts, permnos, dates = build_eval_inputs(args.test_path)
    if not chats:
        raise RuntimeError("no eval samples found in test set")
    indices = list(range(len(chats)))
    random.shuffle(indices)
    eval_n = min(args.eval_size, len(indices))
    indices = indices[:eval_n]
    eval_chats = [chats[i] for i in indices]
    eval_y = [y_true[i] for i in indices]
    permno_map = load_permno_mapping(Path("data/ticker_mapping.csv"))
    eval_meta = []
    for j, i in enumerate(indices):
        ps = extract_profile_strength(eval_chats[j][1]["content"])
        eval_meta.append(
            {
                "id": ids[i],
                "quarter": quarters[i],
                "permno": permnos[i],
                "date": dates[i],
                "holding_t": holding_ts[i],
                "ticker": permno_map.get(permnos[i] or -1, {}).get("ticker"),
                "company": permno_map.get(permnos[i] or -1, {}).get("company"),
                "profile_strength_mode": ps.get("mode"),
                "x_risk_mean": ps.get("x_risk_mean"),
                "x_herd_mean": ps.get("x_herd_mean"),
                "x_profit_mean": ps.get("x_profit_mean"),
                "overall_mean": ps.get("overall_mean"),
            }
        )

    tokenizer, model = load_model_and_tokenizer(
        args.base_model, args.lora_path, torch_dtype=args.torch_dtype
    )
    model.eval()

    stats_map = load_profile_stats(Path(args.profile_stats))
    type_stats = stats_map.get(args.investor_type, {})

    if args.llm_only:
        population = [deepcopy(seed_weights) for _ in range(args.population_size)]
    else:
        population = init_population(seed_weights, n=args.population_size, sigma=0.03)
    best_rewards: List[float] = []
    best_profiles: List[Dict[str, float]] = []
    llm_logs: List[dict] = []

    for gen in range(args.generations):
        print("=" * 60)
        print(f"[gen {gen}] start  (pop={args.population_size}, eval={args.eval_size}, k={args.k_reasoning})")
        print("-" * 60)
        fitness_map: Dict[int, float] = {}
        pop_valid_counts: List[int] = []
        pop_total_counts: List[int] = []
        pop_iter = list(enumerate(population))
        if args.progress:
            pop_iter = maybe_tqdm(pop_iter, total=len(population), desc=f"gen {gen} pop_eval")
        for idx, p in pop_iter:
            scores, details = evaluate_profile_scores(
                p,
                eval_chats,
                eval_y,
                tokenizer,
                model,
                args,
                progress_tag=f"gen {gen} pop {idx+1}/{len(population)}",
                eval_meta=eval_meta,
                return_details=True,
            )
            fitness_map[id(p)] = float(np.mean(scores)) if scores else -1e9
            pop_valid_counts.append(count_valid_details(details))
            pop_total_counts.append(len(details))
            if args.out_dir:
                write_candidate_debug(args.out_dir, gen, "population", idx, p, scores, details)

        if pop_valid_counts:
            mean_valid = float(np.mean(pop_valid_counts))
            min_valid = min(pop_valid_counts)
            max_valid = max(pop_valid_counts)
            mean_total = float(np.mean(pop_total_counts))
            print(
                f"[gen {gen}] population valid_mean={mean_valid:.2f} "
                f"min={min_valid} max={max_valid} total_mean={mean_total:.2f}"
            )

        best_profile = max(population, key=lambda p: fitness_map[id(p)])
        best_reward = fitness_map[id(best_profile)]

        llm_entry: dict = {"generation": gen}
        candidates: List[Dict[str, float]] = []
        if args.llm_guide:
            print(f"[gen {gen}] llm propose -> {args.llm_candidates} candidates")
            best_scores, best_details = evaluate_profile_scores(
                best_profile,
                eval_chats,
                eval_y,
                tokenizer,
                model,
                args,
                progress_tag=f"gen {gen} best",
                eval_meta=eval_meta,
                return_details=True,
            )
            eval_summary = {
                "value_reward_mean": float(np.mean(best_scores)) if best_scores else None,
                "value_reward_std": float(np.std(best_scores)) if best_scores else None,
                "x_risk_mean": type_stats.get("x_risk_mean"),
                "x_risk_std": safe_std_from_var(type_stats.get("x_risk_var")),
                "x_herd_mean": type_stats.get("x_herd_mean"),
                "x_herd_std": safe_std_from_var(type_stats.get("x_herd_var")),
                "x_profit_mean": type_stats.get("x_profit_mean"),
                "x_profit_std": safe_std_from_var(type_stats.get("x_profit_var")),
            }
            trimmed = []
            for d in best_details:
                raw_out = truncate_text(d.get("completion"), args.llm_sample_max_chars)
                trimmed.append({**d, "completion": raw_out})
            examples = select_llm_examples(trimmed, args.llm_sample_best, args.llm_sample_worst)
            prompt = build_llm_prompt(best_profile, eval_summary, args.llm_candidates, examples=examples)
            resp = call_llm(prompt, args)
            if resp:
                llm_entry["llm_prompt"] = prompt
                llm_entry["llm_raw"] = resp.get("_raw")
                llm_entry["diagnosis"] = resp.get("diagnosis")
                llm_entry["evidence"] = resp.get("evidence")
                raw_candidates = resp.get("candidates") or []
                for c in raw_candidates:
                    weights = normalize_candidate(c.get("weights") if isinstance(c, dict) else None)
                    if weights:
                        candidates.append(weights)
                llm_entry["candidate_count"] = len(candidates)
            else:
                print(
                    f"[gen {gen}] llm call failed; no candidates generated "
                    "(check API key/model/base URL connectivity)"
                )

        if candidates:
            cand_valid_counts: List[int] = []
            cand_total_counts: List[int] = []
            cand_iter = list(enumerate(candidates))
            if args.progress:
                cand_iter = maybe_tqdm(cand_iter, total=len(candidates), desc=f"gen {gen} cand_eval")
            for idx, cand in cand_iter:
                scores, details = evaluate_profile_scores(
                    cand,
                    eval_chats,
                    eval_y,
                    tokenizer,
                    model,
                    args,
                    progress_tag=f"gen {gen} cand {idx+1}/{len(candidates)}",
                    eval_meta=eval_meta,
                    return_details=True,
                )
                fitness_map[id(cand)] = float(np.mean(scores)) if scores else -1e9
                cand_valid_counts.append(count_valid_details(details))
                cand_total_counts.append(len(details))
                if args.out_dir:
                    write_candidate_debug(args.out_dir, gen, "llm_candidates", idx, cand, scores, details)
            population_all = population + candidates
            if cand_valid_counts:
                mean_valid = float(np.mean(cand_valid_counts))
                min_valid = min(cand_valid_counts)
                max_valid = max(cand_valid_counts)
                mean_total = float(np.mean(cand_total_counts))
                print(
                    f"[gen {gen}] llm_candidates valid_mean={mean_valid:.2f} "
                    f"min={min_valid} max={max_valid} total_mean={mean_total:.2f}"
                )
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

        if args.llm_only:
            population = sorted(
                population_all, key=lambda p: fitness_map[id(p)], reverse=True
            )[: args.population_size]
        else:
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
