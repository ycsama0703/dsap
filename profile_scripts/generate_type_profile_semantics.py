#!/usr/bin/env python
"""
Generate semantic summaries for type-level profiles (one profile per type) from
artifacts/features/type_profiles.csv.

Outputs JSON with fields:
  profile_id, investor_type, profile_k, objective_weights, summary

If --use-llm is provided (with --api-key), summaries are produced by the model;
otherwise a deterministic template summary is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate semantic summaries for type-level profiles.")
    p.add_argument("--weights", required=True, help="type_profiles.csv or parquet with risk_aversion/herd_behavior/profit_driven")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--use-llm", action="store_true", help="If set, call LLM for summaries (requires --api-key)")
    p.add_argument("--model", default="deepseek-chat", help="LLM model name (e.g., deepseek-chat)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


# ----------------------------
# Load weights
# ----------------------------
def read_weights(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    required = {"type", "risk_aversion", "herd_behavior", "profit_driven"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"weights file missing columns: {missing}")

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # single profile per type
    df["profile_k"] = 0
    return df


# ----------------------------
# Prompt construction
# ----------------------------
def build_prompt(row: pd.Series) -> str:
    return (
        "You are a financial analyst. Produce a structured JSON describing this investor type's profile.\n"
        "Use qualitative labels (low/medium/high) for philosophy and constraints based on the relative weights.\n"
        "Do NOT include any numeric weights in philosophy/constraints; keep numbers only in objective_weights.\n"
        "Return EXACTLY one JSON object with fields: philosophy, constraints, summary.\n\n"
        "Field definitions:\n"
        "- philosophy: {style, activity, benchmark_orientation, horizon}\n"
        "- constraints: {risk_tolerance, turnover_constraint, concentration_constraint, herd_tendency}\n"
        "- summary: 1-2 sentences tying the profile together.\n\n"
        "Heuristics:\n"
        "- Higher risk_aversion -> risk_tolerance=low, style=defensive/balanced, benchmark_orientation=medium-high, turnover_constraint=high.\n"
        "- Higher herd_behavior -> herd_tendency=high, benchmark_orientation=high.\n"
        "- Higher profit_driven -> style=growth/active, activity=high_turnover, benchmark_orientation=low, turnover_constraint=low-mid.\n"
        "- If weights are balanced, use medium labels.\n\n"
        f"Investor type: {row['type']}\n"
        f"risk_aversion weight: {row['risk_aversion']:.4f}\n"
        f"herd_behavior weight: {row['herd_behavior']:.4f}\n"
        f"profit_driven weight: {row['profit_driven']:.4f}\n"
        "Output JSON schema:\n"
        "{\n"
        '  "philosophy": {\n'
        '    "style": "...", "activity": "...", "benchmark_orientation": "...", "horizon": "..."\n'
        "  },\n"
        '  "constraints": {\n'
        '    "risk_tolerance": "...", "turnover_constraint": "...", "concentration_constraint": "...", "herd_tendency": "..."\n'
        "  },\n"
        '  "summary": "one or two sentences (no numbers)"\n'
        "}\n"
    )


# ----------------------------
# DeepSeek LLM call (OpenAI-compatible)
# ----------------------------
def call_llm(
    prompt: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    if OpenAI is None:
        raise ImportError("openai package not installed; pip install openai")

    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("api-key required for LLM")

    # DeepSeek uses OpenAI-compatible API but requires base_url
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )

    resp = client.chat.completions.create(
        model=model,  # e.g. "deepseek-chat"
        messages=[
            {"role": "system", "content": "You are a financial reasoning assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        # try to extract JSON substring
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            raise ValueError(f"LLM did not return valid JSON: {content}")


# ----------------------------
# Deterministic fallback summary
# ----------------------------
def deterministic_summary(row: pd.Series) -> str:
    items = [
        ("risk aversion", row["risk_aversion"]),
        ("herd behavior", row["herd_behavior"]),
        ("profit driven motives", row["profit_driven"]),
    ]
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    strongest, mid, weakest = sorted_items

    return (
        f"{row['type']} investors emphasize {strongest[0]} most strongly, "
        f"show a moderate focus on {mid[0]}, "
        f"and place the least emphasis on {weakest[0]}."
    )


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    args = parse_args()
    df = read_weights(Path(args.weights))

    out_profiles: List[Dict] = []

    for _, row in df.iterrows():
        ph = {}
        cons = {}
        summary = ""
        if args.use_llm:
            result = call_llm(
                build_prompt(row),
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if isinstance(result, dict):
                ph = result.get("philosophy", {}) if isinstance(result.get("philosophy"), dict) else {}
                cons = result.get("constraints", {}) if isinstance(result.get("constraints"), dict) else {}
                summary = result.get("summary", "") if isinstance(result.get("summary"), str) else ""
        if not summary:
            summary = deterministic_summary(row)

        out_profiles.append(
            {
                "profile_id": f"{row['type']}_p0",
                "investor_type": row["type"],
                "profile_k": int(row["profile_k"]),
                "objective_weights": {
                    "risk_aversion": float(row["risk_aversion"]),
                    "herd_behavior": float(row["herd_behavior"]),
                    "profit_driven": float(row["profit_driven"]),
                },
                "philosophy": ph,
                "constraints": cons,
                "summary": summary,
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_profiles, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {len(out_profiles)} profiles -> {out_path}")


if __name__ == "__main__":
    main()
