#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LLM connectivity check using OpenAI-compatible API.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM API connectivity test.")
    ap.add_argument("--prompt", type=str, default="Say 'ok' and nothing else.")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    return ap.parse_args()


def main() -> int:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - env-specific
        print(f"[llm-test] failed to import openai: {exc}", file=sys.stderr)
        return 2

    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[llm-test] missing DEEPSEEK_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        return 2

    model = args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = args.base_url or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Reply to the user prompt."},
            {"role": "user", "content": args.prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    print("[llm-test] ok")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print("response=" + content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
