#!/usr/bin/env python
"""
debug_eval_outputs.py

Run inference on the test set and dump raw outputs together with parsed
predictions and absolute errors to a CSV for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_pred,
    LOG_EPS,
)


def chunked(indices: List[int], size: int):
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump raw model outputs with parsed predictions and absolute errors."
    )
    ap.add_argument("--test-path", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora-path", type=str, default=None, help="LoRA checkpoint directory")
    ap.add_argument("--out-csv", type=str, default="outputs/debug_eval_outputs.csv")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-think", dest="force_think", action="store_true")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false")
    ap.set_defaults(force_think=False)
    args = ap.parse_args()

    chats, y_true, quarters, ids, holding_ts, permnos, dates = build_eval_inputs(args.test_path)
    if args.limit is not None:
        limit = min(args.limit, len(chats))
        chats = chats[:limit]
        y_true = y_true[:limit]
        quarters = quarters[:limit]
        ids = ids[:limit]
        holding_ts = holding_ts[:limit]
        permnos = permnos[:limit]
        dates = dates[:limit]

    tokenizer, model = load_model_and_tokenizer(
        args.base_model, args.lora_path, torch_dtype=args.torch_dtype
    )
    model.eval()

    raw_outputs: List[str] = []
    preds: List[float | None] = []

    indices = list(range(len(chats)))
    iterator = chunked(indices, args.batch_size)
    total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(iterator, total=total_batches, desc="debug_eval")
    except Exception:
        pass

    for batch in iterator:
        batch_msgs = [chats[i] for i in batch]
        completions = infer_chat_batch(
            tokenizer,
            model,
            batch_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            force_think=args.force_think,
        )
        for idx, completion in zip(batch, completions):
            raw_outputs.append(completion)
            preds.append(extract_pred(completion))

    def to_tp1(log_delta, holding):
        if log_delta is None or holding is None:
            return None
        try:
            return float(np.exp(log_delta) * (holding + LOG_EPS) - LOG_EPS)
        except Exception:
            return None

    rows: List[Dict[str, Any]] = []
    for idx, quarter, yt, ht, raw, pred, pm, dt in zip(ids, quarters, y_true, holding_ts, raw_outputs, preds, permnos, dates):
        entry: Dict[str, Any] = {
            "id": idx,
            "quarter": quarter,
            "date": dt,
            "permno": pm,
            "holding_t": ht,
            "y_true": yt,
            "raw_output": raw,
            "parsed_pred": pred,
            "true_tp1": to_tp1(yt, ht),
            "pred_tp1": to_tp1(pred, ht),
            "abs_error": None,
            "abs_tp1_error": None,
        }
        if pred is not None and yt is not None:
            entry["abs_error"] = abs(pred - yt)
        if entry["pred_tp1"] is not None and entry["true_tp1"] is not None:
            entry["abs_tp1_error"] = abs(entry["pred_tp1"] - entry["true_tp1"])
        rows.append(entry)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "quarter",
                "date",
                "permno",
                "holding_t",
                "y_true",
                "raw_output",
                "parsed_pred",
                "abs_error",
                "true_tp1",
                "pred_tp1",
                "abs_tp1_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[debug] Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
