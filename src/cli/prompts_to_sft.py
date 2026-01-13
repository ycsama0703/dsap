# src/cli/prompts_to_sft.py

from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from typing import Callable, Dict


LOG_EPS = 1e-6


def _safe_nonneg(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
        if v < 0:
            return None
        return v
    except Exception:
        return None


def _resolve_absolute(rec: dict) -> float | None:
    """Resolve absolute holding_tp1 value."""
    try:
        if rec.get("label") is not None:
            return float(rec["label"])  # absolute
        if rec.get("label_tp1") is not None:
            return float(rec["label_tp1"])  # absolute
        if rec.get("label_delta_absolute") is not None and rec.get("holding_t") is not None:
            return float(rec["holding_t"]) + float(rec["label_delta_absolute"])  # reconstruct from absolute delta
        if rec.get("label_delta") is not None and rec.get("holding_t") is not None:
            # legacy fallback where label_delta stored absolute change
            return float(rec["holding_t"]) + float(rec["label_delta"])
    except Exception:
        return None
    return None


def _resolve_log_delta(rec: dict) -> float | None:
    """Resolve holding_log_delta value."""
    try:
        if rec.get("label_log_delta") is not None:
            return float(rec["label_log_delta"])
        if rec.get("label_delta") is not None:
            # allow fallback when upstream already overwrote label_delta with log delta
            return float(rec["label_delta"])
        ht = rec.get("holding_t")
        tp1 = rec.get("label_tp1") if rec.get("label_tp1") is not None else rec.get("label")
        if tp1 is None and rec.get("label_delta_absolute") is not None and ht is not None:
            tp1 = float(ht) + float(rec["label_delta_absolute"])
        ht_val = _safe_nonneg(ht)
        tp1_val = _safe_nonneg(tp1)
        if ht_val is None or tp1_val is None:
            return None
        return math.log((tp1_val + LOG_EPS) / (ht_val + LOG_EPS))
    except Exception:
        return None
    return None


def _format_float(value: float, decimals: int) -> float:
    """Round float to the desired decimal places while avoiding scientific notation in JSON."""
    return float(f"{value:.{decimals}f}")


def _parse_block(prompt: str, tag: str) -> Dict[str, float]:
    """Parse a {t data} block into a dict of numeric features."""
    pattern = re.compile(
        rf"{re.escape(tag)}\s*(?:\r?\n)+(?P<body>.*?)(?=(\r?\n\s*\{{)|\Z)",
        re.IGNORECASE | re.S,
    )
    m = pattern.search(prompt)
    if not m:
        return {}
    body = m.group("body")
    values: Dict[str, float] = {}
    for token in re.split(r"[,\n]", body):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, val = token.split("=", 1)
        elif ":" in token:
            key, val = token.split(":", 1)
        else:
            continue
        key = key.strip()
        val = val.strip()
        try:
            values[key] = float(val)
        except ValueError:
            continue
    return values


def _build_auto_think(prompt: str, decimals: int = 2) -> str:
    """Create a lightweight reasoning snippet using t-1 -> t deltas."""
    latest = _parse_block(prompt, "{t data}")
    prev = _parse_block(prompt, "{t-1 data}")
    parts = []
    for key in ("me", "profit", "beta", "aum", "outAUM", "price"):
        if key in latest and key in prev:
            try:
                delta = latest[key] - prev[key]
                parts.append(f"{key}Δ={delta:+.{decimals}f}")
            except Exception:
                continue
    holding_t = latest.get("holding")
    if holding_t is not None:
        try:
            parts.append(f"holding_t={holding_t:.{decimals}f}")
        except Exception:
            pass
    if not parts:
        return "<think>Summarise key fundamental shifts and justify the adjustment.</think>"
    summary = ", ".join(parts[:6])
    return f"<think>{summary}. Explain how this guides the adjustment.</think>"


__all__ = ["_build_auto_think"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="input prompts jsonl or directory with jsonl files")
    ap.add_argument("--out", dest="out", type=str, default="artifacts/sft/sft_train.jsonl")
    ap.add_argument("--system", type=str,
                    default="You are a quantitative portfolio manager.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-think", dest="with_think", action="store_true", default=True,
                    help="Embed a <think>...</think> block before the final answer (default: enabled)")
    ap.add_argument("--no-think", dest="with_think", action="store_false",
                    help="Do not include the <think> block")
    ap.add_argument("--contract-mode", choices=["absolute", "log_delta", "delta"], default="log_delta",
                    help="Select prediction target: absolute holding_tp1 or log change in holdings.")
    ap.add_argument("--think-template", type=str,
                    default="",
                    help="Inner text for the <think> block when --with-think is enabled.")
    ap.add_argument("--decimals", type=int, default=2,
                    help="Maximum decimal places for numeric outputs (default: 2).")
    args = ap.parse_args()

    inp = Path(args.inp)
    files = [inp] if inp.is_file() else sorted(inp.glob("*.jsonl"))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    resolver: Callable[[dict], float | None]
    mode = args.contract_mode
    if mode == "delta":
        mode = "log_delta"
    if mode == "absolute":
        resolver = _resolve_absolute
        answer_key = "holding_tp1"
    elif mode == "log_delta":
        resolver = _resolve_log_delta
        answer_key = "holding_log_delta"
    else:
        raise ValueError(f"Unsupported contract_mode: {args.contract_mode}")

    n = 0
    with outp.open("w", encoding="utf-8") as fout:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    prompt = rec.get("prompt") or rec.get("query")
                    if not prompt:
                        continue
                    label_val = resolver(rec)
                    profile_val = rec.get("label_profile_k") or rec.get("profile_k")
                    if label_val is None or profile_val is None:
                        continue  # skip if required label missing
                    value = _format_float(float(label_val), args.decimals)
                    resp = json.dumps({answer_key: value}, ensure_ascii=False)
                    profile_msg = f"<profile>{{\"profile_k\": {int(profile_val)}}}</profile>"

                    msgs = [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": prompt},
                    ]

                    assistant_msgs = []
                    # profile block (supervised)
                    assistant_msgs.append({"role": "assistant", "content": profile_msg, "loss": True})
                    if args.with_think:
                        think_text = args.think_template.strip()
                        if not think_text:
                            think_text = _build_auto_think(prompt, args.decimals)
                        if think_text:
                            think_lower = think_text.lower()
                            if not think_lower.startswith("<think>"):
                                think_text = f"<think>{think_text}</think>"
                            assistant_msgs.append({"role": "assistant", "content": think_text, "loss": True})
                    assistant_msgs.append({"role": "assistant", "content": f"<answer>{resp}</answer>", "loss": True})

                    out = {"messages": msgs + assistant_msgs}

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n += 1
                    if args.limit and n >= args.limit:
                        break
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} samples -> {outp}")


if __name__ == "__main__":
    main()
