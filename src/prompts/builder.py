from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import math
import pandas as pd


@dataclass
class PromptRow:
    mgrno: str | int
    permno: str | int
    investor_type: str
    # current t snapshot (required)
    holding_t: Optional[float]
    holding_t1: Optional[float]
    # optional profile info
    profile_k: Optional[int] = None
    prev_profile_k: Optional[int] = None
    objective_weights: Optional[dict] = None
    stock_vol_q_prev: Optional[float] = None
    stock_ln_volume_q_prev: Optional[float] = None
    shares: Optional[float] = None
    me: Optional[float] = None
    be: Optional[float] = None
    profit: Optional[float] = None
    Gat: Optional[float] = None
    beta: Optional[float] = None
    aum: Optional[float] = None
    outaum: Optional[float] = None
    prc: Optional[float] = None
    sp500_weight: Optional[float] = None
    date: Optional[str] = None


def _fmt(x):
    if x is None:
        return "NA"
    try:
        v = float(x)
        if pd.isna(v):
            return "NA"
        # decimal formatting: up to 6 decimals, no scientific notation
        s = f"{v:.6f}".rstrip('0').rstrip('.')
        if s == "-0":
            s = "0"
        return s
    except Exception:
        return str(x)


LOG_EPS = 1e-6


def _safe_nonneg(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if pd.isna(v) or v < 0:
            return None
        return v
    except Exception:
        return None


def _row_payload(r: PromptRow) -> dict[str, str]:
    """Return a JSON-serializable snapshot of a PromptRow with consistent formatting."""
    def _fmt_or_na(val):
        if val is None:
            return "NA"
        return _fmt(val)

    return {
        "me": _fmt_or_na(r.me),
        "be": _fmt_or_na(r.be),
        "profit": _fmt_or_na(r.profit),
        "Gat": _fmt_or_na(r.Gat),
        "beta": _fmt_or_na(r.beta),
        "aum": _fmt_or_na(r.aum),
        "outAUM": _fmt_or_na(r.outaum),
        "holding": _fmt_or_na(r.holding_t),
        "price": _fmt_or_na(r.prc),
        "sp500_weight": _fmt_or_na(r.sp500_weight),
        "stock_vol_q_prev": _fmt_or_na(r.stock_vol_q_prev),
        "stock_ln_volume_q_prev": _fmt_or_na(r.stock_ln_volume_q_prev),
    }


def build_history_prompt(
    hist_rows: Sequence[PromptRow],
    hide_date: bool = True,
    target: str = "delta",  # "delta" or "absolute"
    strict_contract: bool = True,
) -> tuple[str, dict]:
    """Build a prompt from historical rows up to current t.

    Returns (prompt_text, extras_dict) where extras includes fields used for labels.
    hist_rows: order oldest -> newest, length >= 2 (supports 2 or 4 by default)
    """
    assert len(hist_rows) >= 1, "need at least 1 row: t"
    r_t = hist_rows[-1]
    # keep last up to 4 rows for labeling convenience; labels are dynamic
    window_len = len(hist_rows)

    role_raw = r_t.investor_type or "Unknown"
    role = role_raw.strip() if isinstance(role_raw, str) else str(role_raw)
    role_phrase = role.lower()
    if role_phrase.endswith("s"):
        role_phrase = role_phrase[:-1]
    mgrno = r_t.mgrno
    permno = r_t.permno

    def block(title: str, r: PromptRow) -> str:
        return (
            f"{title}\n"
            f"me={_fmt(r.me)}, be={_fmt(r.be)}, profit={_fmt(r.profit)}, Gat={_fmt(r.Gat)}, beta={_fmt(r.beta)}\n"
            f"aum={_fmt(r.aum)}, outAUM={_fmt(r.outaum)}, holding={_fmt(r.holding_t)}"
            + (f", price={_fmt(r.prc)}" if r.prc is not None else "")
            + (f", sp500_weight={_fmt(r.sp500_weight)}" if r.sp500_weight is not None else "")
        )

    # history changes (if available)
    def change(prev: PromptRow, curr: PromptRow) -> str:
        prev_val = _safe_nonneg(prev.holding_t)
        curr_val = _safe_nonneg(curr.holding_t)
        if prev_val is None or curr_val is None:
            return "NA"
        try:
            ratio = math.log((curr_val + LOG_EPS) / (prev_val + LOG_EPS))
        except Exception:
            return "NA"
        s = f"{ratio:.4f}".rstrip('0').rstrip('.')
        if s == "-0":
            s = "0"
        return s

    if strict_contract:
        header = (
            f"Act as a {role_phrase} portfolio manager.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n"
            f"\"holding_log_delta\" means the log change in holdings.\n"
            f"Formula: holding_log_delta = log((holding_tp1 + {LOG_EPS:.0e}) / (holding_t + {LOG_EPS:.0e})).\n"
            f"This smooths scale differences while keeping the adjustment direction.\n\n"
        )
    else:
        header = (
            f"You are a {role}.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n\n"
        )

    # Build history blocks dynamically (exclude current t)
    hist_only = hist_rows[:-1]
    hist_lines = []
    for i, row in enumerate(hist_only):
        offset = len(hist_only) - i
        hist_lines.append(f"{{t-{offset} data}}\n{block('', row)}")
        # change to next point (next historical if exists, else current t)
        next_row = hist_only[i + 1] if i + 1 < len(hist_only) else r_t
        hist_lines.append(f"holding_log_delta: {change(row, next_row)}")
        hist_lines.append("")  # blank line separator
    hist = "\n".join(hist_lines)
    if hist and not hist.endswith("\n\n"):
        hist += "\n"

    now_block = f"Now, these are the new data:\n{block('{t data}', r_t)}\n\n"

    if strict_contract:
        instructions = (
            "OUTPUT FORMAT & REASONING GUIDE\n\n"
            "Return exactly three blocks in order:\n"
            "1) <profile>{\"profile_k\": <int>}</profile>\n"
            "2) <think>...profile-aligned reasoning (≤3 sentences)...</think>\n"
            "3) <answer>{\"holding_log_delta\": <float with 2 decimals>}</answer>\n"
            "Use log change: holding_log_delta = log((holding_tp1 + "
            f"{LOG_EPS:.0e}) / (holding_t + {LOG_EPS:.0e})).\n\n"
            "Example (structure only):\n"
            "<profile>{\"profile_k\": 2}</profile>\n"
            "<think>...</think>\n"
            "<answer>{\"holding_log_delta\": 0.00}</answer>\n\n"
        )
    else:
        instructions = ""

    prompt = header
    prompt += (f"Ticker: {permno}\n\n" if permno is not None else "")
    prompt += hist + now_block
    prompt += instructions

    history_dict = {"t": _row_payload(r_t)}
    # Map historical rows with dynamic keys (t-1, t-2, ...)
    for idx, row in enumerate(reversed(hist_only), 1):
        history_dict[f"t-{idx}"] = _row_payload(row)

    extras = {
        "mgrno": mgrno,
        "permno": permno,
        "holding_t": r_t.holding_t,
        "shares": r_t.shares,
        "label_tp1": (float(r_t.holding_t1) if r_t.holding_t1 is not None and not pd.isna(r_t.holding_t1) else None),
        "date": r_t.date,
        "label_profile_k": r_t.profile_k,
        "label_prev_profile_k": r_t.prev_profile_k,
        "objective_weights": r_t.objective_weights,
        "holding_log_delta": None,
        "label_delta_absolute": None,
        "history_rows": history_dict,
        "ticker": str(permno) if permno is not None else None,
        "company": None,
    }
    if extras["holding_t"] is not None and extras["label_tp1"] is not None:
        try:
            ht = float(extras["holding_t"])  # type: ignore[arg-type]
            tp1 = float(extras["label_tp1"])  # type: ignore[arg-type]
            extras["label_delta_absolute"] = tp1 - ht
            if ht >= 0 and tp1 >= 0:
                log_delta = math.log((tp1 + LOG_EPS) / (ht + LOG_EPS))
                extras["holding_log_delta"] = log_delta
        except Exception:
            extras["holding_log_delta"] = None
            extras["label_delta_absolute"] = None

    return prompt, extras
