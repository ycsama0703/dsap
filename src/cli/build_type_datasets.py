from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Optional
import pandas as pd
from openai import OpenAI
import os
import re

from src.cli.build_history_prompts import build_for_file
from src.cli.prompts_to_sft import _build_auto_think

def _load_market_quarterly_safe():
    try:
        from src.cli.build_history_prompts import _load_market_quarterly
    except Exception:
        return None
    try:
        return _load_market_quarterly(Path("data/VIXCLS.csv"), Path("data/sp500_market_volume.csv"))
    except Exception as e:
        print(f"[type-pipeline] warn: failed to load market data for profile constraints: {e}")
        return None

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


def _role_from_type(inv_type: str | None) -> str:
    if not inv_type:
        return ""
    role = inv_type.replace("_", " ").strip()
    if role.endswith("s") and len(role) > 1:
        role = role[:-1]
    return role


def _build_system_prompt(inv_type: str | None, mgrno: str | int | None = None) -> str:
    role = _role_from_type(inv_type)
    role_segment = f"{role} " if role else ""
    mgr_segment = ""
    if mgrno:
        mgr_segment = f" from manager {mgrno}"
    return (
        "A conversation between User and Assistant. The User provides financial data, and the Assistant, "
        f"acting as a {role_segment}quantitative portfolio manager{mgr_segment}, reasons carefully and predicts portfolio adjustments. "
        "If you include reasoning, keep it concise (<=5 sentences)."
    )


def _format_row_for_prompt(row: dict | None) -> str:
    if not isinstance(row, dict):
        return (
            "# Firm-level characteristics: core financial and risk attributes of the company.\n"
            "me=NA, be=NA, profit=NA, Gat=NA, beta=NA\n\n"
            "# Portfolio context: fund-level size metrics providing overall exposure scale.\n"
            "aum=NA, outAUM=NA\n\n"
            "# Benchmark reference: the stock’s weight in the S&P500 indicating relative market importance.\n"
            "spx_weight=NA\n\n"
            "# Current position: the fund’s existing exposure to the stock.\n"
            "holding=NA, price=NA\n\n"
            "# Stock-level market aggregates (prev quarter).\n"
            "stock_vol_q_prev=NA, stock_ln_volume_q_prev=NA"
        )

    def _get(key: str) -> str:
        val = row.get(key)
        return "NA" if val is None else str(val)

    parts = [
        "# Firm-level characteristics: core financial and risk attributes of the company.",
        f"me={_get('me')}, be={_get('be')}, profit={_get('profit')}, Gat={_get('Gat')}, beta={_get('beta')}",
        "",
        "# Portfolio context: fund-level size metrics providing overall exposure scale.",
        f"aum={_get('aum')}, outAUM={_get('outAUM')}",
        "",
        "# Benchmark reference: the stock’s weight in the S&P500 indicating relative market importance.",
        f"spx_weight={_get('sp500_weight')}",
        "",
        "# Current position: the fund’s existing exposure to the stock.",
        f"holding={_get('holding')}, price={_get('price')}",
        "",
        "# Stock-level market aggregates (prev quarter).",
        f"stock_vol_q_prev={_get('stock_vol_q_prev')}, stock_ln_volume_q_prev={_get('stock_ln_volume_q_prev')}",
    ]
    return "\n".join(parts)


def _build_structured_prompt(rec: dict, *, curr_only: bool = False) -> str:
    history = rec.get("history_rows")
    ticker = rec.get("ticker") or rec.get("permno") or "{ticker}"
    company = rec.get("company") or "the company"
    ticker_str = str(ticker)
    company_str = str(company)

    if not isinstance(history, dict):
        return rec.get("prompt") or rec.get("query") or ""

    current_line = _format_row_for_prompt(history.get("t"))

    if curr_only:
        return (
            "Given the current fundamentals for "
            f"{ticker_str} ({company_str}), estimate the normalized log change in portfolio holding from time t to t+1.\n\n"
            "Current (t):\n"
            f"{current_line}"
        )

    hist_parts = []
    # gather available historical keys like t-1, t-2 ... sorted oldest -> newest
    hist_keys = [k for k in history.keys() if k.startswith("t-")]
    def _key_order(k: str) -> int:
        try:
            return -int(k.split("-")[1])
        except Exception:
            return 0
    for k in sorted(hist_keys, key=_key_order, reverse=True):
        row_line = _format_row_for_prompt(history.get(k))
        hist_parts.append(f"{{{k} data}}: {row_line}")
    hist_block = "\n".join(hist_parts)

    return (
        "Given recent historical fundamentals and the current data for "
        f"{ticker_str} ({company_str}), estimate the normalized log change in portfolio holding from time t to t+1.\n\n"
        "Past data:\n"
        f"{hist_block}\n\n"
        "Current (t):\n"
        f"{current_line}"
    )


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _parse_permno_filter(raw: str | None, path_str: str | None) -> set[int] | None:
    vals: set[int] = set()

    def _consume(text: str):
        for token in re.split(r"[\s,]+", text.strip()):
            if not token:
                continue
            try:
                vals.add(int(token))
            except Exception:
                continue

    if raw:
        _consume(raw)
    if path_str:
        try:
            p = Path(path_str)
            if p.exists():
                with p.open() as f:
                    for line in f:
                        _consume(line)
        except Exception:
            pass
    return vals or None


def _estimate_total_records(fp: Path, limit: Optional[int]) -> Optional[int]:
    if limit is not None:
        return limit
    try:
        with fp.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def _create_progress_bar(label: str, total: Optional[int]):
    if not label:
        return None
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm(total=total, desc=f"{label} (convert)")


THINK_PLACEHOLDER = (
    "<think>\n"
    "[Using the investor profile constraints (risk_aversion, herd_behavior, profit_driven),\n"
    "reason about how these preferences affect the interpretation of the current fundamentals\n"
    "and justify the direction and magnitude of the log change over holding_t.]\n"
    "</think>"
)
GRPO_ANSWER_PLACEHOLDER = "<value>"
_NEUTRAL_PROFILE_SUMMARY = "Neutral placeholder profile for format-only SFT."


def _build_neutral_profile_context() -> dict:
    return {
        "objective_weights": {
            "risk_aversion": 0.333333,
            "herd_behavior": 0.333333,
            "profit_driven": 0.333333,
        },
        "philosophy": {
            "style": "neutral",
            "activity": "neutral",
            "benchmark_orientation": "neutral",
            "horizon": "neutral",
        },
        "constraints": {
            "risk_tolerance": "neutral",
            "turnover_constraint": "neutral",
            "concentration_constraint": "neutral",
            "herd_tendency": "neutral",
        },
        "summary": _NEUTRAL_PROFILE_SUMMARY,
    }


def _convert_prompts_to_sft(
    inp: Path,
    outp: Path,
    *,
    system: str,
    inv_type: str | None = None,
    with_think: bool = True,
    contract_mode: str = "delta",
    decimals: int = 2,
    think_template: str = "",
    profile_mode: str = "real",
    limit: Optional[int] = None,
    label: str = "sft",
    progress_every: int = 100,
    curr_only_prompt: bool = False,
) -> int:
    from src.cli.prompts_to_sft import _resolve_absolute, _resolve_log_delta as _resolve_delta, _format_float


    if contract_mode not in {"absolute", "delta"}:
        raise ValueError(f"unknown contract_mode={contract_mode}")
    profile_mode = (profile_mode or "real").lower()
    if profile_mode not in {"real", "neutral", "none"}:
        raise ValueError(f"unknown profile_mode={profile_mode}")
    resolver = _resolve_delta if contract_mode == "delta" else _resolve_absolute
    answer_key = "holding_log_delta" if contract_mode == "delta" else "holding_tp1"


    _ensure_dir(outp)
    n = 0
    # preload semantics maps (profile-level and type-level)
    sem_map_profiles: dict[tuple[str, int], dict] = {}
    sem_map_type: dict[str, dict] = {}
    sem_path = Path("artifacts/features/profile_semantics_llm.json")
    sem_type_path = Path("artifacts/features/type_profile_semantics.json")
    try:
        if sem_path.exists():
            data = json.loads(sem_path.read_text(encoding="utf-8"))
            for item in data:
                pid = item.get("profile_id")
                if not pid or "_p" not in pid:
                    continue
                tname, kstr = pid.rsplit("_p", 1)
                try:
                    k = int(kstr)
                except Exception:
                    continue
                sem_map_profiles[(tname, k)] = item
    except Exception as e:
        print(f"[warn] failed to load profile semantics: {e}")
    try:
        if sem_type_path.exists():
            data = json.loads(sem_type_path.read_text(encoding="utf-8"))
            for item in data:
                t = item.get("investor_type")
                if not t:
                    continue
                sem_map_type[t] = item
    except Exception as e:
        print(f"[warn] failed to load type profile semantics: {e}")
    eff_progress = progress_every
    total_records = _estimate_total_records(inp, limit)
    pbar = _create_progress_bar(label, total_records)
    if pbar is not None:
        eff_progress = 0
    if limit and (not eff_progress or eff_progress > limit):
        eff_progress = max(1, limit // 5) if limit >= 5 else 1
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = _build_structured_prompt(rec, curr_only=curr_only_prompt)
            if not prompt:
                prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            label_val = resolver(rec)
            profile_val = rec.get("label_profile_k") or rec.get("profile_k")
            if profile_val is None:
                profile_val = 0  # default to single profile per type
            if label_val is None or profile_val is None:
                continue
            value = _format_float(float(label_val), decimals)
            resp_json = json.dumps({answer_key: value}, ensure_ascii=False)
            mgrno_val = rec.get("mgrno")
            system_content = _build_system_prompt(inv_type, mgrno_val) if inv_type else system
            # inject profile_context
            ctx = None
            if profile_mode == "neutral":
                ctx = _build_neutral_profile_context()
            elif profile_mode == "real":
                semantics_ctx = {}
                if inv_type is not None:
                    semantics_ctx = sem_map_profiles.get((inv_type, int(profile_val)), {})
                    if not semantics_ctx:
                        semantics_ctx = sem_map_type.get(inv_type, {})
                if semantics_ctx:
                    ow_sem = semantics_ctx.get("objective_weights") or {}
                    # normalize keys to risk_aversion/herd_behavior/profit_driven
                    ow_norm = {
                        "risk_aversion": ow_sem.get("risk_aversion") if "risk_aversion" in ow_sem else ow_sem.get("risk"),
                        "herd_behavior": ow_sem.get("herd_behavior") if "herd_behavior" in ow_sem else ow_sem.get("tc"),
                        "profit_driven": ow_sem.get("profit_driven") if "profit_driven" in ow_sem else ow_sem.get("alpha"),
                    }
                    ow_norm = {k: v for k, v in ow_norm.items() if v is not None}
                    ctx = {
                        "profile_id": semantics_ctx.get("profile_id") or f"{inv_type}_p{int(profile_val)}",
                        "objective_weights": ow_norm if ow_norm else semantics_ctx.get("objective_weights"),
                    }
                    # include philosophy/constraints if present
                    if "philosophy" in semantics_ctx:
                        ctx["philosophy"] = semantics_ctx.get("philosophy")
                    if "constraints" in semantics_ctx:
                        ctx["constraints"] = semantics_ctx.get("constraints")
                    if "summary" in semantics_ctx:
                        ctx["summary"] = semantics_ctx.get("summary")
            if ctx:
                prompt = "<profile_context>\n" + json.dumps(ctx, ensure_ascii=False) + "\n</profile_context>\n\n" + prompt
            msgs = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            assistant_parts = []

            if with_think:
                think_text = rec.get("think")

                if not think_text:
                    if think_template:
                        think_text = think_template
                    else:
                        # profile-conditioned prompt (no label leakage)
                        if profile_mode != "real":
                            neutral_ctx = _build_neutral_profile_context()
                            phi = neutral_ctx.get("philosophy", {})
                            cons = neutral_ctx.get("constraints", {})
                            obj = neutral_ctx.get("objective_weights", {})
                            profile_desc = (
                                f"- Style: {phi.get('style','NA')}\n"
                                f"- Activity: {phi.get('activity','NA')}\n"
                                f"- Risk tolerance: {cons.get('risk_tolerance','NA')}\n"
                                f"- Turnover preference: {cons.get('turnover_constraint','NA')}\n"
                                f"- Objective emphasis: risk_aversion={obj.get('risk_aversion','NA')}, "
                                f"herd_behavior={obj.get('herd_behavior','NA')}, "
                                f"profit_driven={obj.get('profit_driven','NA')}\n"
                            )
                        else:
                            profile_id = f"{inv_type}_p{int(profile_val)}" if inv_type is not None else None
                            semantics_path = Path("artifacts/features/profile_semantics_llm.json")
                            sem_map = {}
                            sem_type_path = Path("artifacts/features/type_profile_semantics.json")
                            sem_type_map = {}
                            try:
                                if semantics_path.exists():
                                    data = json.loads(semantics_path.read_text(encoding="utf-8"))
                                    for item in data:
                                        pid = item.get("profile_id")
                                        if pid:
                                            sem_map[pid] = item
                            except Exception as e:
                                print(f"[warn] failed to load profile semantics: {e}")
                            try:
                                if sem_type_path.exists():
                                    data = json.loads(sem_type_path.read_text(encoding="utf-8"))
                                    for item in data:
                                        t = item.get("investor_type")
                                        if t:
                                            sem_type_map[t] = item
                            except Exception as e:
                                print(f"[warn] failed to load type profile semantics: {e}")
                            sem = sem_map.get(profile_id, {})
                            if not sem and inv_type:
                                sem = sem_type_map.get(inv_type, {})
                            phi = sem.get("philosophy", {}) if isinstance(sem, dict) else {}
                            cons = sem.get("constraints", {}) if isinstance(sem, dict) else {}
                            obj = sem.get("objective_weights", {}) if isinstance(sem, dict) else {}
                            # Normalize objective weights for description (prefer new schema; fallback to legacy keys)
                            risk_pref = obj.get("risk_aversion")
                            if risk_pref is None:
                                risk_pref = obj.get("risk")
                            herd_pref = obj.get("herd_behavior")
                            if herd_pref is None:
                                herd_pref = obj.get("tc")
                            prof_pref = obj.get("profit_driven")
                            if prof_pref is None:
                                prof_pref = obj.get("alpha")
                            profile_desc = (
                                f"- Style: {phi.get('style','NA')}\n"
                                f"- Activity: {phi.get('activity','NA')}\n"
                                f"- Risk tolerance: {cons.get('risk_tolerance','NA')}\n"
                                f"- Turnover preference: {cons.get('turnover_constraint','NA')}\n"
                                f"- Objective emphasis: risk_aversion={risk_pref if risk_pref is not None else 'NA'}, "
                                f"herd_behavior={herd_pref if herd_pref is not None else 'NA'}, "
                                f"profit_driven={prof_pref if prof_pref is not None else 'NA'}\n"
                            )
                        try:
                            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
                            model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
                            if not api_key:
                                raise RuntimeError("Missing DEEPSEEK_API_KEY/OPENAI_API_KEY for DeepSeek generation.")
                            client = OpenAI(api_key=api_key, base_url=api_base)
                            clean_prompt = re.split(r"OUTPUT FORMAT", prompt, maxsplit=1)[0].strip()
                            ds_prompt = f"""You are an experienced portfolio manager.

You follow this investor profile:
{profile_desc}

Given the market and portfolio information below, briefly explain what considerations would guide a holding adjustment decision.
Do NOT guess or mention any true labels. Focus on reasoning only (<=4 sentences).

---
{clean_prompt}
"""
                            resp_ds = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": "You are a financial reasoning assistant."},
                                    {"role": "user", "content": ds_prompt},
                                ],
                                temperature=0.7,
                                max_tokens=200,
                            )
                            think_text = resp_ds.choices[0].message.content.strip()
                            if not think_text.startswith("<think>"):
                                think_text = f"<think>{think_text}</think>"
                        except Exception as e:
                            print(f"[WARN] DeepSeek generation failed: {e}")
                            think_text = think_template or THINK_PLACEHOLDER

                if think_text:
                    think_lower = think_text.lower()
                    if not think_lower.startswith("<think>"):
                        think_text = f"<think>{think_text}</think>"
                    assistant_parts.append(think_text)

            # ---------------- 拼接最终输出 ----------------
            assistant_parts.append(f"<answer>\n{resp_json}\n</answer>")
            msgs.append({
                "role": "assistant",
                "content": "\n".join(assistant_parts),
                "loss": True,
            })
            out = {"messages": msgs}
            # trim history_rows when curr_only to avoid carrying t-1.. data
            if curr_only_prompt and "history_rows" in rec and isinstance(rec["history_rows"], dict):
                rec = dict(rec)
                rec["history_rows"] = {"t": rec["history_rows"].get("t")}
            meta_keys = [
                "permno", "mgrno", "date", "holding_t", "shares",
                "label_tp1", "label_log_delta", "label_delta_absolute", "history_rows",
            ]
            for k in meta_keys:
                if k in rec:
                    out[k] = rec[k]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if pbar is not None:
                pbar.update(1)
            if eff_progress and n % eff_progress == 0:
                limit_str = f"/{limit}" if limit else ""
                print(f"[type-pipeline] {label}: converted {n}{limit_str}")
            if limit and n >= limit:
                break
    if pbar is not None:
        pbar.close()
    print(f"[type-pipeline] {label}: done (total={n})")
    return n


def _convert_prompts_to_grpo(
    inp: Path,
    outp: Path,
    *,
    system: str,
    inv_type: str | None = None,
    no_think_example: bool = False,
    limit: Optional[int] = None,
    label: str = "grpo",
    progress_every: int = 100,
    curr_only_prompt: bool = False,
) -> int:
    # 预加载 profile 语义表（用于 prompt 注入）
    sem_path = Path("artifacts/features/profile_semantics_llm.json")
    sem_map: dict[tuple[str, int], dict] = {}
    # also load type-level semantics (one profile per type) if present
    sem_type_path = Path("artifacts/features/type_profile_semantics.json")
    sem_type_map: dict[str, dict] = {}
    if sem_path.exists():
        try:
            data = json.loads(sem_path.read_text(encoding="utf-8"))
            for item in data:
                pid = item.get("profile_id")
                if not pid or "_p" not in pid:
                    continue
                tname, kstr = pid.rsplit("_p", 1)
                try:
                    k = int(kstr)
                except Exception:
                    continue
                sem_map[(tname, k)] = item
        except Exception as e:
            print(f"[warn] failed to load profile semantics: {e}")
    if sem_type_path.exists():
        try:
            data = json.loads(sem_type_path.read_text(encoding="utf-8"))
            for item in data:
                t = item.get("investor_type")
                if not t:
                    continue
                sem_type_map[t] = item
        except Exception as e:
            print(f"[warn] failed to load type profile semantics: {e}")

    _ensure_dir(outp)
    n = 0
    eff_progress = progress_every
    total_records = _estimate_total_records(inp, limit)
    pbar = _create_progress_bar(label, total_records)
    if pbar is not None:
        eff_progress = 0
    if limit and (not eff_progress or eff_progress > limit):
        eff_progress = max(1, limit // 5) if limit >= 5 else 1
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = _build_structured_prompt(rec, curr_only=curr_only_prompt)
            if not prompt:
                prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            mgrno_val = rec.get("mgrno")
            system_content = _build_system_prompt(inv_type, mgrno_val) if inv_type else system

            # 取 profile_k 与语义，构造 profile_context 注入到 user prompt 顶部
            prof_k = rec.get("label_profile_k") or rec.get("profile_k")
            if prof_k is None:
                prof_k = 0
            prof_prev = rec.get("label_prev_profile_k")
            obj_w = rec.get("objective_weights")
            semantics = {}
            # Try profile-level semantics first
            if prof_k is not None and inv_type is not None:
                semantics = sem_map.get((inv_type, int(prof_k)), {})
                # 优先使用语义文件中的 objective_weights，确保与 profile 语义一致
                obj = semantics.get("objective_weights") or {}
                if obj:
                    obj_w = {
                        "alpha_w": obj.get("alpha"),
                        "risk_w": obj.get("risk"),
                        "tc_w": obj.get("tc"),
                    }
            # If no profile_k or missing semantics, fall back to type-level semantics (profile_k=0)
            if (not semantics) and inv_type is not None:
                sem_fallback = sem_type_map.get(inv_type)
                if sem_fallback:
                    semantics = sem_fallback
                    obj = semantics.get("objective_weights") or {}
                    if obj:
                        obj_w = {
                            "alpha_w": obj.get("risk_aversion"),
                            "risk_w": obj.get("herd_behavior"),
                            "tc_w": obj.get("profit_driven"),
                        }
                        prof_k = 0
            profile_context = ""
            if prof_k is not None:
                # normalize objective weights keys to risk_aversion/herd_behavior/profit_driven for the prompt
                ow_sem = obj_w or {}
                ow_norm = {
                    "risk_aversion": ow_sem.get("risk_aversion") if "risk_aversion" in ow_sem else ow_sem.get("alpha_w") or ow_sem.get("alpha"),
                    "herd_behavior": ow_sem.get("herd_behavior") if "herd_behavior" in ow_sem else ow_sem.get("risk_w") or ow_sem.get("risk"),
                    "profit_driven": ow_sem.get("profit_driven") if "profit_driven" in ow_sem else ow_sem.get("tc_w") or ow_sem.get("tc"),
                }
                ow_norm = {k: v for k, v in ow_norm.items() if v is not None}
                ctx = {}
                if ow_norm:
                    ctx["objective_weights"] = ow_norm
                # 附加语义（philosophy/constraints）用于条件化
                if semantics:
                    ctx["philosophy"] = semantics.get("philosophy", {})
                    ctx["constraints"] = semantics.get("constraints", {})
                    if "summary" in semantics:
                        ctx["summary"] = semantics.get("summary")
                profile_context = "<profile_context>\n" + json.dumps(ctx, ensure_ascii=False) + "\n</profile_context>\n\n"
                prompt = profile_context + prompt

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            if not no_think_example:
                messages.append({
                    "role": "assistant",
                    # Use a profile-aware placeholder to guide reasoning without leaking labels.
                    "content": f"{THINK_PLACEHOLDER}\n<answer>\n{{\"holding_log_delta\": \"{GRPO_ANSWER_PLACEHOLDER}\"}}\n</answer>",
                    "loss": False,
                })
            if curr_only_prompt and "history_rows" in rec and isinstance(rec["history_rows"], dict):
                rec = dict(rec)
                rec["history_rows"] = {"t": rec["history_rows"].get("t")}
            out = {
                "messages": messages,
            }
            # 计算/填充标签
            ht = rec.get("holding_t")
            tp1 = rec.get("label_tp1") or rec.get("label")
            hld = rec.get("holding_log_delta")
            if hld is None and ht is not None and tp1 is not None:
                try:
                    hld = math.log((float(tp1) + 1e-6) / (float(ht) + 1e-6))
                except Exception:
                    hld = None

            out.update({
                "holding_log_delta": hld,
                "label_delta": hld,
                "label_tp1": tp1,
                "holding_t": ht,
                "shares": rec.get("shares"),
                "mgrno": rec.get("mgrno"),
                "permno": rec.get("permno"),
                "date": rec.get("date"),
                "history_rows": rec.get("history_rows"),
                # 市场聚合
                "vix_q_prev": rec.get("vix_q_prev"),
                "ln_market_volume_q_prev": rec.get("ln_market_volume_q_prev"),
                # 个股聚合
                "stock_vol_q_prev": rec.get("stock_vol_q_prev"),
                "stock_ln_volume_q_prev": rec.get("stock_ln_volume_q_prev"),
                # reward/EMA-facing objective weights (model不可见)
                "profile_semantics": semantics,
            })
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if pbar is not None:
                pbar.update(1)
            if eff_progress and n % eff_progress == 0:
                limit_str = f"/{limit}" if limit else ""
                print(f"[type-pipeline] {label}: converted {n}{limit_str}")
            if limit and n >= limit:
                break
    if pbar is not None:
        pbar.close()
    print(f"[type-pipeline] {label}: done (total={n})")
    return n


def _count_zeros_in_prompts(fp: Path) -> tuple[int, int]:
    total = 0
    zeros = 0
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ht = rec.get("holding_t")
                total += 1
                if ht is not None and float(ht) == 0.0:
                    zeros += 1
            except Exception:
                continue
    return total, zeros


def main():
    ap = argparse.ArgumentParser(description="Build SFT/GRPO/Test datasets for a single firm type in one command")
    ap.add_argument("--type", required=True, help="Firm type stem, e.g., 'banks'")
    ap.add_argument("--in-dir", type=str, default="data/processed/panel_quarter.parquet")
    ap.add_argument("--per-type-limit", type=int, default=1000)
    ap.add_argument("--sft-limit", type=int, default=None)
    ap.add_argument("--grpo-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)

    # date splits
    ap.add_argument("--sft-end", type=str, default="2016-12-31")
    ap.add_argument("--grpo-start", type=str, default="2017-01-01")
    ap.add_argument("--grpo-end", type=str, default="2021-12-31")
    ap.add_argument("--test-start", type=str, default="2022-01-01")
    ap.add_argument("--prompt-curr-only", action="store_true",
                    help="If set, prompts only include the current (t) fundamentals block.")
    ap.add_argument("--history-len", type=int, choices=[2, 4], default=4,
                    help="Number of consecutive quarters per window (default: 4; set 2 for t-1,t windows).")
    ap.add_argument("--include-permnos", type=str, default="",
                    help="Comma/space separated list of permnos to include. Empty = all.")
    ap.add_argument("--include-permnos-file", type=str, default="",
                    help="Optional file with permnos (one per line) to include.")
    ap.add_argument("--test-permnos", type=str, default="",
                    help="Comma/space separated list of permnos for TEST split. Empty = use --include-permnos.")
    ap.add_argument("--test-permnos-file", type=str, default="",
                    help="Optional file with permnos (one per line) for TEST split. Empty = use include-permnos; default fallback: data/sp500_top10_panel_2015_2024.csv if present.")
    ap.add_argument("--test-take-all", action="store_true",
                    help="If set, take all available test windows for the specified permnos (no sampling/limit).")
    ap.add_argument("--single-ticker", type=str, default=None,
                    help="Restrict to a single ticker (symbol). Implies --include-permnos for that ticker and disables sampling.")
    # filter control
    ap.add_argument("--exclude-zero-holding-t", dest="exclude_zero", action="store_true")
    ap.add_argument("--include-zero-holding-t", dest="exclude_zero", action="store_false")
    ap.set_defaults(exclude_zero=True)
    ap.add_argument("--profile-dir", type=str, default=None,
                    help="Directory with *_iq_profile.(csv|parquet) per type (optional)")
    ap.add_argument("--profile-weights-dir", type=str, default=None,
                    help="Directory with *_profile_objective_weights.(csv|parquet) (optional)")
    ap.add_argument("--random-sample", action="store_true", help="Use random sampling instead of stratified (pipeline).")
    # converter knobs
    ap.add_argument("--sft-with-think", dest="sft_with_think", action="store_true",
                    help="Force enabling <think> message in SFT data (default: enabled)")
    ap.add_argument("--sft-no-think", dest="sft_with_think", action="store_false",
                    help="Disable the <think> message in SFT data")
    ap.add_argument("--sft-contract-mode", choices=["absolute", "delta"], default="delta",
                    help="Target for SFT outputs (default: delta)")
    ap.add_argument("--sft-decimals", type=int, default=2,
                    help="Round SFT labels to this many decimals (default: 2)")
    ap.add_argument("--sft-profile-mode", choices=["real", "neutral", "none"], default="neutral",
                    help="Profile context mode for SFT prompts (default: neutral)")
    ap.add_argument("--sft-think-template", type=str,
                    default="",
                    help="Template injected when --sft-with-think is active (leave blank to auto-generate reasoning)")
    ap.add_argument("--grpo-no-think-example", action="store_true")
    ap.add_argument("--emit-base-min-test", dest="emit_base_min", action="store_true",
                    help="Also emit a minimal-format test set for base model parsing (no think, simple system prompt).")
    ap.add_argument("--no-emit-base-min-test", dest="emit_base_min", action="store_false")
    ap.set_defaults(sft_with_think=True)
    ap.set_defaults(emit_base_min=True)
    args = ap.parse_args()

    t = args.type
    in_dir = Path(args.in_dir)
    in_file = in_dir / f"{t}.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"input parquet not found: {in_file}")

    # load optional mapping
    mapping = None
    try:
        if load_mapping is not None:
            mp_path = Path("data/ticker_mapping.csv")
            if mp_path.exists():
                mapping = load_mapping(mp_path)
                print(f"[type-pipeline] loaded mapping: {len(mapping)}")
    except Exception as e:
        print(f"[type-pipeline] mapping load failed: {e}")

    permno_filter = _parse_permno_filter(args.include_permnos, args.include_permnos_file)
    test_permno_filter = _parse_permno_filter(args.test_permnos, args.test_permnos_file)
    if not test_permno_filter:
        fallback = Path("data/sp500_top10_panel_2015_2024.csv")
        if fallback.exists():
            test_permno_filter = _parse_permno_filter(None, str(fallback))
            if test_permno_filter:
                print(f"[type-pipeline] TEST permnos defaulted to {fallback} ({len(test_permno_filter)})")
    single_permno = None
    single_ticker = args.single_ticker.strip().upper() if args.single_ticker else None
    if single_ticker:
        if mapping is None:
            raise ValueError("--single-ticker requires ticker_mapping.csv to be loaded")
        ticker_to_permnos: dict[str, list[int]] = {}
        for perm, (_name, tick) in mapping.items():
            if not tick:
                continue
            ticker_to_permnos.setdefault(tick.upper(), []).append(perm)
        matches = ticker_to_permnos.get(single_ticker)
        if not matches:
            raise ValueError(f"ticker '{single_ticker}' not found in mapping")
        matches = sorted(set(matches))
        single_permno = matches[0]
        if len(matches) > 1:
            print(f"[type-pipeline] ticker {single_ticker} maps to multiple permnos {matches}; using {single_permno}")
        permno_filter = {single_permno}
        print(f"[type-pipeline] single-ticker mode: {single_ticker} -> permno {single_permno} (sampling disabled)")

    if permno_filter:
        sample_preview = ", ".join(str(p) for p in sorted(list(permno_filter))[:5])
        more = "..." if len(permno_filter) > 5 else ""
        print(f"[type-pipeline] restricting prompts to {len(permno_filter)} permnos: {sample_preview}{more}")
    if test_permno_filter:
        sample_preview = ", ".join(str(p) for p in sorted(list(test_permno_filter))[:5])
        more = "..." if len(test_permno_filter) > 5 else ""
        print(f"[type-pipeline] TEST restricted to {len(test_permno_filter)} permnos: {sample_preview}{more}")

    take_all_mode = single_permno is not None
    take_all_mode_test = take_all_mode or bool(test_permno_filter) or args.test_take_all
    # When explicitly taking all test windows (or using permno filter), ignore test_limit to keep full coverage.
    test_limit_override = None if take_all_mode_test else args.test_limit

    # outputs
    ph_sft_src = Path("artifacts/prompts_hist_sft") / f"{t}.jsonl"              # 构建原始 prompts 的输出（不带 think）
    ph_sft_enriched = Path("artifacts/prompts_hist_sft_with_think") / f"{t}.jsonl"  # DeepSeek 富集后的输入（带 think）
    #ph_sft = Path("artifacts/prompts_hist_sft_with_think") / f"{t}.jsonl"
    ph_grpo = Path("artifacts/prompts_hist_grpo") / f"{t}.jsonl"
    ph_test = Path("artifacts/prompts_hist_test") / f"{t}.jsonl"
    sft_out = Path("artifacts/sft") / f"sft_train_{t}.jsonl"
    # Put test chat JSONL into a separate folder to avoid mixing with SFT train data
    sft_test_out = Path("artifacts/test") / f"test_{t}.jsonl"
    grpo_out = Path("artifacts/grpo") / f"grpo_{t}.jsonl"
    base_min_out = Path("artifacts/test") / f"test_{t}_base_min.jsonl"

    system_prompt = _build_system_prompt(t)

    # 1) Build prompts for three splits
    print(f"[type-pipeline] building SFT prompts for {t} (<= {args.sft_end})")
    market_df = _load_market_quarterly_safe()

    build_for_file(
        in_file=in_file,
        out_file=ph_sft_src,
        per_type_limit=args.sft_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=None,
        date_end=args.sft_end,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=permno_filter,
        take_all=take_all_mode,
        limit_override=args.sft_limit,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
    )

    print(f"[type-pipeline] building GRPO prompts for {t} ({args.grpo_start}..{args.grpo_end})")
    build_for_file(
        in_file=in_file,
        out_file=ph_grpo,
        per_type_limit=args.grpo_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=args.grpo_start,
        date_end=args.grpo_end,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=permno_filter,
        take_all=take_all_mode,
        limit_override=args.grpo_limit,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
    )

    print(f"[type-pipeline] building TEST prompts for {t} (>= {args.test_start})")
    build_for_file(
        in_file=in_file,
        out_file=ph_test,
        per_type_limit=args.test_limit or args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        history_len=args.history_len,
        date_start=args.test_start,
        date_end=None,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
        include_permnos=test_permno_filter or permno_filter,
        take_all=take_all_mode_test,
        limit_override=test_limit_override,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
        random_sample=args.random_sample,
        market_df=market_df,
    )

    # 2) Convert
    # ✅ 永远使用原始 SFT prompts，DeepSeek 由本脚本自动生成
    sft_input_for_convert = ph_sft_src
    print(f"[type-pipeline] SFT convert input = {sft_input_for_convert} (DeepSeek integrated generation)")


    _convert_prompts_to_sft(
        sft_input_for_convert,
        sft_out,
        system=system_prompt,
        inv_type=t,
        with_think=args.sft_with_think,
        contract_mode=args.sft_contract_mode,
        decimals=args.sft_decimals,
        think_template=args.sft_think_template,
        profile_mode=args.sft_profile_mode,
        label=f"sft_train_{t}",
        progress_every=100,
        curr_only_prompt=args.prompt_curr_only,
    )


    print(f"[type-pipeline] convert TEST -> chat ({sft_test_out})")
    _convert_prompts_to_sft(
        ph_test,
        sft_test_out,
        system=system_prompt,
        inv_type=t,
        with_think=True,
        contract_mode="absolute",
        decimals=args.sft_decimals,
        think_template=THINK_PLACEHOLDER,
        label=f"test_chat_{t}",
        progress_every=100,
        curr_only_prompt=args.prompt_curr_only,
    )

    if args.emit_base_min:
        print(f"[type-pipeline] convert TEST -> base-min chat ({base_min_out})")
        _convert_prompts_to_sft(
            ph_test,
            base_min_out,
            system='Output exactly one JSON: {"holding_tp1": <float with 2 decimals>}. No other text.',
            inv_type=None,
            with_think=False,
            contract_mode="absolute",
            decimals=args.sft_decimals,
            label=f"test_base_min_{t}",
            progress_every=100,
            curr_only_prompt=args.prompt_curr_only,
        )

    print(f"[type-pipeline] convert GRPO -> dataset ({grpo_out})")
    _convert_prompts_to_grpo(
        ph_grpo,
        grpo_out,
        system=system_prompt,
        inv_type=t,
        no_think_example=args.grpo_no_think_example,
        label=f"grpo_{t}",
        progress_every=100,
        curr_only_prompt=args.prompt_curr_only,
    )

    # 3) Report stats
    stats_files = [
        (ph_sft_src, "prompts_sft_raw"),
        (ph_sft_enriched, "prompts_sft_with_think"),
        (ph_grpo, "prompts_grpo"),
        (ph_test, "prompts_test"),
    ]

    for fp, name in stats_files:
        if fp.exists():
            total, zeros = _count_zeros_in_prompts(fp)
            print(f"[type-pipeline] {name}: total={total} t==0={zeros} ratio={(zeros/total if total else 0):.3f}")


    print("[type-pipeline] done.")


if __name__ == "__main__":
    main()
