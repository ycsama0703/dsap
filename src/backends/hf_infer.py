# src/backends/hf_infer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, json, math, inspect
from typing import List, Dict, Any, Tuple, Dict as TypingDict
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
try:
    from peft import PeftConfig  # type: ignore
except ImportError:  # pragma: no cover
    PeftConfig = None  # type: ignore
try:
    from peft import LoraConfig  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from peft.tuners.lora import LoraConfig  # type: ignore
    except ImportError:  # pragma: no cover
        LoraConfig = None  # type: ignore

_JSON_RE = re.compile(r'\{.*?\}', re.S)
_FLOAT_RE = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
_HOLDING_T_PATTERN = re.compile(r'holding\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', re.IGNORECASE)
LOG_EPS = 1e-6
_LORA_ALLOWED_KEYS: set[str] = set()
if LoraConfig is not None:  # pragma: no branch
    try:
        sig = inspect.signature(LoraConfig.__init__)  # type: ignore[attr-defined]
        _LORA_ALLOWED_KEYS = {k for k in sig.parameters.keys() if k != "self"}
    except (ValueError, TypeError):
        _LORA_ALLOWED_KEYS = set()


def load_samples(fp: str):
    rows = []
    if fp.endswith(".jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
    else:
        rows = json.load(open(fp, "r", encoding="utf-8"))
    return rows


def parse_quarter_from_user(user_text: str) -> str:
    m = re.search(r'Timeline:\s*\(t\)\s*([0-9]{4})-([0-9]{2})-', user_text)
    if not m: return "NA"
    y, mth = int(m.group(1)), int(m.group(2))
    q = (mth-1)//3 + 1
    return f"{y}Q{q}"


def _parse_label_obj(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def extract_y_true(assistant_content: str, holding_t: float | None = None):
    """Extract ground-truth holding_log_delta (with optional fallback to absolute targets)."""

    def _to_log_delta(obj: Dict[str, Any]) -> float | None:
        if "holding_log_delta" in obj:
            try:
                val = float(obj["holding_log_delta"])
                return val if math.isfinite(val) else None
            except Exception:
                return None
        if holding_t is None:
            return None
        if "holding_delta" in obj:
            try:
                delta = float(obj["holding_delta"])
                tp1 = float(holding_t) + delta
                log_delta = math.log((tp1 + LOG_EPS) / (float(holding_t) + LOG_EPS))
                return log_delta if math.isfinite(log_delta) else None
            except Exception:
                return None
        if "holding_tp1" in obj:
            try:
                tp1 = float(obj["holding_tp1"])
                log_delta = math.log((tp1 + LOG_EPS) / (float(holding_t) + LOG_EPS))
                return log_delta if math.isfinite(log_delta) else None
            except Exception:
                return None
        return None

    obj = _parse_label_obj(assistant_content)
    if obj is None:
        return None
    return _to_log_delta(obj)


def extract_pred(text: str, holding_t: float | None = None) -> float | None:
    """Extract predicted holding_log_delta from model completion (robust version).

    holding_t is accepted for backward compatibility; it is unused.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    # ---------- 优先：解析 JSON ----------
    m = _JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("holding_log_delta") is not None:
                val = float(obj["holding_log_delta"])
                if math.isfinite(val) and -10 < val < 10:
                    return val
        except Exception:
            pass

    # ---------- 其次：仅在 <answer> 段中提取 ----------
    answer_block = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_block:
        content = answer_block.group(1)
        m2 = _FLOAT_RE.search(content or "")
        if m2:
            try:
                val = float(m2.group(0))
                if math.isfinite(val) and -10 < val < 10:
                    return val
            except Exception:
                pass

    # ---------- fallback：返回 None ----------
    return None



def load_model_and_tokenizer(base_model: str, lora_path: str|None, torch_dtype="bfloat16"):
    dtype = torch.bfloat16 if torch_dtype=="bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    if lora_path and lora_path.lower()!="none":
        peft_kwargs: Dict[str, Any] = {}
        adapter_path = Path(lora_path)
        adapter_cfg_path = adapter_path / "adapter_config.json"
        raw_cfg: TypingDict[str, Any] | None = None
        if adapter_cfg_path.exists():
            try:
                text = adapter_cfg_path.read_text()
                try:
                    raw_cfg = json.loads(text)
                except json.JSONDecodeError:
                    try:
                        text = text.strip()
                        text = re.sub(r"[,\\s]*\\}$", "}", text)
                        text = text.replace("null", "null")
                        raw_cfg = json.loads(text)
                    except json.JSONDecodeError:
                        try:
                            raw_cfg = json.loads(text.replace("\\n", "").replace(",}", "}"))
                        except json.JSONDecodeError:
                            raw_cfg = None
            except Exception:
                raw_cfg = None
        if raw_cfg and LoraConfig is not None:
            cfg_dict: Dict[str, Any] = dict(raw_cfg)
            cfg_dict.pop("corda_config", None)
            cfg_dict.pop("loftq_config", None)
            cfg_dict.pop("eva_config", None)
            cfg_dict.pop("adapter_mapping", None)
            cfg_dict.pop("qalora_group_size", None)
            allowed = _LORA_ALLOWED_KEYS or set(cfg_dict.keys())
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed}
            if cfg_dict.get("peft_type", "").lower() not in ("lora", ""):
                cfg_dict.pop("peft_type", None)
            if "task_type" in cfg_dict and isinstance(cfg_dict["task_type"], str):
                cfg_dict["task_type"] = cfg_dict["task_type"].upper()
            peft_kwargs["config"] = LoraConfig(**cfg_dict)
        mdl = PeftModel.from_pretrained(mdl, lora_path, **peft_kwargs)
        try: mdl = mdl.merge_and_unload()
        except Exception: pass
    return tok, mdl


@torch.no_grad()
def infer_chat_batch(tokenizer, model, list_messages: List[List[Dict[str,str]]],
                     max_new_tokens=48, temperature=0.0, force_think: bool = False) -> List[str]:
    prompts = []
    for msgs in list_messages:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).rstrip()
        if force_think:
            if not prompt.endswith("<think>"):
                prompt = f"{prompt}\n<think>"
        prompts.append(prompt)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    out = model.generate(
        **inputs,
        do_sample=(temperature>0),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    decoded = []
    for seq, prompt_len in zip(out, prompt_lengths):
        prompt_len = int(prompt_len)
        generated_tokens = seq[prompt_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if force_think:
            text = text.lstrip()
            if not text.startswith("<think>"):
                text = f"<think>{text}"
        decoded.append(text)
    return decoded


def parse_holding_t(user_text: str) -> float | None:
    m = _HOLDING_T_PATTERN.search(user_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _resolve_y_true_from_row(row: Dict[str, Any], holding_t: float | None) -> float | None:
    """Resolve holding_log_delta from top-level fields when assistant is missing."""
    for key in ("holding_log_delta", "label_delta"):
        if row.get(key) is not None:
            try:
                val = float(row[key])
                return val if math.isfinite(val) else None
            except Exception:
                pass
    if holding_t is not None and row.get("label_tp1") is not None:
        try:
            tp1 = float(row["label_tp1"])
            log_delta = math.log((tp1 + LOG_EPS) / (float(holding_t) + LOG_EPS))
            return log_delta if math.isfinite(log_delta) else None
        except Exception:
            return None
    return None


def build_eval_inputs(test_path: str) -> Tuple[List[List[Dict[str,str]]], List[float], List[str], List[int], List[float | None], List[int | None], List[str | None]]:
    rows = load_samples(test_path)
    chat_inputs, y_true, quarter, ids = [], [], [], []
    holding_ts: List[float | None] = []
    permnos: List[int | None] = []
    dates: List[str | None] = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        sys = next((m for m in msgs if m["role"]=="system"), None)
        usr = next((m for m in msgs if m["role"]=="user"), None)
        assistants = [m for m in msgs if m.get("role") == "assistant"]
        if not (sys and usr):
            continue
        # pick the assistant message that has loss=True if present; otherwise pick the last assistant
        holding_val = parse_holding_t(usr["content"])
        yt = None
        if assistants:
            ast = next((m for m in assistants if m.get("loss") is True), assistants[-1])
            yt = extract_y_true(ast.get("content", ""), holding_val)
        if yt is None:
            yt = _resolve_y_true_from_row(r, holding_val)
        if yt is None:
            continue
        chat_inputs.append([{"role":"system","content":sys["content"]},
                            {"role":"user","content":usr["content"]}])
        y_true.append(yt)
        quarter.append(parse_quarter_from_user(usr["content"]))
        ids.append(i)
        holding_ts.append(holding_val)
        permnos.append(r.get("permno"))
        dates.append(r.get("date"))
    return chat_inputs, y_true, quarter, ids, holding_ts, permnos, dates
