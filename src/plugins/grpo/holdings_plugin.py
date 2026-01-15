import json
import math
import re
from typing import List

from swift.plugin.orm import ORM, orms

import numpy as np


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_CONTENT_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def _extract_answer_body(text: str) -> str | None:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_json_from_answer(text):
    """Try to extract a JSON object either from <answer>...</answer> or anywhere in text.

    This makes the reward robust when the model does not emit <answer> tags yet.
    Also tolerates the caller传入 dict 已经解析好的情况。
    """
    if isinstance(text, dict):
        # already a parsed dict: return as-is if it looks like the answer payload
        if any(k in text for k in ("holding_log_delta", "holding_delta")):
            return text
        # otherwise try to read inner content
        if "content" in text:
            text = text.get("content")
        else:
            return text
    def _try_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) Prefer content inside <answer> tags
    body = _extract_answer_body(text)
    candidates: list[str] = []
    if isinstance(body, str) and body:
        # strip surrounding code fences if present
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|```$", "", body.strip())
        candidates.append(cleaned)
        # substring between first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1])

    # 2) Fallback: search whole completion
    whole = text or ""
    whole_cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|```$", "", whole.strip())
    start2 = whole_cleaned.find("{")
    end2 = whole_cleaned.rfind("}")
    if start2 != -1 and end2 != -1 and end2 > start2:
        candidates.append(whole_cleaned[start2:end2 + 1])

    # Try load in order
    for s in candidates:
        obj = _try_load(s)
        if isinstance(obj, dict):
            return obj
    return None


class ContractHoldingsORM(ORM):
    """Format/number contract for holdings output while allowing reasoning.

    - Extracts the first occurrence of {"holding_log_delta": <num>} from the <answer> body (tolerates extra text).
    - Numeric value must be finite (no NaN/inf) with up to 6 decimals.
    Reward: +1.0 if a valid value is found, else -1.0.
    """

    _LOG_DELTA_SEARCH = re.compile(r'"holding_log_delta"\s*:\s*(-?\d+(?:\.\d{1,6})?)', re.IGNORECASE)

    def __call__(self, completions, holding_t=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        if not isinstance(holding_t, list):
            holding_t = [holding_t] * len(completions)
        for comp, ht in zip(completions, holding_t):
            try:
                text = comp or ""
                lower_text = text.lower()
                if "<answer>" not in lower_text or "</answer>" not in lower_text:
                    rewards.append(-1.0)
                    continue
                think_match = THINK_CONTENT_RE.search(text)
                if not think_match or not think_match.group(1).strip():
                    rewards.append(-10.0)
                    continue
                answer_pos = lower_text.find("<answer>")
                if answer_pos > -1:
                    prefix = text[:answer_pos]
                    cleaned_prefix = THINK_CONTENT_RE.sub("", prefix).strip()
                    if "assistant" in cleaned_prefix.lower() or "</answer>" in cleaned_prefix.lower():
                        rewards.append(-1.0)
                        continue

                # Accept values inside <answer> or anywhere in completion as fallback
                body = _extract_answer_body(comp) or comp or ""

                m = self._LOG_DELTA_SEARCH.search(body)
                if m:
                    val = float(m.group(1))
                    if not math.isfinite(val):
                        rewards.append(-1.0)
                        continue
                    rewards.append(1.0)
                    continue

                rewards.append(-1.0)
            except Exception:
                rewards.append(-1.0)
        return rewards


class ContractMSEZORM(ORM):
    """Composite reward matching the paper-style structure.

    R = lambda_r * I[contract] - lambda_q * MSE(q_hat, q) - lambda_z * MSE(z_hat, z)

    Expected:
      - q_hat in <answer> JSON as "holding_log_delta"
      - optional z_hat in <answer> JSON as "z" (or "z_hat")
      - q label from label_delta/holding_log_delta
      - optional z label from label_z/z (if provided)
    """

    def __call__(
        self,
        completions,
        label_delta=None,
        holding_log_delta=None,
        label_z=None,
        z=None,
        **kwargs,
    ) -> List[float]:
        lambda_r = float(kwargs.get("lambda_r", 1.0))
        lambda_q = float(kwargs.get("lambda_q", 1.0))
        lambda_z = float(kwargs.get("lambda_z", 1.0))
        penalize_missing = bool(kwargs.get("penalize_missing", False))
        missing_q_penalty = float(kwargs.get("missing_q_penalty", 1.0))
        missing_z_penalty = float(kwargs.get("missing_z_penalty", 1.0))

        if label_delta is None and holding_log_delta is not None:
            label_delta = holding_log_delta
        if label_delta is None and "holding_log_delta" in kwargs:
            label_delta = kwargs.get("holding_log_delta")
        if label_z is None:
            label_z = z if z is not None else kwargs.get("label_z")

        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)
        if not isinstance(label_z, list):
            label_z = [label_z] * len(completions)

        rewards: List[float] = []
        for comp, q_true, z_true in zip(completions, label_delta, label_z):
            obj = _extract_json_from_answer(comp)
            contract = 0.0
            q_hat = None
            z_hat = None
            if isinstance(obj, dict):
                if obj.get("holding_log_delta") is not None:
                    try:
                        q_hat = float(obj["holding_log_delta"])
                        if math.isfinite(q_hat):
                            contract = 1.0
                        else:
                            q_hat = None
                    except Exception:
                        q_hat = None
                if obj.get("z") is not None:
                    try:
                        z_hat = float(obj["z"])
                        if not math.isfinite(z_hat):
                            z_hat = None
                    except Exception:
                        z_hat = None
                if z_hat is None and obj.get("z_hat") is not None:
                    try:
                        z_hat = float(obj["z_hat"])
                        if not math.isfinite(z_hat):
                            z_hat = None
                    except Exception:
                        z_hat = None

            mse_q = 0.0
            if q_hat is None or q_true is None:
                if penalize_missing:
                    mse_q = missing_q_penalty
            else:
                try:
                    qt = float(q_true)
                    if math.isfinite(qt):
                        mse_q = float((q_hat - qt) ** 2)
                    elif penalize_missing:
                        mse_q = missing_q_penalty
                except Exception:
                    if penalize_missing:
                        mse_q = missing_q_penalty

            mse_z = 0.0
            if z_hat is None or z_true is None:
                if penalize_missing:
                    mse_z = missing_z_penalty
            else:
                try:
                    zt = float(z_true)
                    if math.isfinite(zt):
                        mse_z = float((z_hat - zt) ** 2)
                    elif penalize_missing:
                        mse_z = missing_z_penalty
                except Exception:
                    if penalize_missing:
                        mse_z = missing_z_penalty

            reward = (lambda_r * contract) - (lambda_q * mse_q) - (lambda_z * mse_z)
            rewards.append(float(reward))

        return rewards


class HoldingsDeltaORM(ORM):
    """Composite reward = w_mag * R_mag + w_dir * R_dir.

    - Magnitude reward R_mag: normalized Huber on error e=pred-target with adaptive/robust scale.
    - Direction reward R_dir: sigmoid on scaled pred aligned with sign(target).
    Accepts holding_delta values only.
    """

    def __init__(self):
        self._ema_abs_err = None  # type: float | None
        self._ema_abs_r = None    # type: float | None

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(min(x, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _huber(e: float, c: float) -> float:
        ae = abs(e)
        if ae <= c:
            return 0.5 * e * e
        return c * (ae - 0.5 * c)

    @staticmethod
    def _robust_scale(vals: list[float], method: str, k: float, eps: float = 1e-8) -> float:
        v = [abs(x) for x in vals if x is not None]
        if not v:
            return eps
        v.sort()
        if method == 'iqr':
            q1 = v[int(0.25 * (len(v) - 1))]
            q3 = v[int(0.75 * (len(v) - 1))]
            iqr = max(q3 - q1, eps)
            return k * iqr
        # mad
        med = v[len(v) // 2]
        mad = [abs(x - med) for x in v]
        mad.sort()
        m = mad[len(mad) // 2]
        return k * max(m, eps)

    def __call__(self, completions, label_delta=None, label_tp1=None, holding_t=None, **kwargs) -> List[float]:
        k_mag = float(kwargs.get('k_mag', 1.5))
        k_dir = float(kwargs.get('k_dir', 1.0))
        ema_lambda = float(kwargs.get('ema_lambda', 0.9))
        alpha = float(kwargs.get('alpha', 5.0))
        margin = float(kwargs.get('margin', 0.0))
        w_mag = float(kwargs.get('w_mag', 0.6))
        w_dir = float(kwargs.get('w_dir', 0.4))
        robust_mode = str(kwargs.get('robust_mode', 'ema'))  # 'ema', 'mad', 'iqr'

        rewards: List[float] = []
        # allow holding_log_delta alias
        if label_delta is None and "holding_log_delta" in kwargs:
            label_delta = kwargs.get("holding_log_delta")
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)
        if not isinstance(label_tp1, list):
            label_tp1 = [label_tp1] * len(completions)
        if not isinstance(holding_t, list):
            holding_t = [holding_t] * len(completions)

        preds: list[float | None] = []
        targets: list[float | None] = []
        for comp, gt_delta, gt_tp1, ht in zip(completions, label_delta, label_tp1, holding_t):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict):
                    if obj.get('holding_log_delta') is not None:
                        pred = float(obj['holding_log_delta'])
                    elif obj.get('holding_delta') is not None:
                        pred = float(obj['holding_delta'])
            except Exception:
                pred = None

            tgt = None
            try:
                if gt_delta is not None:
                    tgt = float(gt_delta)
            except Exception:
                tgt = None

            preds.append(pred)
            targets.append(tgt)

        es: list[float] = []
        rs: list[float] = []
        for pred, tgt in zip(preds, targets):
            if pred is None or tgt is None:
                es.append(None)  # type: ignore
                rs.append(None)  # type: ignore
            else:
                es.append(pred - tgt)
                rs.append(tgt)

        eps = 1e-6
        if robust_mode == 'ema':
            def _mean_abs(vs):
                v = [abs(x) for x in vs if x is not None]
                return (sum(v) / len(v)) if v else None

            mean_abs_e = _mean_abs(es)
            mean_abs_r = _mean_abs(rs)
            if mean_abs_e is not None:
                self._ema_abs_err = (
                    float(mean_abs_e) if self._ema_abs_err is None else ema_lambda * float(self._ema_abs_err)
                    + (1 - ema_lambda) * float(mean_abs_e)
                )
            if mean_abs_r is not None:
                self._ema_abs_r = (
                    float(mean_abs_r) if self._ema_abs_r is None else ema_lambda * float(self._ema_abs_r)
                    + (1 - ema_lambda) * float(mean_abs_r)
                )
            c_mag = k_mag * float(self._ema_abs_err if self._ema_abs_err is not None else 1.0)
            c_dir = k_dir * float(self._ema_abs_r if self._ema_abs_r is not None else 1.0)
        else:
            method = 'mad' if robust_mode == 'mad' else 'iqr'
            c_mag = self._robust_scale([x for x in es if x is not None], method, k_mag, eps)
            c_dir = self._robust_scale([x for x in rs if x is not None], method, k_dir, eps)
        c_mag = max(c_mag, eps)
        c_dir = max(c_dir, eps)

        for pred, tgt, e, r_val in zip(preds, targets, es, rs):
            if pred is None or tgt is None or e is None or r_val is None:
                rewards.append(-1.0)
                continue
            sig = math.copysign(math.log1p(abs(e)), e)
            sim01 = 1.0 - min((sig * sig) / 4.0, 1.0)
            r_mag = 2.0 * sim01 - 1.0

            s = (pred / c_dir) * (1.0 if r_val >= 0 else -1.0)
            r_dir = self._sigmoid(alpha * (s - margin))
            rewards.append(float(w_mag * r_mag + w_dir * (2.0 * r_dir - 1.0)))

        return rewards


# register names for --reward_funcs
orms['external_holdings'] = HoldingsDeltaORM
orms['contract_holdings'] = ContractHoldingsORM
orms["contract_mse_z"] = ContractMSEZORM


class DirectionHoldingsORM(ORM):
    """Reward measuring directional alignment with optional soft weighting."""

    def __call__(self, completions, label_delta=None, **kwargs) -> List[float]:
        if label_delta is None and "holding_log_delta" in kwargs:
            label_delta = kwargs.get("holding_log_delta")
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)

        rewards: List[float] = []
        tol = float(kwargs.get("direction_eps", 2e-2))
        scale = float(kwargs.get("sign_scale", 5.0))
        use_weight = bool(kwargs.get("sign_weighted", True))
        weight_cap = float(kwargs.get("sign_weight_cap", 0.2))

        for comp, tgt in zip(completions, label_delta):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict) and obj.get("holding_log_delta") is not None:
                    pred = float(obj["holding_log_delta"])
            except Exception:
                pred = None

            try:
                tgt_val = float(tgt) if tgt is not None else None
            except Exception:
                tgt_val = None

            if pred is None or tgt_val is None or not math.isfinite(pred) or not math.isfinite(tgt_val):
                rewards.append(-1.0)
                continue

            tgt_abs = abs(tgt_val)
            if tgt_abs < tol:
                rewards.append(0.0)
                continue

            soft = math.tanh(scale * (pred * tgt_val))
            if use_weight:
                weight = min(tgt_abs / weight_cap, 1.0) if weight_cap > 0 else 1.0
                soft *= weight

            rewards.append(float(soft))

        return rewards


orms["direction_holdings"] = DirectionHoldingsORM


class MagnitudeHoldingsORM(ORM):
    """Reward for magnitude closeness; maps absolute error to [-1, 1]."""

    def __call__(self, completions, label_delta=None, **kwargs) -> List[float]:
        if label_delta is None and "holding_log_delta" in kwargs:
            label_delta = kwargs.get("holding_log_delta")
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)

        threshold = float(kwargs.get("threshold", 0.2))
        threshold = max(threshold, 1e-6)

        rewards: List[float] = []
        for comp, tgt in zip(completions, label_delta):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict) and obj.get("holding_log_delta") is not None:
                    pred = float(obj["holding_log_delta"])
            except Exception:
                pred = None

            try:
                tgt_val = float(tgt) if tgt is not None else None
            except Exception:
                tgt_val = None

            if pred is None or tgt_val is None or not math.isfinite(pred) or not math.isfinite(tgt_val):
                rewards.append(-1.0)
                continue

            diff = abs(pred - tgt_val)
            ratio = min(diff / threshold, 1.0)
            r_mag = 1.0 - ratio
            rewards.append(2.0 * r_mag - 1.0)

        return rewards


orms["magnitude_holdings"] = MagnitudeHoldingsORM


class HuberHoldingsORM(ORM):
    """Reward based on Huber loss mapped linearly into [-1, 1]."""

    def __init__(self):
        self._ema_scale: float | None = None
        self._recent_errors: list[float] = []

    def __call__(self, completions, label_delta=None, **kwargs) -> List[float]:
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)

        delta = float(kwargs.get("delta", 0.05))
        base_cap = float(kwargs.get("huber_cap", 0.12))
        base_cap = max(base_cap, 1e-8)
        adaptive_mode = str(kwargs.get("adaptive_cap", "ema")).lower()
        ema_lambda = float(kwargs.get("ema_lambda", 0.9))
        cap_scale = float(kwargs.get("cap_scale", 2.0))
        cap_floor = float(kwargs.get("cap_floor", base_cap))
        cap_percentile = float(kwargs.get("cap_percentile", 90.0))
        cap_window = int(kwargs.get("cap_window", 512))

        rewards: List[float] = []
        errs: list[float | None] = []
        for comp, tgt in zip(completions, label_delta):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict) and obj.get("holding_log_delta") is not None:
                    pred = float(obj["holding_log_delta"])
            except Exception:
                pred = None

            try:
                tgt_val = float(tgt) if tgt is not None else None
            except Exception:
                tgt_val = None

            if pred is None or tgt_val is None or not math.isfinite(pred) or not math.isfinite(tgt_val):
                errs.append(None)
                continue

            errs.append(pred - tgt_val)

        valid_errs = [e for e in errs if e is not None]
        huber_cap = base_cap
        if adaptive_mode == "ema" and valid_errs:
            mean_abs = float(sum(abs(e) for e in valid_errs) / len(valid_errs))
            if self._ema_scale is None:
                self._ema_scale = mean_abs
            else:
                self._ema_scale = float(ema_lambda * self._ema_scale + (1 - ema_lambda) * mean_abs)
            huber_cap = max(cap_floor, float(self._ema_scale) * cap_scale)
        elif adaptive_mode == "percentile" and valid_errs:
            self._recent_errors.extend(abs(e) for e in valid_errs)
            if len(self._recent_errors) > cap_window:
                self._recent_errors = self._recent_errors[-cap_window:]
            if self._recent_errors:
                huber_cap = max(cap_floor, float(np.percentile(self._recent_errors, cap_percentile)))

        huber_cap = max(huber_cap, 1e-8)

        for e in errs:
            if e is None:
                rewards.append(-1.0)
                continue
            ae = abs(e)
            if ae <= delta:
                huber = 0.5 * (e ** 2)
            else:
                huber = delta * (ae - 0.5 * delta)
            ratio = min(huber / huber_cap, 1.0)
            rewards.append(1.0 - 2.0 * ratio)

        return rewards


orms["huber_holdings"] = HuberHoldingsORM

class ProfileNumericDeviationORM(ORM):
    """
    Profile alignment reward (Gaussian, per-dimension, 0~1).

    For each dimension d in {risk, herd, profit}:
        z_d = (I_d - mu_d) / sigma_d
        r_d = exp(-0.5 * z_d^2)

    Final reward = mean(r_risk, r_herd, r_profit)
    """

    def __init__(self, stats_path="data/grpo_profile_stats.csv"):
        import pandas as pd
        self.stats = pd.read_csv(stats_path).set_index("type")

        # ===== 对齐你 CSV 的列名 =====
        self.mu_risk = "x_risk_mean"
        self.var_risk = "x_risk_var"

        self.mu_herd = "x_herd_mean"
        self.var_herd = "x_herd_var"

        self.mu_profit = "x_profit_mean"
        self.var_profit = "x_profit_var"

    def __call__(
        self,
        completions,
        vix_q_prev=None,
        ln_market_volume_q_prev=None,
        profile_semantics=None,
        history_rows=None,
        **kwargs
    ) -> List[float]:

        # -------- broadcast（保持你系统的风格）--------
        if not isinstance(vix_q_prev, list):
            vix_q_prev = [vix_q_prev] * len(completions)
        if not isinstance(ln_market_volume_q_prev, list):
            ln_market_volume_q_prev = [ln_market_volume_q_prev] * len(completions)
        if not isinstance(profile_semantics, list):
            profile_semantics = [profile_semantics] * len(completions)
        if not isinstance(history_rows, list):
            history_rows = [history_rows] * len(completions)

        rewards: List[float] = []
        eps = float(kwargs.get("eps", 1e-9))
        debug = bool(kwargs.get("debug_profile_dev", False))

        # profit 分母下限：避免 profit 缺失/为 0 时 I_p 爆炸 -> r_p 下溢为 0
        profit_floor = float(kwargs.get("profit_floor", 1e-3))

        def safe_std_from_var(v):
            try:
                vv = float(v)
                if not math.isfinite(vv) or vv <= 0:
                    return 1e-8
                return float(math.sqrt(vv))
            except Exception:
                return 1e-8

        def gaussian(x, mu, sd):
            sd = max(sd, 1e-8)
            z = (x - mu) / sd
            # 防止 exp 下溢太快（可选，但通常不需要）
            z = max(min(z, 20.0), -20.0)
            return float(math.exp(-0.5 * z * z))

        for i, comp in enumerate(completions):
            try:
                # -------- 模型输出 --------
                obj = _extract_json_from_answer(comp)
                if not isinstance(obj, dict) or obj.get("holding_log_delta") is None:
                    rewards.append(0.0)
                    continue

                delta = float(obj["holding_log_delta"])
                if not math.isfinite(delta):
                    rewards.append(0.0)
                    continue

                abs_delta = abs(delta)

                # -------- 市场状态 --------
                vix_raw = float(vix_q_prev[i])
                volm_raw = float(ln_market_volume_q_prev[i])
                vix = max(abs(vix_raw), eps)
                volm = max(abs(volm_raw), eps)

                # -------- type --------
                ps = profile_semantics[i] or {}
                inv_type = ps.get("investor_type")
                if inv_type not in self.stats.index:
                    rewards.append(0.0)
                    continue
                row = self.stats.loc[inv_type]

                # -------- stats: mu & std（由 var 开方得到）--------
                mu_r = float(row[self.mu_risk])
                sd_r = safe_std_from_var(row[self.var_risk])

                mu_h = float(row[self.mu_herd])
                sd_h = safe_std_from_var(row[self.var_herd])

                mu_p = float(row[self.mu_profit]) if math.isfinite(float(row[self.mu_profit])) else 0.0
                sd_p = safe_std_from_var(row[self.var_profit])

                # -------- profit_t --------
                hr = history_rows[i] or {}
                profit_t = float((hr.get("t") or {}).get("profit", 0.0))
                profit_denom = max(abs(profit_t), profit_floor, eps)

                # -------- intensities（与你之前一致）--------
                I_r = abs_delta / vix
                I_h = abs_delta / volm
                I_p = delta / profit_denom

                # -------- Gaussian rewards (0~1) --------
                r_r = gaussian(I_r, mu_r, sd_r)
                r_h = gaussian(I_h, mu_h, sd_h)
                r_p = gaussian(I_p, mu_p, sd_p)

                r = (r_r + r_h + r_p) / 3.0
                rewards.append(float(r))

                if debug and i == 0:
                    print(
                        "[ProfileGaussian]",
                        f"type={inv_type}",
                        f"I_r={I_r:.4e} mu_r={mu_r:.4e} sd_r={sd_r:.4e} r_r={r_r:.4f}",
                        f"I_h={I_h:.4e} mu_h={mu_h:.4e} sd_h={sd_h:.4e} r_h={r_h:.4f}",
                        f"I_p={I_p:.4e} mu_p={mu_p:.4e} sd_p={sd_p:.4e} r_p={r_p:.4f}",
                        f"reward={r:.4f}",
                        f"(vix={vix_raw}, volm={volm_raw}, profit_t={profit_t})",
                    )

            except Exception as e:
                if debug:
                    print("[ProfileGaussian][EXCEPTION]", repr(e))
                rewards.append(0.0)

        return rewards


orms["profile_numeric_deviation"] = ProfileNumericDeviationORM
