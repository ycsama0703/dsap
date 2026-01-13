#!/usr/bin/env python
"""
End-to-end pipeline from per-type debug outputs to 3 CSVs:
1) per-stock-per-date predictions + metrics
2) per-stock averages
3) per-cap-size averages
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

LOG_EPS = 1e-6
MAPE_EPS = 1e-6

CAP_LARGE = {10107, 14593, 22752, 22111, 84788}
CAP_MID = {91416, 77606, 40125, 89217, 57817}
CAP_SMALL = {91277, 14273, 13947, 80539, 80054, 42585}


def _cap_label(permno: int | None) -> str:
    if permno is None:
        return "unknown"
    if permno in CAP_LARGE:
        return "large"
    if permno in CAP_MID:
        return "mid"
    if permno in CAP_SMALL:
        return "small"
    return "unknown"


def _load_test_shares(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "permno": rec.get("permno"),
                    "date": rec.get("date"),
                    "shares": rec.get("shares"),
                }
            )
    if not rows:
        df = pd.DataFrame({
            "permno": pd.Series([], dtype="Int64"),
            "date": pd.to_datetime(pd.Series([], dtype="datetime64[ns]")),
            "shares": pd.Series([], dtype="float64"),
        })
        return df
    df = pd.DataFrame(rows)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    return df.dropna(subset=["permno", "date"])


def _load_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["pred_tp1", "true_tp1", "holding_t", "parsed_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pred_tp1" not in df.columns:
        df["pred_tp1"] = np.nan
    if "true_tp1" not in df.columns:
        df["true_tp1"] = np.nan

    if df["pred_tp1"].isna().all() and "parsed_pred" in df.columns and "holding_t" in df.columns:
        def _calc(row):
            if pd.isna(row.get("parsed_pred")) or pd.isna(row.get("holding_t")):
                return np.nan
            try:
                return math.exp(float(row["parsed_pred"])) * (float(row["holding_t"]) + LOG_EPS) - LOG_EPS
            except Exception:
                return np.nan
        df["pred_tp1"] = df.apply(_calc, axis=1)

    return df


def _discover_types(pred_dir: Path, test_dir: Path, pred_suffix: str) -> list[str]:
    pattern = f"test_*_{pred_suffix}.csv"
    pred_types = {p.stem.replace("test_", "").replace(f"_{pred_suffix}", "") for p in pred_dir.glob(pattern)}
    test_types = {p.stem.replace("test_", "").replace("_all", "") for p in test_dir.glob("test_*_all.jsonl")}
    return sorted(pred_types & test_types)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build price metrics CSVs from per-type debug outputs.")
    ap.add_argument("--pred-dir", type=Path, required=True, help="Directory with test_<type>_grpo.csv files.")
    ap.add_argument("--test-dir", type=Path, required=True, help="Directory with test_<type>_all.jsonl files.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--types", type=str, default="", help="Comma-separated types to include.")
    ap.add_argument("--pred-suffix", type=str, default="grpo",
                    help="Prediction CSV suffix, e.g. 'grpo' or 'sft' (default: grpo).")
    ap.add_argument("--run-id", type=str, default="auto",
                    help="Subfolder name under out-dir. Use 'auto' for timestamp or empty to disable.")
    ap.add_argument("--direction-eps", type=float, default=1e-6,
                    help="Threshold for direction classification (default: 1e-6)")
    ap.add_argument("--no-cap", action="store_true",
                    help="Skip cap-size aggregation; write overall summary instead.")
    args = ap.parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"pred dir not found: {args.pred_dir}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"test dir not found: {args.test_dir}")

    if args.types.strip():
        types = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        types = _discover_types(args.pred_dir, args.test_dir, args.pred_suffix)
    if not types:
        raise RuntimeError("no matching types between pred and test directories")

    frames = []
    for t in types:
        pred_path = args.pred_dir / f"test_{t}_{args.pred_suffix}.csv"
        test_path = args.test_dir / f"test_{t}_all.jsonl"
        if not pred_path.exists() or not test_path.exists():
            print(f"[warn] missing for type={t}: {pred_path.exists()} / {test_path.exists()}")
            continue
        pred_df = _load_pred_csv(pred_path)
        share_df = _load_test_shares(test_path)
        merged = pred_df.merge(share_df, on=["permno", "date"], how="left")
        merged["type"] = t
        merged = merged.dropna(subset=["pred_tp1", "shares"])
        frames.append(merged)

    if not frames:
        raise RuntimeError("no rows after merge; check inputs")

    all_df = pd.concat(frames, ignore_index=True)
    group_cols = ["permno", "date"]

    agg = (
        all_df.groupby(group_cols, as_index=False)
        .agg(
            sum_pred_tp1=("pred_tp1", "sum"),
            sum_true_tp1=("true_tp1", "sum"),
            sum_holding_t=("holding_t", "sum"),
            sum_shares_t=("shares", "sum"),
            n_types=("type", "nunique"),
        )
    )

    def _safe_div(num, den):
        if den is None or den == 0 or pd.isna(den):
            return np.nan
        return num / den

    agg["pred_price"] = agg.apply(lambda r: _safe_div(r["sum_pred_tp1"], r["sum_shares_t"]), axis=1)
    agg["true_price"] = agg.apply(lambda r: _safe_div(r["sum_true_tp1"], r["sum_shares_t"]), axis=1)

    # per-row metrics
    agg["abs_error"] = (agg["pred_price"] - agg["true_price"]).abs()
    agg["mae"] = agg["abs_error"]
    denom = agg["true_price"].abs()
    agg["mape"] = np.where(denom > MAPE_EPS, agg["abs_error"] / denom, np.nan)
    # direction accuracy (3-way: negative/zero/positive)
    agg["pred_log_delta"] = np.log((agg["sum_pred_tp1"] + LOG_EPS) / (agg["sum_holding_t"] + LOG_EPS))
    agg["true_log_delta"] = np.log((agg["sum_true_tp1"] + LOG_EPS) / (agg["sum_holding_t"] + LOG_EPS))
    eps = float(args.direction_eps)
    def _dir_class(v: float | None) -> float | None:
        if v is None or pd.isna(v):
            return np.nan
        if v > eps:
            return 1.0
        if v < -eps:
            return -1.0
        return 0.0
    agg["pred_dir"] = agg["pred_log_delta"].apply(_dir_class)
    agg["true_dir"] = agg["true_log_delta"].apply(_dir_class)
    agg["dir_acc"] = np.where(
        agg["pred_dir"].notna() & agg["true_dir"].notna(),
        (agg["pred_dir"] == agg["true_dir"]).astype(float),
        np.nan,
    )
    agg = agg.sort_values(["permno", "date"])

    # per-stock averages
    stock = (
        agg.groupby("permno", as_index=False)
        .agg(
            n_obs=("permno", "size"),
            mean_mae=("mae", "mean"),
            mean_mape=("mape", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_pred_price=("pred_price", "mean"),
            mean_true_price=("true_price", "mean"),
            mean_dir_acc=("dir_acc", "mean"),
        )
    )
    stock["cap_size"] = stock["permno"].apply(lambda v: _cap_label(int(v)) if pd.notna(v) else "unknown")
    stock = stock.sort_values("permno")

    run_id = args.run_id.strip()
    if run_id.lower() == "auto":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_id if run_id else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_row = out_dir / "price_pred_by_permno_date_metrics.csv"
    out_stock = out_dir / "price_metrics_by_stock.csv"

    agg.to_csv(out_row, index=False)
    stock.to_csv(out_stock, index=False)
    print(f"[ok] wrote {len(agg):,} rows -> {out_row}")
    print(f"[ok] wrote {len(stock):,} rows -> {out_stock}")
    if args.no_cap:
        valid = agg[agg["true_price"].abs() > MAPE_EPS].copy()
        weights = valid["sum_holding_t"].abs()
        total_abs_error = float(valid["abs_error"].sum()) if not valid.empty else float("nan")
        total_abs_true = float(valid["true_price"].abs().sum()) if not valid.empty else float("nan")
        overall_mape = total_abs_error / total_abs_true if total_abs_true and not math.isnan(total_abs_true) else float("nan")
        total_w_abs_error = float((valid["abs_error"] * weights).sum()) if not valid.empty else float("nan")
        total_w_abs_true = float((valid["true_price"].abs() * weights).sum()) if not valid.empty else float("nan")
        overall_wape = total_w_abs_error / total_w_abs_true if total_w_abs_true and not math.isnan(total_w_abs_true) else float("nan")
        overall = pd.DataFrame([{
            "n_rows": int(len(valid)),
            "overall_mape": overall_mape,
            "overall_wape": overall_wape,
            "mean_mape": float(valid["mape"].mean()) if not valid.empty else float("nan"),
            "mean_mae": float(valid["mae"].mean()) if not valid.empty else float("nan"),
            "mean_dir_acc": float(valid["dir_acc"].mean()) if not valid.empty else float("nan"),
            "total_abs_error": total_abs_error,
            "total_abs_true_price": total_abs_true,
        }])
        out_overall = out_dir / "price_metrics_overall.csv"
        overall.to_csv(out_overall, index=False)
        print(f"[ok] wrote 1 rows -> {out_overall}")
    else:
        cap = (
            stock.groupby("cap_size", as_index=False)
            .agg(
                n_stocks=("permno", "size"),
                mean_mae=("mean_mae", "mean"),
                mean_mape=("mean_mape", "mean"),
                mean_abs_error=("mean_abs_error", "mean"),
                mean_pred_price=("mean_pred_price", "mean"),
                mean_true_price=("mean_true_price", "mean"),
                mean_dir_acc=("mean_dir_acc", "mean"),
                n_obs_total=("n_obs", "sum"),
            )
        )
        out_cap = out_dir / "price_metrics_by_cap.csv"
        cap.to_csv(out_cap, index=False)
        print(f"[ok] wrote {len(cap):,} rows -> {out_cap}")


if __name__ == "__main__":
    main()
