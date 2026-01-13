#!/usr/bin/env python
"""
Compute one time-invariant profile per investor type using quarterly holding deltas
and aggregated market data (VIX + market volume).

Profile definition (per type):

  risk_aversion =
      mean( |Δlog_holding| / market_vol_q_prev )

  herd_behavior =
      mean( |Δlog_holding| / market_volm_q_prev )

  profit_driven =
      mean( sign(Δlog_holding) * profit )

Then convert raw scores into relative preference weights via:
  (1) robust z-score across types
  (2) row-wise centering
  (3) softmax

Daily → quarterly aggregation:
  For a quarter starting at Q_t (e.g., 2016-04-01), use daily data from the
  previous quarter [Q_{t-1}_start, Q_t_start) and take the mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-type profiles (risk / herd / profit weights).")
    p.add_argument("--panel-dir", required=True, help="Directory with per-type parquet files.")
    p.add_argument("--stock-daily-path", default="data/stock_daily.parquet",
                   help="Parquet with columns permno,ticker,date,AdjClose/Close,Volume.")
    p.add_argument("--out-path", default="artifacts/features/type_profiles.csv")
    p.add_argument("--eps", type=float, default=1e-9)
    return p.parse_args()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _read_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    date_key = None
    for key in ["date", "observation_date", "fdate"]:
        if key in cols_lower:
            date_key = cols_lower[key]
            break
    if date_key is None:
        raise ValueError(f"{path} must contain a date column")
    df[date_key] = pd.to_datetime(df[date_key])
    df.columns = [c.lower() for c in df.columns]
    return df.rename(columns={date_key.lower(): "date"})


# ------------------------------------------------------------
# Stock-level daily → quarterly (prev quarter aligned)
# ------------------------------------------------------------
def load_stock_quarterly(daily_path: Path, eps: float) -> pd.DataFrame:
    """Aggregate per-permno daily price/volume to prev-quarter metrics."""
    df = pd.read_parquet(daily_path)
    df = df.rename(columns={"adjclose": "AdjClose", "close": "Close", "volume": "Volume"})
    # prefer AdjClose if available
    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
    if price_col not in df.columns or "Volume" not in df.columns:
        raise ValueError("stock daily data must contain price (AdjClose/Close) and Volume")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["permno", price_col, "Volume"])

    # compute daily log returns per permno
    df = df.sort_values(["permno", "date"])
    df["ret"] = df.groupby("permno")[price_col].transform(lambda s: (s / s.shift(1)).pipe(np.log))

    # assign quarter start
    df["quarter_start"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp(how="start")

    # aggregate per permno, quarter_start using prev-quarter window
    agg = (
        df.groupby(["permno", "quarter_start"], as_index=False)
        .agg(
            ret_std=("ret", lambda x: x.dropna().std() * np.sqrt(252)),
            ln_vol=("Volume", lambda x: np.log(x).mean() if len(x) > 0 else np.nan),
        )
    )
    # shift forward one quarter: (t-1) → t
    shift = pd.offsets.QuarterBegin(startingMonth=1)
    agg["date"] = agg["quarter_start"] + shift

    agg = agg.rename(columns={"ret_std": "stock_vol_q_prev", "ln_vol": "stock_ln_volume_q_prev"})
    # replace zeros/negatives with eps guard
    agg["stock_vol_q_prev"] = agg["stock_vol_q_prev"].fillna(eps)
    agg["stock_ln_volume_q_prev"] = agg["stock_ln_volume_q_prev"].fillna(eps)
    return agg[["permno", "date", "stock_vol_q_prev", "stock_ln_volume_q_prev"]]


# ------------------------------------------------------------
# Panel loading
# ------------------------------------------------------------
def load_panel_frames(panel_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(panel_dir.glob("*.parquet")):
        df = pq.read_table(path).to_pandas()
        if "type" not in df.columns:
            df["type"] = path.stem
        else:
            df["type"] = (
                df["type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
        frames.append(df)
    return frames


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    eps = args.eps

    # stock-level quarterly aggregates
    market = load_stock_quarterly(Path(args.stock_daily_path), eps)

    # panel
    panel = pd.concat(load_panel_frames(Path(args.panel_dir)), ignore_index=True)
    if "date" not in panel.columns:
        raise ValueError("panel must contain date column")
    panel["date"] = pd.to_datetime(panel["date"])

    # Δlog holding
    panel["delta_log"] = np.log(panel["holding_t1"]) - np.log(panel["holding_t"])
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_log"])

    # merge stock-level market features
    panel = panel.merge(market, on=["permno", "date"], how="left")
    panel = panel.dropna(subset=["stock_vol_q_prev", "stock_ln_volume_q_prev"])

    # ratios for risk / herd (stock-level denominators)
    abs_delta = panel["delta_log"].abs()
    panel["risk_ratio"] = abs_delta / np.maximum(panel["stock_vol_q_prev"].abs(), eps)
    panel["herd_ratio"] = abs_delta / np.maximum(panel["stock_ln_volume_q_prev"].abs(), eps)

    # ⭐ PROFIT-DRIVEN (stock-level): delta_log scaled by profit
    if "profit" not in panel.columns:
        raise ValueError("panel must contain profit column for profit_driven (scheme stock)")
    panel["profit_align"] = panel["delta_log"] / np.maximum(panel["profit"].abs(), eps)

    # aggregate by type
    agg = (
        panel.groupby("type")
        .agg(
            risk_raw=("risk_ratio", "mean"),
            herd_raw=("herd_ratio", "mean"),
            prof_raw=("profit_align", "mean"),
            n_obs=("delta_log", "size"),
        )
        .reset_index()
    )

    # robust z-score across types
    def robust_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        if mad < eps:
            mad = x.std(ddof=0) if x.std(ddof=0) > eps else 1.0
        return (x - med) / mad

    z_df = pd.DataFrame({
        "risk_z": robust_z(agg["risk_raw"]),
        "herd_z": robust_z(agg["herd_raw"]),
        "prof_z": robust_z(agg["prof_raw"]),
    })

    # ⭐ row-wise centering (critical for preference interpretation)
    z_vals = z_df.to_numpy(dtype=float)
    z_vals = z_vals - z_vals.mean(axis=1, keepdims=True)

    # softmax → preference weights
    exp_vals = np.exp(z_vals)
    weights = exp_vals / np.maximum(exp_vals.sum(axis=1, keepdims=True), eps)

    weights_df = pd.DataFrame(
        weights,
        columns=["risk_aversion", "herd_behavior", "profit_driven"],
    )

    out_df = pd.concat([agg, z_df, weights_df], axis=1)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)

    print(f"[ok] wrote {len(out_df)} profiles -> {out_path}")


if __name__ == "__main__":
    main()
