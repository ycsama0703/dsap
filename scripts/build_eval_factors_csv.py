#!/usr/bin/env python
"""
Build a per-stock-per-date CSV for the 5x5 eval set.

Joins:
  - aggregated predictions (sum_pred_tp1, sum_true_tp1, prices)
  - stock-level factors at time t (me, be, profit, Gat, beta, vol/volume)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["permno", "date"])


def _load_panel_factors(panel_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "permno",
        "date",
        "ticker",
        "company",
        "me",
        "be",
        "profit",
        "Gat",
        "beta",
        "prc",
        "sp500_weight",
        "stock_vol_q_prev",
        "stock_ln_volume_q_prev",
    ]
    key_pairs = set(zip(keys["permno"].astype(int), keys["date"].dt.date))
    frames = []
    for p in sorted(panel_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=cols)
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df.empty:
            continue
        mask = df.apply(
            lambda r: (int(r["permno"]), r["date"].date()) in key_pairs,
            axis=1,
        )
        sub = df[mask].copy()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=cols)

    all_rows = pd.concat(frames, ignore_index=True)
    # collapse across types by taking first non-null per permno/date
    agg = (
        all_rows.sort_values(["permno", "date"])
        .groupby(["permno", "date"], as_index=False)
        .agg({c: "first" for c in cols if c not in ["permno", "date"]})
    )
    return agg


def _load_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["permno", "ticker_map", "company_map"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    permno_col = cols.get("permno")
    ticker_col = cols.get("ticker")
    company_col = cols.get("comnam")
    if not permno_col or not ticker_col or not company_col:
        return pd.DataFrame(columns=["permno", "ticker_map", "company_map"])
    out = df[[permno_col, ticker_col, company_col]].copy()
    out.columns = ["permno", "ticker_map", "company_map"]
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out.dropna(subset=["permno"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/sp500_eval/metrics/final_groups_5x5/price_pred_by_permno_date_metrics.csv"),
        help="Aggregated per-stock-per-date metrics CSV.",
    )
    ap.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/processed/type_agg_stock_yf"),
        help="Type-aggregated panel directory.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/sp500_eval/metrics/final_groups_5x5/price_pred_with_factors.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("data/ticker_mapping.csv"),
        help="Optional permno->ticker/company mapping CSV.",
    )
    args = ap.parse_args()

    metrics = _load_metrics(args.metrics_path)
    if metrics.empty:
        raise SystemExit(f"no rows in metrics: {args.metrics_path}")

    factors = _load_panel_factors(args.panel_dir, metrics[["permno", "date"]])
    merged = metrics.merge(factors, on=["permno", "date"], how="left")
    mapping = _load_mapping(args.mapping_csv)
    if not mapping.empty:
        merged = merged.merge(mapping, on="permno", how="left")
        if "ticker" not in merged.columns:
            merged["ticker"] = pd.NA
        if "company" not in merged.columns:
            merged["company"] = pd.NA
        merged["ticker"] = merged["ticker"].fillna(merged["ticker_map"])
        merged["company"] = merged["company"].fillna(merged["company_map"])
        merged = merged.drop(columns=["ticker_map", "company_map"])

    merged = merged.rename(
        columns={
            "sum_true_tp1": "true_total_holding",
            "sum_pred_tp1": "pred_total_holding",
        }
    )

    out_path = args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(merged):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
