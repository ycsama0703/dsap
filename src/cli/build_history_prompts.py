from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
import numpy as np

from src.prompts.sampler import _quarter_id, build_continuous_windows, stratified_sample_windows
from src.prompts.builder import PromptRow, build_history_prompt

try:
    # Reuse mapping utilities if available
    from src.cli.map_ticker_names import load_mapping, replace_ticker_line  # type: ignore
except Exception:
    load_mapping = None  # type: ignore
    replace_ticker_line = None  # type: ignore


def _row_from_dfrow(r: pd.Series) -> PromptRow:
    return PromptRow(
        mgrno=r.get("mgrno"),
        permno=r.get("permno"),
        investor_type=(r.get("type") or "Unknown"),
        profile_k=r.get("profile_k") if "profile_k" in r.index else None,
        prev_profile_k=r.get("prev_profile_k") if "prev_profile_k" in r.index else None,
        objective_weights=(
            {
                "alpha_w": r.get("alpha_w"),
                "risk_w": r.get("risk_w"),
                "tc_w": r.get("tc_w"),
                "te_w": r.get("te_w"),
            }
            if {"alpha_w", "risk_w", "tc_w", "te_w"}.issubset(set(r.index))
            else None
        ),
        holding_t=r.get("holding_t"),
        holding_t1=r.get("holding_t1"),
        stock_vol_q_prev=r.get("stock_vol_q_prev"),
        stock_ln_volume_q_prev=r.get("stock_ln_volume_q_prev"),
        shares=r.get("shares"),
        me=r.get("me") or r.get("factor1"),
        be=r.get("be") or r.get("factor2"),
        profit=r.get("profit") or r.get("factor3"),
        Gat=r.get("Gat") or r.get("factor4"),
        beta=r.get("beta") or r.get("factor5"),
        aum=r.get("aum"),
        outaum=r.get("outaum"),
        prc=r.get("prc"),
        sp500_weight=r.get("sp500_weight"),
        date=str(r.get("date")) if "date" in r.index else None,
    )


def _normalize_prompt_text(s: str) -> str:
    """Fix any legacy encoding artifacts to match prior prompt style exactly.
    - Replace broken '≥' and arrows in instruction examples.
    """
    try:
        s = s.replace("holding_delta �?-holding_t", "holding_delta ≥ -holding_t")
        s = s.replace("me�?(", "me↓ (")
        s = s.replace("profit�?(", "profit↑ (")
        s = s.replace(" �?moderate", " → moderate")
    except Exception:
        pass
    return s


REQUIRED_COLS = [
    "type", "mgrno", "permno", "date", "holding_t", "holding_t1",
    "me", "be", "profit", "Gat", "beta", "aum", "outaum", "prc",
    "shares", "sp500_weight",
    "stock_vol_q_prev", "stock_ln_volume_q_prev",
]


def _read_min_columns(fp: Path) -> pd.DataFrame:
    # Try to read a subset of columns to save memory; fallback to full read if not supported
    try:
        # prefer pyarrow if available for speed; fall back silently
        try:
            return pd.read_parquet(fp, columns=[c for c in REQUIRED_COLS if True], engine="pyarrow")
        except Exception:
            return pd.read_parquet(fp, columns=[c for c in REQUIRED_COLS if True])
    except Exception:
        return pd.read_parquet(fp)


def _quarter_str_from_date(s: pd.Series) -> pd.Series:
    """Convert datetime series to 'YYYYQX' string."""
    try:
        # drop timezone if present
        s = pd.to_datetime(s).dt.tz_localize(None)
    except Exception:
        s = pd.to_datetime(s, errors="coerce")
    return pd.PeriodIndex(s, freq="Q").astype(str)


def _normalize_type(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(" ", "_")


def _load_market_quarterly(vix_path: Path, vol_path: Path) -> pd.DataFrame:
    """Lightweight copy of load_market_quarterly (daily -> prev-quarter mean)."""
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

    vix = _read_with_date(vix_path)
    vol = _read_with_date(vol_path)
    if "vixcls" in vix.columns and "vix" not in vix.columns:
        vix = vix.rename(columns={"vixcls": "vix"})
    if "vix" not in vix.columns:
        raise ValueError("VIX file must contain vix or vixcls")
    if "ln_market_volume" not in vol.columns:
        raise ValueError("Volume file must contain ln_market_volume")

    def to_qstart(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["quarter_start"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp(how="start")
        return df

    vix = to_qstart(vix)
    vol = to_qstart(vol)
    vix_q = vix.groupby("quarter_start", as_index=False)["vix"].mean()
    vol_q = vol.groupby("quarter_start", as_index=False)["ln_market_volume"].mean()

    shift = pd.offsets.QuarterBegin(startingMonth=1)
    vix_q["date"] = vix_q["quarter_start"] + shift
    vol_q["date"] = vol_q["quarter_start"] + shift

    market = vix_q.merge(vol_q, on="date", how="outer")
    return market[["date", "vix", "ln_market_volume"]]


def _merge_profile_info(
    df: pd.DataFrame,
    *,
    type_stem: str,
    profile_dir: Path | None,
    weights_dir: Path | None,
) -> pd.DataFrame:
    """Attach profile_k / prev_profile_k (and optional objective weights) to df."""
    if profile_dir is None:
        return df
    prof_path = None
    candidates = [
        profile_dir / f"{type_stem}_iq_profile.parquet",
        profile_dir / f"{type_stem}_iq_profile.csv",
        profile_dir / f"{type_stem}.parquet",
        profile_dir / f"{type_stem}.csv",
    ]
    for c in candidates:
        if c.exists():
            prof_path = c
            break
    if prof_path is None:
        return df

    try:
        prof = pd.read_parquet(prof_path) if prof_path.suffix.lower() != ".csv" else pd.read_csv(prof_path)
    except Exception:
        return df

    # Normalize columns
    inv_col = "mgrno" if "mgrno" in prof.columns else ("investor_id" if "investor_id" in prof.columns else None)
    if inv_col is None or "quarter" not in prof.columns:
        return df
    typ_col = "type" if "type" in prof.columns else ("investor_type" if "investor_type" in prof.columns else None)
    if typ_col is None:
        prof["type"] = type_stem
        typ_col = "type"

    prof = prof.copy()
    prof["__type_norm"] = _normalize_type(prof[typ_col])
    prof["__quarter_str"] = _quarter_str_from_date(prof["quarter"])
    prof = prof.rename(columns={inv_col: "mgrno"})
    prof = prof[["mgrno", "__quarter_str", "__type_norm", "profile_k"]].drop_duplicates()

    df = df.copy()
    df["__type_norm"] = _normalize_type(df["type"])
    df["__quarter_str"] = _quarter_str_from_date(df["date"])
    df = df.merge(
        prof,
        how="left",
        on=["mgrno", "__quarter_str", "__type_norm"],
    )

    # prev_profile_k within investor (type, mgrno) sorted by date
    df = df.sort_values(["type", "mgrno", "date"])
    df["prev_profile_k"] = df.groupby(["type", "mgrno"], sort=False)["profile_k"].shift()

    # Optional objective weights
    if weights_dir is not None:
        w_path = None
        w_cands = [
            weights_dir / f"{type_stem}_profile_objective_weights.parquet",
            weights_dir / f"{type_stem}_profile_objective_weights.csv",
        ]
        for c in w_cands:
            if c.exists():
                w_path = c
                break
        if w_path is not None:
            try:
                wdf = pd.read_parquet(w_path) if w_path.suffix.lower() != ".csv" else pd.read_csv(w_path)
                w_typ = "investor_type" if "investor_type" in wdf.columns else "type"
                wdf["__type_norm"] = _normalize_type(wdf[w_typ])
                wdf = wdf.rename(columns={"profile_k": "profile_k", "alpha_w": "alpha_w", "risk_w": "risk_w", "tc_w": "tc_w", "te_w": "te_w"})
                wdf = wdf[["__type_norm", "profile_k", "alpha_w", "risk_w", "tc_w", "te_w"]]
                df = df.merge(wdf, how="left", on=["__type_norm", "profile_k"])
            except Exception:
                pass

    return df


def build_for_file(
    in_file: Path,
    out_file: Path,
    per_type_limit: int,
    time_bins: int,
    cap_per_pair: int,
    seed: int,
    *,
    history_len: int = 4,
    date_start: str | None = None,
    date_end: str | None = None,
    head: int | None = None,
    progress_every: int = 1000,
    use_tqdm: bool = False,
    mapping: dict | None = None,
    exclude_zero_holding_t: bool = False,
    include_permnos: set[int] | None = None,
    take_all: bool = False,
    limit_override: int | None = None,
    profile_dir: Path | None = None,
    profile_weights_dir: Path | None = None,
    random_sample: bool = False,
    market_df: pd.DataFrame | None = None,
    require_holding_t1: bool = False,
):
    t0 = time.perf_counter()
    print(f"[history-prompts] reading: {in_file}", flush=True)
    df = _read_min_columns(in_file)
    # Optional date filtering to shrink data for speed
    if date_start or date_end:
        try:
            if "date" in df.columns:
                ds = pd.to_datetime(date_start) if date_start else None
                de = pd.to_datetime(date_end) if date_end else None
                if ds is not None:
                    df = df[df["date"] >= ds]
                if de is not None:
                    df = df[df["date"] <= de]
        except Exception:
            pass
    if "type" not in df.columns:
        df = df.copy(); df["type"] = in_file.stem
    if include_permnos:
        try:
            df = df[df["permno"].isin(include_permnos)]
        except Exception:
            pass
    if head is not None and head > 0:
        # Keep head after sorting for deterministic subset
        df = df.sort_values(["type", "mgrno", "permno", "date"]).head(head)
    # Attach profile labels / prev_profile if available
    try:
        df = _merge_profile_info(
            df,
            type_stem=in_file.stem,
            profile_dir=profile_dir,
            weights_dir=profile_weights_dir,
        )
    except Exception as e:
        print(f"[history-prompts] warning: failed to merge profiles: {e}")

    # create windows and sample
    print(f"[history-prompts] building windows...", flush=True)
    windows = build_continuous_windows(
        df,
        use_tqdm=use_tqdm,
        progress_every=progress_every,
        label=in_file.stem,
        window_len=history_len,
    )
    if require_holding_t1:
        kept = []
        for w in windows:
            try:
                v = df.loc[w.idx_t]["holding_t1"]
                if pd.isna(v):
                    continue
                kept.append(w)
            except Exception:
                continue
        windows = kept
    print(
        f"[history-prompts] {in_file.stem}: rows={len(df):,} "
        f"groups={df.groupby(['type','mgrno','permno'], sort=False, observed=False).ngroups} "
        f"windows={len(windows):,}",
        flush=True,
    )
    def _nonzero_window(ws):
        if not exclude_zero_holding_t:
            return ws
        kept = []
        for w in ws:
            try:
                v = df.loc[w.idx_t]["holding_t"]
                if pd.isna(v):
                    continue
                if float(v) == 0.0:
                    continue
                kept.append(w)
            except Exception:
                continue
        return kept

    if take_all:
        picked = _nonzero_window(windows)
        total_before = len(picked)
        if limit_override is not None:
            lim = max(0, int(limit_override))
            picked = picked[:lim]
        print(
            f"[history-prompts] {in_file.stem}: selected {len(picked):,} / {total_before:,} windows (no sampling)",
            flush=True,
        )
    elif random_sample:
        rng = np.random.default_rng(seed)
        pool = _nonzero_window(windows)
        rng.shuffle(pool)
        total_before = len(pool)
        lim = per_type_limit if limit_override is None else max(0, int(limit_override))
        picked = pool[:lim]
        print(f"[history-prompts] {in_file.stem}: random sampled={len(picked):,} / {total_before:,} (limit={lim})", flush=True)
    else:
        picked = stratified_sample_windows(
            df,
            windows,
            per_type_limit=per_type_limit,
            time_bins=time_bins,
            cap_per_pair=cap_per_pair,
            seed=seed,
            exclude_zero_holding_t=exclude_zero_holding_t,
        )
        if limit_override is not None and limit_override >= 0:
            lim = max(0, int(limit_override))
            if lim < len(picked):
                picked = picked[:lim]
        print(f"[history-prompts] {in_file.stem}: sampled={len(picked):,} (limit={per_type_limit})", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total = len(picked)
    pbar = None
    _use_tqdm = False
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total, desc=f"write {in_file.stem}")
            _use_tqdm = True
        except Exception:
            _use_tqdm = False
    with out_file.open("w", encoding="utf-8") as f:
        for i, w in enumerate(picked, 1):
            idxs = [w.idx_tm3, w.idx_tm2, w.idx_tm1, w.idx_t]
            rows = [df.loc[idx] for idx in idxs[-history_len:]]
            hist_rows = [_row_from_dfrow(r) for r in rows]
            prompt, extras = build_history_prompt(hist_rows, hide_date=True, target="delta", strict_contract=True)
            prompt = _normalize_prompt_text(prompt)
            # Automatically map Ticker line to include company name if mapping provided
            if mapping is not None and replace_ticker_line is not None:
                permno_val = extras.get("permno")
                try:
                    permno_int = int(permno_val) if permno_val is not None else None
                except Exception:
                    permno_int = None
                name = ticker = None
                if permno_int is not None and permno_int in mapping:
                    # mapping value shape (name, ticker)
                    val = mapping[permno_int]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        name, ticker = val[0], val[1]
                prompt = replace_ticker_line(prompt, permno_int, name, ticker, mode="append")
                if ticker:
                    extras["ticker"] = ticker
                if name:
                    extras["company"] = name
            # attach profile info if available
            try:
                cur_row = df.loc[w.idx_t]
                if "profile_k" in cur_row:
                    pk = cur_row.get("profile_k")
                    if pd.notna(pk):
                        extras["profile_k"] = int(pk)
                if "prev_profile_k" in cur_row:
                    pp = cur_row.get("prev_profile_k")
                    if pd.notna(pp):
                        extras["prev_profile_k"] = int(pp)
                if "stock_vol_q_prev" in cur_row:
                    sv = cur_row.get("stock_vol_q_prev")
                    if pd.notna(sv):
                        extras["stock_vol_q_prev"] = float(sv)
                if "stock_ln_volume_q_prev" in cur_row:
                    slv = cur_row.get("stock_ln_volume_q_prev")
                    if pd.notna(slv):
                        extras["stock_ln_volume_q_prev"] = float(slv)
                if {"alpha_w", "risk_w", "tc_w", "te_w"}.issubset(df.columns):
                    aw = cur_row.get("alpha_w")
                    rw = cur_row.get("risk_w")
                    tw = cur_row.get("tc_w")
                    te = cur_row.get("te_w")
                    if not all(pd.isna([aw, rw, tw, te])):
                        extras["objective_weights"] = {
                            "alpha_w": None if pd.isna(aw) else float(aw),
                            "risk_w": None if pd.isna(rw) else float(rw),
                            "tc_w": None if pd.isna(tw) else float(tw),
                            "te_w": None if pd.isna(te) else float(te),
                        }
            except Exception:
                pass
            # attach market aggregates for the target quarter (prev-quarter mean)
            try:
                if market_df is not None and extras.get("date") is not None:
                    dt = pd.to_datetime(extras["date"]).tz_localize(None)
                    mrow = market_df.loc[market_df["date"] == dt]
                    if not mrow.empty:
                        extras["vix_q_prev"] = float(mrow.iloc[0]["vix"])
                        extras["ln_market_volume_q_prev"] = float(mrow.iloc[0]["ln_market_volume"])
            except Exception:
                pass

            rec = {"prompt": prompt, **extras}

            # Ensure numpy scalars are converted to Python types for JSON serialization
            def _json_default(o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.bool_,)):
                    return bool(o)
                return str(o)

            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
            if _use_tqdm and pbar is not None:
                pbar.update(1)
            elif progress_every and (i % progress_every == 0 or i == total):
                print(f"[history-prompts] {in_file.stem}: wrote {i}/{total}", flush=True)
    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass
    t1 = time.perf_counter()
    print(f"[history-prompts] {in_file.stem}: done in {t1 - t0:.1f}s -> {out_file}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description="Build history-style prompts (t-3..t) with per-type sampling")
    ap.add_argument("--in-dir", type=str, default="data/processed/panel_quarter.parquet",
                    help="Input directory with per-investor-type parquet files")
    ap.add_argument("--out-dir", type=str, default="artifacts/prompts_hist",
                    help="Output directory for per-type JSONL files")
    ap.add_argument("--include-types", type=str, default="",
                    help="Comma-separated file stems to include (e.g., 'banks,mutual_funds'). Empty = all.")
    ap.add_argument("--exclude-types", type=str, default="",
                    help="Comma-separated file stems to exclude.")
    ap.add_argument("--per-type-limit", type=int, default=1000)
    ap.add_argument("--time-bins", type=int, default=10)
    ap.add_argument("--cap-per-pair", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    # Speed/visibility knobs
    ap.add_argument("--max-files", type=int, default=None, help="Process at most N parquet files")
    ap.add_argument("--history-len", type=int, default=4, choices=[1, 2, 4], help="Number of consecutive quarters per window (default: 4)")
    ap.add_argument("--date-start", type=str, default=None, help="Filter rows with date >= this (e.g., 2015-01-01)")
    ap.add_argument("--date-end", type=str, default=None, help="Filter rows with date <= this (e.g., 2020-12-31)")
    ap.add_argument("--head", type=int, default=None, help="Use only first N rows after sorting (debug speed)")
    ap.add_argument("--progress-every", type=int, default=2000, help="Print progress every N writes if no tqdm")
    ap.add_argument("--use-tqdm", action="store_true", help="Use tqdm progress bars if installed")
    ap.add_argument("--exclude-zero-holding-t", action="store_true", help="Exclude windows where t-time holding_t == 0")
    ap.add_argument("--require-holding-t1", action="store_true", help="Require holding_t1 label at time t")
    ap.add_argument("--profile-dir", type=str, default=None, help="Directory with *_iq_profile.(csv|parquet) per type (optional)")
    ap.add_argument("--profile-weights-dir", type=str, default=None, help="Directory with *_profile_objective_weights.(csv|parquet) (optional)")
    ap.add_argument("--random-sample", action="store_true", help="Use simple random sampling (shuffled) instead of stratified sampling.")
    ap.add_argument("--vix-path", type=str, default="data/VIXCLS.csv", help="Daily VIX csv (for market aggregation, optional)")
    ap.add_argument("--volume-path", type=str, default="data/sp500_market_volume.csv", help="Daily SP500 volume csv (for market aggregation, optional)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    inc = {s.strip() for s in args.include_types.split(',') if s.strip()}
    exc = {s.strip() for s in args.exclude_types.split(',') if s.strip()}
    total = 0
    market_df = None
    try:
        market_df = _load_market_quarterly(Path(args.vix_path), Path(args.volume_path))
    except Exception as e:
        print(f"[history-prompts] warn: failed to load market data: {e}")
    # Load ticker mapping automatically if available
    mp = None
    try:
        if load_mapping is not None:
            default_map = Path("data/ticker_mapping.csv")
            if default_map.exists():
                mp = load_mapping(default_map)
                print(f"[history-prompts] loaded mapping: {len(mp)} PERMNO -> (name,ticker)")
            else:
                print("[history-prompts] mapping file not found: data/ticker_mapping.csv (skipping name mapping)")
        else:
            print("[history-prompts] mapping utilities unavailable (skipping name mapping)")
    except Exception as e:
        print(f"[history-prompts] failed to load mapping: {e}")

    for i, fp in enumerate(sorted(in_dir.glob("*.parquet")), 1):
        stem = fp.stem
        if inc and stem not in inc:
            continue
        if exc and stem in exc:
            continue
        outp = out_dir / f"{fp.stem}.jsonl"
        n = build_for_file(
            fp,
            outp,
            args.per_type_limit,
            args.time_bins,
            args.cap_per_pair,
            args.seed,
            history_len=args.history_len,
            date_start=args.date_start,
            date_end=args.date_end,
            head=args.head,
            progress_every=args.progress_every,
            use_tqdm=args.use_tqdm,
            mapping=mp,
            exclude_zero_holding_t=args.exclude_zero_holding_t,
            profile_dir=Path(args.profile_dir) if args.profile_dir else None,
            profile_weights_dir=Path(args.profile_weights_dir) if args.profile_weights_dir else None,
            random_sample=args.random_sample,
            market_df=market_df,
            require_holding_t1=args.require_holding_t1,
        )
        print(f"[history-prompts] {fp.stem}: wrote {n} -> {outp}")
        total += n
        if args.max_files is not None and i >= args.max_files:
            print(f"[history-prompts] reached max-files={args.max_files}, stopping.")
            break
    print(f"[history-prompts] total: {total}")


if __name__ == "__main__":
    main()
