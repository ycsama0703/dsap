from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import time
import numpy as np
import pandas as pd


def _quarter_id(dt_series: pd.Series) -> pd.Series:
    """Convert datetimes (tz-aware or naive) to integer quarter id year*4+q.
    Silences tz dropping warnings by removing tz first if present.
    """
    s = dt_series
    try:
        # If tz-aware, drop timezone for period conversion
        s = s.dt.tz_localize(None)
    except Exception:
        # already naive
        pass
    q = s.dt.to_period("Q")
    return (q.dt.year * 4 + q.dt.quarter).astype(int)


@dataclass
class WindowIndex:
    group_key: Tuple
    idx_tm3: int
    idx_tm2: int
    idx_tm1: int
    idx_t: int
    qid_t: int
    aum_t: float | None
    delta_sign: int | None
    window_len: int = 4


def build_continuous_windows(
    df: pd.DataFrame,
    *,
    use_tqdm: bool = False,
    progress_every: int = 2000,
    label: str | None = None,
    window_len: int = 4,
) -> List[WindowIndex]:
    """Create continuous quarterly windows for each (type, mgrno, permno).
    Expects columns: ['type','mgrno','permno','date','holding_t','holding_t1','aum'] (others optional)
    Returns list of WindowIndex with absolute row indices.

    window_len controls how many consecutive quarters are required (>=1).
    """
    assert window_len >= 1, "need at least 1 quarter to build a window"
    assert {"type", "mgrno", "permno", "date", "holding_t"}.issubset(df.columns)
    out: List[WindowIndex] = []
    t0 = time.perf_counter()
    df = df.sort_values(["type", "mgrno", "permno", "date"])  # stable order
    qid = _quarter_id(df["date"])
    df = df.assign(__qid=qid.values)

    gb = df.groupby(["type", "mgrno", "permno"], sort=False, observed=False)
    total_groups = None
    try:
        total_groups = gb.ngroups
    except Exception:
        pass

    pbar = None
    _use_tqdm = False
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore
            desc = f"windows {label}" if label else "windows"
            pbar = tqdm(total=total_groups if total_groups is not None else None, desc=desc)
            _use_tqdm = True
        except Exception:
            _use_tqdm = False

    seen = 0
    for (tp, mgr, perm), g in gb:
        if len(g) < window_len:
            seen += 1
            if _use_tqdm and pbar is not None:
                pbar.update(1)
            elif progress_every and (seen % progress_every == 0):
                print(f"[windows] processed groups: {seen}")
            continue
        q = g["__qid"].to_numpy()
        idx = g.index.to_numpy()
        for i in range(window_len - 1, len(g)):
            # need strictly consecutive quarters: q[i]-q[i-1]==1 ... for window_len steps
            ok = True
            for k in range(window_len - 1):
                if q[i - k] - q[i - k - 1] != 1:
                    ok = False
                    break
            if not ok:
                continue
            # delta sign if label available
            ht = g.iloc[i]["holding_t"]
            ht1 = g.iloc[i]["holding_t1"] if "holding_t1" in g.columns else np.nan
            dsign = None
            try:
                if pd.notna(ht) and pd.notna(ht1):
                    delta = float(ht1) - float(ht)
                    dsign = 0 if delta == 0 else (1 if delta > 0 else -1)
            except Exception:
                dsign = None

            if window_len == 1:
                idx_t = int(idx[i])
                out.append(
                    WindowIndex(
                        group_key=(tp, mgr, perm),
                        idx_tm3=idx_t,
                        idx_tm2=idx_t,
                        idx_tm1=idx_t,
                        idx_t=idx_t,
                        qid_t=int(q[i]),
                        aum_t=float(g.iloc[i]["aum"]) if "aum" in g.columns and pd.notna(g.iloc[i]["aum"]) else None,
                        delta_sign=dsign,
                        window_len=window_len,
                    )
                )
            else:
                out.append(
                    WindowIndex(
                        group_key=(tp, mgr, perm),
                        idx_tm3=int(idx[i - 3]) if window_len >= 4 else int(idx[max(0, i - 3)]),
                        idx_tm2=int(idx[i - 2]) if window_len >= 3 else int(idx[max(0, i - 2)]),
                        idx_tm1=int(idx[i - 1]),
                        idx_t=int(idx[i]),
                        qid_t=int(q[i]),
                        aum_t=float(g.iloc[i]["aum"]) if "aum" in g.columns and pd.notna(g.iloc[i]["aum"]) else None,
                        delta_sign=dsign,
                        window_len=window_len,
                    )
                )
        seen += 1
        if _use_tqdm and pbar is not None:
            pbar.update(1)
        elif progress_every and (seen % progress_every == 0):
            print(f"[windows] processed groups: {seen}")
    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass
    t1 = time.perf_counter()
    print(f"[windows] built {len(out):,} windows from {seen:,} groups in {t1 - t0:.1f}s")
    return out


def stratified_sample_windows(
    df: pd.DataFrame,
    windows: List[WindowIndex],
    per_type_limit: int = 1000,
    time_bins: int = 10,
    cap_per_pair: int = 3,
    seed: int = 42,
    *,
    exclude_zero_holding_t: bool = False,
) -> List[WindowIndex]:
    """Per-type stratified sampling by time buckets with per (mgrno,permno) cap.
    Returns selected windows (subset of input windows).
    """
    rng = np.random.default_rng(seed)
    # bucketize time by qid_t with approx equal-frequency per type
    selected: List[WindowIndex] = []
    # prepare fast lookups
    for tp, ws in _group_by(windows, key=lambda w: w.group_key[0]).items():
        if not ws:
            continue
        # Optional: drop windows whose t-time holding equals 0.0
        if exclude_zero_holding_t:
            def _is_nonzero(w: WindowIndex) -> bool:
                try:
                    v = df.loc[w.idx_t]["holding_t"]
                    if pd.isna(v):
                        return False
                    return float(v) != 0.0
                except Exception:
                    return False
            ws = [w for w in ws if _is_nonzero(w)]
            if not ws:
                continue
        qids = np.array([w.qid_t for w in ws])
        # compute bin edges by quantiles
        unique_q = np.unique(qids)
        B = int(min(max(3, min(time_bins, len(unique_q))), 12))
        if B <= 1:
            B = 2
        qs = np.linspace(0, 1, B + 1)
        edges = np.quantile(qids, qs)
        # ensure strictly increasing edges
        edges = np.unique(edges)
        # assign bucket ids
        bucket_ids = np.digitize(qids, edges[1:-1], right=True)

        # quota per bucket
        L = per_type_limit
        base = L // B
        rem = L % B

        # group windows by bucket
        buckets = {b: [] for b in range(B)}
        for w, b in zip(ws, bucket_ids):
            buckets[b].append(w)

        # cap tracker per pair (persists across buckets within this type)
        cap = {}

        # remember the index where this type's selection begins
        base_sel_idx = len(selected)

        for b in range(B):
            quota = base + (1 if b < rem else 0)
            cand = buckets.get(b, [])
            if not cand:
                continue
            rng.shuffle(cand)
            # round-robin by (mgrno,permno)
            per_pair_lists = {}
            for w in cand:
                pair = (w.group_key[1], w.group_key[2])
                per_pair_lists.setdefault(pair, []).append(w)
            # iteratively take one per pair until quota
            took = 0
            ptrs = {p: 0 for p in per_pair_lists}
            pairs = list(per_pair_lists.keys())
            rng.shuffle(pairs)
            while took < quota and pairs:
                new_pairs = []
                for p in pairs:
                    used = cap.get(p, 0)
                    if used >= cap_per_pair:
                        continue
                    i = ptrs[p]
                    if i >= len(per_pair_lists[p]):
                        continue
                    selected.append(per_pair_lists[p][i])
                    ptrs[p] = i + 1
                    cap[p] = used + 1
                    took += 1
                    if took >= quota:
                        break
                    new_pairs.append(p)
                pairs = new_pairs
                if not pairs:
                    break

        # Fallback fill: if we couldn't reach L due to sparse buckets, fill from
        # remaining candidates across all buckets while respecting per-pair cap.
        cur_sel = selected[base_sel_idx:]
        if len(cur_sel) < L:
            picked_keys = set((w.idx_tm3, w.idx_tm2, w.idx_tm1, w.idx_t) for w in cur_sel)
            # Build remaining pool grouped by pair
            remaining: List[WindowIndex] = []
            for b in range(B):
                for w in buckets.get(b, []):
                    k = (w.idx_tm3, w.idx_tm2, w.idx_tm1, w.idx_t)
                    if k not in picked_keys:
                        remaining.append(w)
            if remaining:
                rng.shuffle(remaining)
                per_pair_lists = {}
                for w in remaining:
                    pair = (w.group_key[1], w.group_key[2])
                    per_pair_lists.setdefault(pair, []).append(w)
                ptrs = {p: 0 for p in per_pair_lists}
                pairs = list(per_pair_lists.keys())
                rng.shuffle(pairs)
                took_extra = 0
                while len(cur_sel) + took_extra < L and pairs:
                    new_pairs = []
                    for p in pairs:
                        used = cap.get(p, 0)
                        if used >= cap_per_pair:
                            continue
                        i = ptrs[p]
                        if i >= len(per_pair_lists[p]):
                            continue
                        w = per_pair_lists[p][i]
                        selected.append(w)
                        ptrs[p] = i + 1
                        cap[p] = used + 1
                        took_extra += 1
                        if len(cur_sel) + took_extra >= L:
                            break
                        new_pairs.append(p)
                    pairs = new_pairs

    return selected


def _group_by(items: Iterable, key):
    out = {}
    for it in items:
        k = key(it)
        out.setdefault(k, []).append(it)
    return out
