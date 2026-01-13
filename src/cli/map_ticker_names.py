from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_mapping(csv_path: Path) -> Dict[int, Tuple[str | None, str | None]]:
    """Load PERMNO -> (COMNAM, TICKER) mapping.

    If multiple rows per PERMNO exist, keep the last occurrence assuming it is the latest.
    Columns are case-insensitive; expected names: PERMNO, COMNAM, TICKER.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    need = ["permno", "comnam", "ticker"]
    for n in need:
        if n not in cols:
            raise ValueError(f"mapping file missing required column: {n} (case-insensitive)")
    # standardize types
    perm_col = cols["permno"]
    name_col = cols["comnam"]
    tick_col = cols["ticker"]

    # keep last occurrence per PERMNO
    df = df[[perm_col, name_col, tick_col]].copy()
    # Ensure permno is int-like
    try:
        df[perm_col] = pd.to_numeric(df[perm_col], errors="coerce").astype("Int64")
    except Exception:
        pass
    df = df.dropna(subset=[perm_col])
    # last occurrence per PERMNO
    df_last = df.groupby(perm_col, sort=False).tail(1)

    mapping: Dict[int, Tuple[str | None, str | None]] = {}
    for _, r in df_last.iterrows():
        try:
            k = int(r[perm_col])
        except Exception:
            continue
        name = str(r[name_col]).strip() if pd.notna(r[name_col]) else None
        ticker = str(r[tick_col]).strip() if pd.notna(r[tick_col]) else None
        mapping[k] = (name if name else None, ticker if ticker else None)
    return mapping


def replace_ticker_line(prompt: str, permno: int | None, name: str | None, ticker: str | None, mode: str) -> str:
    """Replace the 'Ticker: xxx' line.

    - mode='append': Ticker: 12345 -> Ticker: {ticker or 12345} | Company: {name}
    - mode='replace': Ticker: 12345 -> Company: {name}
    Falls back gracefully if mapping not found or no ticker line present.
    """
    # Find a line starting with 'Ticker:'
    lines = prompt.splitlines()
    out_lines = []
    pattern = re.compile(r"^Ticker:\s*(.+?)\s*$")
    replaced = False
    for ln in lines:
        m = pattern.match(ln)
        if m and not replaced:
            # build replacement text
            disp_ticker = ticker or (str(permno) if permno is not None else m.group(1))
            if name:
                if mode == "replace":
                    new_ln = f"Company: {name}"
                else:
                    new_ln = f"Ticker: {disp_ticker} | Company: {name}"
            else:
                # no name found; optionally still normalize ticker text
                new_ln = f"Ticker: {disp_ticker}"
            out_lines.append(new_ln)
            replaced = True
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser(description="Map prompt 'Ticker:' to company name via PERMNO mapping")
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="Input prompts jsonl file or directory of jsonl files")
    ap.add_argument("--out", dest="out", type=str, required=True,
                    help="Output jsonl file or directory (mirrors input kind)")
    ap.add_argument("--mapping", type=str, default="data/ticker_mapping.csv",
                    help="CSV path with columns PERMNO, COMNAM, TICKER")
    ap.add_argument("--mode", type=str, choices=["append", "replace"], default="append",
                    help="How to render the company name relative to the existing Ticker line")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    mp = load_mapping(Path(args.mapping))

    files = [inp] if inp.is_file() else sorted(inp.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no input files found under: {inp}")

    # Prepare output
    if inp.is_file():
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_is_dir = False
    else:
        outp.mkdir(parents=True, exist_ok=True)
        out_is_dir = True

    n = 0
    for fp in files:
        # per-file output path
        cur_out = outp / fp.name if out_is_dir else outp
        with fp.open("r", encoding="utf-8") as fin, cur_out.open("w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prompt = rec.get("prompt") or rec.get("query")
                if not prompt:
                    fout.write(line)
                    continue
                # get permno from record or parse from prompt line
                permno_val = rec.get("permno")
                permno_int = None
                if permno_val is not None:
                    try:
                        permno_int = int(permno_val)
                    except Exception:
                        pass
                if permno_int is None:
                    # attempt to parse from Ticker line
                    m = re.search(r"^Ticker:\s*(\d+)", prompt, flags=re.MULTILINE)
                    if m:
                        try:
                            permno_int = int(m.group(1))
                        except Exception:
                            permno_int = None
                name = ticker = None
                if permno_int is not None and permno_int in mp:
                    name, ticker = mp[permno_int]

                new_prompt = replace_ticker_line(prompt, permno_int, name, ticker, args.mode)
                if new_prompt != prompt:
                    rec["prompt"] = new_prompt
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    break
        if args.limit and n >= args.limit:
            break

    print(f"mapped {n} prompts -> {outp}")


if __name__ == "__main__":
    main()

