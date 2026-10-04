"""Command-line screener (no dashboard).

Examples:
  python bottleneck_screener/cli.py                          # default preset, full universe
  python bottleneck_screener/cli.py --preset "Early turn (pre-momentum)"
  python bottleneck_screener/cli.py --tickers MU,SNDK,WDC --all
  python bottleneck_screener/cli.py --set rev_yoy>=25 --set peg<=1 --csv out.csv
  python bottleneck_screener/cli.py --demo                   # offline, synthetic data
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import scoring  # noqa: E402
import universe  # noqa: E402

SHOW = ["group", "score", "rev_yoy", "gm_change_pp", "fcf_margin_ttm", "fwd_pe", "peg",
        "dio_change", "rs_6m_pp", "signal_count"]


def parse_override(expr):
    m = re.fullmatch(r"\s*(\w+)\s*(>=|<=)\s*(-?[\d.]+)\s*", expr)
    if not m or m.group(1) not in scoring.METRIC_BY_KEY:
        valid = ", ".join(scoring.METRIC_BY_KEY)
        raise argparse.ArgumentTypeError(f"bad --set '{expr}'. Use key>=N or key<=N; keys: {valid}")
    return m.group(1), m.group(2), float(m.group(3))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find MU-type bottleneck stocks.")
    ap.add_argument("--preset", default="MU-type inflection", choices=list(scoring.PRESETS))
    ap.add_argument("--tickers", help="comma-separated list (default: built-in universe)")
    ap.add_argument("--set", action="append", default=[], type=parse_override,
                    metavar="KEY>=N", help="add/override a filter, repeatable")
    ap.add_argument("--off", action="append", default=[], metavar="KEY", help="disable a preset filter")
    ap.add_argument("--all", action="store_true", help="also list tickers that fail filters")
    ap.add_argument("--missing-passes", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic offline data")
    ap.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")
    ap.add_argument("--csv", help="write the full result table to this file")
    args = ap.parse_args(argv)

    tickers = args.tickers.split(",") if args.tickers else universe.all_tickers()
    filters = scoring.preset_filters(args.preset)
    for key, op, thr in args.set:
        filters[key] = (True, op, thr)
    for key in args.off:
        en, op, thr = filters[key]
        filters[key] = (False, op, thr)

    def progress(frac, t):
        print(f"\r  loading {frac:5.0%} {t:<6}", end="", file=sys.stderr, flush=True)

    raws = data_mod.load_universe(tickers, demo=args.demo, use_cache=not args.refresh, progress=progress)
    print(file=sys.stderr)
    bench = data_mod.load_benchmark(universe.BENCHMARK, demo=args.demo, use_cache=not args.refresh)
    table = metrics_mod.build_table(raws, bench, universe.group_of)

    bad = table[table["error"].notna()]
    if len(bad):
        print(f"No data for: {', '.join(bad.index)}", file=sys.stderr)
    table = table[table["error"].isna()]
    if table.empty:
        print("Nothing loaded (network blocked?). Try --demo.", file=sys.stderr)
        return 1

    res = scoring.run_screen(table, filters, scoring.DEFAULT_WEIGHTS, args.missing_passes)
    active = {k: v for k, v in filters.items() if v[0]}
    print(f"Preset: {args.preset}" + (" [DEMO DATA]" if args.demo else ""))
    print("Filters: " + (", ".join(f"{k} {op} {thr:g}" for k, (_, op, thr) in active.items()) or "none"))
    view = res if args.all else res[res["passes"]]
    print(f"{int(res['passes'].sum())} of {len(res)} pass\n")
    if not view.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", "{:,.1f}".format):
            cols = SHOW + (["passes"] if args.all else [])
            print(view[cols].to_string())
    if args.csv:
        out = res.copy()
        out["signals"] = out["signals"].apply("; ".join)
        out["fail_reasons"] = out["fail_reasons"].apply("; ".join)
        out.to_csv(args.csv)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
