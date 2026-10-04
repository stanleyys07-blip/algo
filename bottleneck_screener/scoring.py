"""Threshold filters, percentile scoring and presets.

Two independent ways to rank the universe:
  1. Filters  - hard pass/fail thresholds on any metric (the dashboard sliders).
  2. Score    - each metric is percentile-ranked within the universe (0-100,
                direction-aware), averaged into six category scores, then into
                a weighted composite "Bottleneck Score".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = ["Growth", "Margins", "FCF", "Valuation", "Supply-Demand", "Momentum"]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    category: str
    higher_is_better: bool
    unit: str          # "%", "pp", "x", "days", "$B", ""
    lo: float          # slider range
    hi: float
    step: float
    help: str
    scored: bool = True

    @property
    def op(self):
        return ">=" if self.higher_is_better else "<="


METRICS = [
    # Growth
    Metric("rev_yoy", "Revenue growth YoY", "Growth", True, "%", -50, 200, 1,
           "Latest quarter vs same quarter last year."),
    Metric("rev_accel_pp", "Revenue acceleration", "Growth", True, "pp", -30, 30, 0.5,
           "Latest QoQ growth minus prior QoQ growth. >0 = growth is speeding up."),
    Metric("eps_fwd_growth", "Forward EPS growth", "Growth", True, "%", -50, 300, 5,
           "Forward EPS vs trailing EPS (consensus). Capped at 300%."),
    # Margins
    Metric("gross_margin", "Gross margin", "Margins", True, "%", 0, 90, 1,
           "Latest quarter gross margin."),
    Metric("gm_change_pp", "Gross margin change YoY", "Margins", True, "pp", -30, 40, 0.5,
           "Pricing-power tell: the first thing that moves when supply is tight."),
    Metric("op_margin", "Operating margin", "Margins", True, "%", -50, 70, 1,
           "Latest quarter operating margin."),
    Metric("opm_change_pp", "Operating margin change YoY", "Margins", True, "pp", -40, 50, 0.5,
           "Operating leverage on rising prices."),
    # FCF
    Metric("fcf_margin_ttm", "FCF margin (TTM)", "FCF", True, "%", -50, 60, 1,
           "Trailing-twelve-month free cash flow / revenue."),
    Metric("fcf_margin_change_pp", "FCF margin change YoY", "FCF", True, "pp", -40, 50, 0.5,
           "Quarterly FCF margin now vs a year ago."),
    Metric("fcf_yield", "FCF yield", "FCF", True, "%", -10, 15, 0.25,
           "TTM free cash flow / market cap."),
    # Valuation
    Metric("fwd_pe", "Forward P/E", "Valuation", False, "x", 0, 150, 1,
           "Price / next-12-month consensus EPS."),
    Metric("peg", "PEG (fwd P/E ÷ EPS growth)", "Valuation", False, "x", 0, 5, 0.05,
           "<1 = growth cheaper than the multiple implies."),
    Metric("ev_sales", "EV / Sales", "Valuation", False, "x", 0, 40, 0.25,
           "Enterprise value / TTM revenue."),
    Metric("ev_ebitda", "EV / EBITDA", "Valuation", False, "x", 0, 100, 1,
           "Enterprise value / TTM EBITDA."),
    # Supply-demand
    Metric("dio_change", "Inventory days change YoY", "Supply-Demand", False, "days", -150, 150, 1,
           "Negative = inventory turning faster: demand outrunning supply."),
    Metric("rev_minus_inv_growth_pp", "Revenue growth − inventory growth", "Supply-Demand", True, "pp",
           -100, 150, 1, "Sales growing faster than inventory = channel is lean."),
    Metric("capex_intensity_change_pp", "Capex intensity change YoY", "Supply-Demand", True, "pp",
           -20, 30, 0.5, "Rising capex/revenue = company racing to add capacity."),
    Metric("deferred_rev_growth", "Deferred revenue growth YoY", "Supply-Demand", True, "%", -50, 300, 5,
           "Customer prepayments / contract liabilities: buyers locking in supply."),
    Metric("dio", "Inventory days", "Supply-Demand", False, "days", 0, 400, 5,
           "Current inventory / quarterly COGS × 91.", scored=False),
    # Momentum
    Metric("ret_6m", "6-month return", "Momentum", True, "%", -60, 300, 1,
           "Total price change over ~126 trading days."),
    Metric("ret_12m", "12-month return", "Momentum", True, "%", -80, 500, 1,
           "Total price change over ~252 trading days."),
    Metric("rs_6m_pp", "6-month relative strength vs SPY", "Momentum", True, "pp", -80, 300, 1,
           "6-month return minus SPY's 6-month return."),
    Metric("pct_vs_200dma", "% above 200-day average", "Momentum", True, "%", -60, 150, 1,
           "Trend filter: >0 means above the 200-day moving average."),
    Metric("pct_from_52w_high", "% from 52-week high", "Momentum", True, "%", -90, 0, 1,
           "0 = at the high. Use a ceiling (e.g. ≤ -20) in 'early turn' screens.", scored=False),
    Metric("rsi14", "RSI (14)", "Momentum", True, "", 0, 100, 1,
           "Use as a ceiling to avoid chasing overbought names.", scored=False),
    # Size (filter only)
    Metric("market_cap_b", "Market cap", "Size", True, "$B", 0, 5000, 1,
           "Minimum size / liquidity floor.", scored=False),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}


# --------------------------------------------------------------------------- #
# Presets. Each entry: metric -> (enabled, op, threshold).
# --------------------------------------------------------------------------- #
PRESETS = {
    "MU-type inflection": {
        "_desc": "Supply-constrained supplier mid-upswing: accelerating sales, expanding "
                 "margins, inventory tightening, FCF positive, multiple still reasonable, "
                 "trend confirmed.",
        "rev_yoy": (True, ">=", 15), "rev_accel_pp": (True, ">=", 0),
        "gm_change_pp": (True, ">=", 3), "fcf_margin_ttm": (True, ">=", 0),
        "peg": (True, "<=", 1.5), "fwd_pe": (True, "<=", 40),
        "dio_change": (True, "<=", 0), "rev_minus_inv_growth_pp": (True, ">=", 0),
        "pct_vs_200dma": (True, ">=", 0), "market_cap_b": (True, ">=", 1),
    },
    "Early turn (pre-momentum)": {
        "_desc": "Catch it before the chart does: fundamentals turning up while the stock "
                 "is still well off its high.",
        "rev_accel_pp": (True, ">=", 0), "gm_change_pp": (True, ">=", 0),
        "dio_change": (True, "<=", -5), "rev_minus_inv_growth_pp": (True, ">=", 0),
        "ev_sales": (True, "<=", 8), "pct_from_52w_high": (True, "<=", -20),
        "market_cap_b": (True, ">=", 1),
    },
    "Quality compounder": {
        "_desc": "Already-profitable bottleneck owners with high margins and real cash flow.",
        "rev_yoy": (True, ">=", 20), "gross_margin": (True, ">=", 45),
        "op_margin": (True, ">=", 20), "fcf_margin_ttm": (True, ">=", 15),
        "peg": (True, "<=", 2), "rs_6m_pp": (True, ">=", 0),
        "market_cap_b": (True, ">=", 5),
    },
    "Momentum breakout": {
        "_desc": "Strong trend with fundamentals confirming; skip overbought prints.",
        "rev_yoy": (True, ">=", 10), "rs_6m_pp": (True, ">=", 15),
        "pct_vs_200dma": (True, ">=", 10), "pct_from_52w_high": (True, ">=", -10),
        "rsi14": (True, "<=", 75), "market_cap_b": (True, ">=", 1),
    },
    "Show everything": {"_desc": "No filters - rank the whole universe by score."},
}

DEFAULT_WEIGHTS = {
    "Growth": 20, "Margins": 20, "FCF": 15, "Valuation": 15, "Supply-Demand": 20, "Momentum": 10,
}


def preset_filters(name):
    """Full filter dict (every metric present) for a preset."""
    p = PRESETS[name]
    out = {}
    for m in METRICS:
        default_thr = 0.0 if m.lo <= 0 <= m.hi else m.lo
        out[m.key] = p.get(m.key, (False, m.op, default_thr))
    return out


# --------------------------------------------------------------------------- #
# Filtering and scoring
# --------------------------------------------------------------------------- #
def apply_filters(df, filters, missing_passes=False):
    """Return (passed_mask, fail_reasons Series[list[str]])."""
    passed = pd.Series(True, index=df.index)
    reasons = {t: [] for t in df.index}
    for key, (enabled, op, thr) in filters.items():
        if not enabled or key not in df.columns:
            continue
        col = pd.to_numeric(df[key], errors="coerce")
        ok = col >= thr if op == ">=" else col <= thr
        missing = col.isna()
        ok = ok | missing if missing_passes else ok & ~missing
        label = METRIC_BY_KEY[key].label if key in METRIC_BY_KEY else key
        for t in df.index[~ok]:
            reasons[t].append(f"{label} {'n/a' if missing[t] else 'fails'}")
        passed &= ok
    return passed, pd.Series(reasons, dtype=object).reindex(df.index)


def percentile_scores(df):
    """0-100 percentile rank per scored metric, flipped for lower-is-better."""
    out = pd.DataFrame(index=df.index)
    for m in METRICS:
        if not m.scored or m.key not in df.columns:
            continue
        col = pd.to_numeric(df[m.key], errors="coerce")
        if col.notna().sum() < 2:
            continue
        # A negative or missing multiple (loss-making) is not "cheap": rank it worst.
        if m.category == "Valuation":
            col = col.where(col > 0)
        r = col.rank(pct=True, ascending=m.higher_is_better) * 100
        out[m.key] = r
    return out


def category_scores(pct):
    out = pd.DataFrame(index=pct.index)
    for cat in CATEGORIES:
        keys = [m.key for m in METRICS if m.category == cat and m.scored and m.key in pct.columns]
        if keys:
            # Missing metric counts as 0 within the category, but a ticker missing
            # the whole category gets NaN and is handled at the composite step.
            sub = pct[keys]
            cover = sub.notna().sum(axis=1)
            out[cat] = sub.fillna(0).sum(axis=1).div(len(keys)).where(cover > 0)
        else:
            out[cat] = np.nan
    return out


def composite_score(cats, weights):
    w = pd.Series({c: float(weights.get(c, 0)) for c in CATEGORIES})
    if w.sum() <= 0:
        return pd.Series(np.nan, index=cats.index)
    # Missing category scores as 0 so incomplete data is penalized, not rewarded.
    return (cats[CATEGORIES].fillna(0) * w).sum(axis=1) / w.sum()


def run_screen(df, filters, weights, missing_passes=False):
    """Full pipeline. Returns the metric table with score/pass columns, sorted."""
    pct = percentile_scores(df)
    cats = category_scores(pct)
    out = df.copy()
    for c in CATEGORIES:
        out[f"score_{c}"] = cats[c]
    out["score"] = composite_score(cats, weights)
    passed, reasons = apply_filters(df, filters, missing_passes)
    out["passes"] = passed
    out["fail_reasons"] = reasons
    out["n_fails"] = reasons.apply(len)
    return out.sort_values(["passes", "score"], ascending=[False, False])
