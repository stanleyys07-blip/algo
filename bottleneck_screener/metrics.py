"""Turn RawData into one flat row of screening metrics per ticker.

Conventions
  * Percentages are stored as percent numbers (25.0 means 25%).
  * Changes in margins / intensities are in percentage points ("pp").
  * Inventory days are in days.
  * "yoy" compares the latest quarter with the same quarter a year earlier
    (4 quarters back). If the provider returned fewer quarters, the oldest
    available quarter is used and `yoy_span_q` records the gap.
  * Missing inputs produce NaN, never an exception.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ROW_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue", "Cost Of Goods Sold"],
    "gross_profit": ["Gross Profit"],
    "op_income": ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "inventory": ["Inventory"],
    "deferred_rev": [
        "Current Deferred Revenue", "Current Deferred Liabilities",
        "Non Current Deferred Revenue", "Deferred Revenue",
    ],
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "fcf": ["Free Cash Flow"],
}

NAN = float("nan")


def _row(df, key):
    """Return the first matching line item as a float Series (newest first)."""
    if df is None or df.empty:
        return None
    for name in ROW_ALIASES[key]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce")
            if isinstance(s, pd.DataFrame):  # duplicated row label
                s = s.iloc[0]
            if s.notna().any():
                return s
    return None


def _at(s, i):
    if s is None or len(s) <= i:
        return NAN
    v = s.iloc[i]
    return float(v) if pd.notna(v) else NAN


def _ok(*vals):
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


def _growth(new, old):
    if not _ok(new, old) or old <= 0:
        return NAN
    return (new / old - 1.0) * 100.0


def _ratio(num, den, scale=100.0):
    if not _ok(num, den) or den == 0:
        return NAN
    return num / den * scale


def _yoy_index(s):
    """Index of the comparison quarter: 4 back if available, else the oldest."""
    if s is None:
        return None
    n = int(s.notna().sum())
    if n < 2:
        return None
    return min(4, n - 1)


def _ttm(s, start=0):
    if s is None or len(s) < start + 4:
        return NAN
    window = s.iloc[start:start + 4]
    return float(window.sum()) if window.notna().all() else NAN


def _ret(prices, days):
    if prices is None or len(prices) <= days:
        return NAN
    return (prices.iloc[-1] / prices.iloc[-1 - days] - 1.0) * 100.0


def _rsi(prices, n=14):
    if prices is None or len(prices) <= n:
        return NAN
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100.0 - 100.0 / (1.0 + rs)


def compute_metrics(raw, benchmark_prices=None):
    info = raw.info or {}
    inc, bal, cf = raw.income, raw.balance, raw.cashflow
    m = {"ticker": raw.ticker, "name": info.get("shortName") or raw.ticker,
         "industry": info.get("industry") or "", "error": raw.error}

    rev = _row(inc, "revenue")
    cogs = _row(inc, "cogs")
    gp = _row(inc, "gross_profit")
    if gp is None and rev is not None and cogs is not None:
        gp = rev - cogs
    if cogs is None and rev is not None and gp is not None:
        cogs = rev - gp
    opi = _row(inc, "op_income")
    inv = _row(bal, "inventory")
    dref = _row(bal, "deferred_rev")
    ocf = _row(cf, "ocf")
    capex = _row(cf, "capex")
    fcf = _row(cf, "fcf")
    if fcf is None and ocf is not None and capex is not None:
        fcf = ocf + capex  # capex is reported as a negative number

    k = _yoy_index(rev)
    m["quarters"] = int(rev.notna().sum()) if rev is not None else 0
    m["yoy_span_q"] = k if k is not None else NAN

    # ---------------- Growth ----------------
    m["rev_yoy"] = _growth(_at(rev, 0), _at(rev, k)) if k else NAN
    if not _ok(m["rev_yoy"]) and _ok(info.get("revenueGrowth")):
        m["rev_yoy"] = info["revenueGrowth"] * 100.0
    qoq0 = _growth(_at(rev, 0), _at(rev, 1))
    qoq1 = _growth(_at(rev, 1), _at(rev, 2))
    m["rev_qoq"] = qoq0
    m["rev_accel_pp"] = qoq0 - qoq1 if _ok(qoq0, qoq1) else NAN

    t_eps, f_eps = info.get("trailingEps"), info.get("forwardEps")
    if _ok(t_eps, f_eps) and t_eps > 0:
        m["eps_fwd_growth"] = min((f_eps / t_eps - 1.0) * 100.0, 300.0)
    elif _ok(t_eps, f_eps) and t_eps <= 0 < f_eps:
        m["eps_fwd_growth"] = 300.0  # loss-to-profit turnaround: cap as "very high"
    else:
        m["eps_fwd_growth"] = NAN

    # ---------------- Margins ----------------
    gm0 = _ratio(_at(gp, 0), _at(rev, 0))
    gmk = _ratio(_at(gp, k), _at(rev, k)) if k else NAN
    m["gross_margin"] = gm0 if _ok(gm0) else (info.get("grossMargins") or NAN) * 100.0
    m["gm_change_pp"] = gm0 - gmk if _ok(gm0, gmk) else NAN
    om0 = _ratio(_at(opi, 0), _at(rev, 0))
    omk = _ratio(_at(opi, k), _at(rev, k)) if k else NAN
    m["op_margin"] = om0 if _ok(om0) else (info.get("operatingMargins") or NAN) * 100.0
    m["opm_change_pp"] = om0 - omk if _ok(om0, omk) else NAN

    # ---------------- Free cash flow ----------------
    rev_ttm, fcf_ttm = _ttm(rev), _ttm(fcf)
    if not _ok(fcf_ttm) and _ok(info.get("freeCashflow")):
        fcf_ttm = float(info["freeCashflow"])
    if not _ok(rev_ttm) and _ok(info.get("totalRevenue")):
        rev_ttm = float(info["totalRevenue"])
    m["fcf_margin_ttm"] = _ratio(fcf_ttm, rev_ttm)
    fm0 = _ratio(_at(fcf, 0), _at(rev, 0))
    fmk = _ratio(_at(fcf, k), _at(rev, k)) if k else NAN
    m["fcf_margin_change_pp"] = fm0 - fmk if _ok(fm0, fmk) else NAN
    mcap = info.get("marketCap")
    m["market_cap_b"] = mcap / 1e9 if _ok(mcap) else NAN
    m["fcf_yield"] = _ratio(fcf_ttm, mcap)

    # ---------------- Valuation ----------------
    fpe = info.get("forwardPE")
    m["fwd_pe"] = float(fpe) if _ok(fpe) and fpe > 0 else NAN
    g = m["eps_fwd_growth"]
    m["peg"] = m["fwd_pe"] / g if _ok(m["fwd_pe"], g) and g > 0 else NAN
    m["ev_sales"] = float(info["enterpriseToRevenue"]) if _ok(info.get("enterpriseToRevenue")) else NAN
    eve = info.get("enterpriseToEbitda")
    m["ev_ebitda"] = float(eve) if _ok(eve) and eve > 0 else NAN

    # ---------------- Supply / demand ----------------
    dio0 = _ratio(_at(inv, 0), _at(cogs, 0), 91.0)
    diok = _ratio(_at(inv, k), _at(cogs, k), 91.0) if k else NAN
    m["dio"] = dio0
    m["dio_change"] = dio0 - diok if _ok(dio0, diok) else NAN
    inv_yoy = _growth(_at(inv, 0), _at(inv, k)) if k else NAN
    m["rev_minus_inv_growth_pp"] = m["rev_yoy"] - inv_yoy if _ok(m["rev_yoy"], inv_yoy) else NAN
    capex_ttm = _ttm(capex)
    m["capex_intensity"] = _ratio(abs(capex_ttm), rev_ttm) if _ok(capex_ttm) else NAN
    ci0 = _ratio(abs(_at(capex, 0)), _at(rev, 0)) if _ok(_at(capex, 0)) else NAN
    cik = _ratio(abs(_at(capex, k)), _at(rev, k)) if k and _ok(_at(capex, k)) else NAN
    m["capex_intensity_change_pp"] = ci0 - cik if _ok(ci0, cik) else NAN
    m["deferred_rev_growth"] = _growth(_at(dref, 0), _at(dref, k)) if k else NAN

    # ---------------- Price momentum ----------------
    p = raw.prices if raw.prices is not None and len(raw.prices) else None
    m["price"] = float(p.iloc[-1]) if p is not None else NAN
    m["ret_3m"] = _ret(p, 63)
    m["ret_6m"] = _ret(p, 126)
    m["ret_12m"] = _ret(p, 252)
    if benchmark_prices is not None and len(benchmark_prices):
        b6 = _ret(benchmark_prices, 126)
        m["rs_6m_pp"] = m["ret_6m"] - b6 if _ok(m["ret_6m"], b6) else NAN
    else:
        m["rs_6m_pp"] = NAN
    if p is not None and len(p) >= 200:
        m["pct_vs_200dma"] = (p.iloc[-1] / p.iloc[-200:].mean() - 1.0) * 100.0
        m["sma50_over_200"] = bool(p.iloc[-50:].mean() > p.iloc[-200:].mean())
    else:
        m["pct_vs_200dma"] = NAN
        m["sma50_over_200"] = None
    if p is not None and len(p) >= 20:
        hi = p.iloc[-252:].max()
        m["pct_from_52w_high"] = (p.iloc[-1] / hi - 1.0) * 100.0
    else:
        m["pct_from_52w_high"] = NAN
    m["rsi14"] = _rsi(p)

    m["signals"] = bottleneck_signals(m)
    m["signal_count"] = len(m["signals"])
    return m


# Plain-language checks that make a row's supply-demand story readable.
SIGNAL_RULES = [
    ("Pricing power", lambda m: m["gm_change_pp"] >= 5),
    ("Inventory tightening", lambda m: m["dio_change"] <= -10),
    ("Demand > supply", lambda m: m["rev_minus_inv_growth_pp"] >= 10),
    ("Capacity race", lambda m: m["capex_intensity_change_pp"] >= 2),
    ("Customer prepayments", lambda m: m["deferred_rev_growth"] >= 25),
    ("Revenue accelerating", lambda m: m["rev_accel_pp"] > 0 and m["rev_qoq"] > 0),
    ("FCF inflecting", lambda m: m["fcf_margin_change_pp"] >= 5),
]


def bottleneck_signals(m):
    hits = []
    for label, rule in SIGNAL_RULES:
        try:
            if rule(m):  # NaN comparisons are False, so missing data never fires
                hits.append(label)
        except (TypeError, KeyError):
            pass
    return hits


def build_table(raw_by_ticker, benchmark_prices=None, group_of=None):
    rows = []
    for t, raw in raw_by_ticker.items():
        row = compute_metrics(raw, benchmark_prices)
        row["group"] = group_of(t) if group_of else ""
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ticker")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def quarterly_history(raw):
    """Per-quarter trend table for the detail view (oldest first)."""
    rev = _row(raw.income, "revenue")
    if rev is None:
        return pd.DataFrame()
    gp = _row(raw.income, "gross_profit")
    cogs = _row(raw.income, "cogs")
    if gp is None and cogs is not None:
        gp = rev - cogs
    if cogs is None and gp is not None:
        cogs = rev - gp
    opi = _row(raw.income, "op_income")
    inv = _row(raw.balance, "inventory")
    fcf = _row(raw.cashflow, "fcf")
    if fcf is None:
        ocf, capex = _row(raw.cashflow, "ocf"), _row(raw.cashflow, "capex")
        if ocf is not None and capex is not None:
            fcf = ocf + capex

    df = pd.DataFrame({"Revenue ($M)": rev / 1e6})
    if gp is not None:
        df["Gross margin %"] = gp / rev * 100
    if opi is not None:
        df["Operating margin %"] = opi / rev * 100
    if fcf is not None:
        df["FCF margin %"] = fcf.reindex(rev.index) / rev * 100
    if inv is not None and cogs is not None:
        df["Inventory days"] = inv.reindex(rev.index) / cogs * 91
    df = df.dropna(how="all").sort_index()
    df.index = [d.strftime("%Y-Q") + str((d.month - 1) // 3 + 1) for d in df.index]
    return df
