"""Data loading: live fundamentals + prices from Yahoo Finance (yfinance), with
an on-disk cache, plus a deterministic synthetic "demo" source so the
dashboard can be explored offline.

Every source returns the same `RawData` shape:
    info      dict of snapshot fields (marketCap, forwardPE, ...)
    income    quarterly income statement   (rows = line items, cols = dates, newest first)
    balance   quarterly balance sheet      (same layout)
    cashflow  quarterly cash-flow statement (same layout)
    prices    daily close prices (pd.Series indexed by date, oldest first)
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CACHE_DIR = os.environ.get(
    "SCREENER_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache")
)
CACHE_TTL_HOURS = 12

INFO_FIELDS = [
    "shortName", "sector", "industry", "marketCap", "enterpriseValue",
    "currentPrice", "regularMarketPrice", "trailingPE", "forwardPE",
    "trailingEps", "forwardEps", "pegRatio", "trailingPegRatio",
    "enterpriseToRevenue", "enterpriseToEbitda", "priceToBook",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "grossMargins", "operatingMargins", "profitMargins", "freeCashflow",
    "operatingCashflow", "totalRevenue", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "recommendationMean", "numberOfAnalystOpinions", "targetMeanPrice",
]


@dataclass
class RawData:
    ticker: str
    info: dict = field(default_factory=dict)
    income: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    error: str | None = None
    synthetic: bool = False


# --------------------------------------------------------------------------- #
# Live source (yfinance)
# --------------------------------------------------------------------------- #
def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"{ticker.upper()}.pkl")


def _read_cache(ticker, ttl_hours):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) > ttl_hours * 3600:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _write_cache(raw):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(raw.ticker), "wb") as f:
            pickle.dump(raw, f)
    except Exception:
        pass


def _sorted_cols(df):
    """Newest quarter first, regardless of how the provider ordered it."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = pd.to_datetime(df.columns)
    return df.sort_index(axis=1, ascending=False)


def fetch_live(ticker, ttl_hours=CACHE_TTL_HOURS, use_cache=True):
    ticker = ticker.upper().strip()
    if use_cache:
        cached = _read_cache(ticker, ttl_hours)
        if cached is not None:
            return cached

    import yfinance as yf  # imported lazily so demo mode needs no network lib

    raw = RawData(ticker=ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        raw.info = {k: info.get(k) for k in INFO_FIELDS}
        raw.income = _sorted_cols(t.quarterly_income_stmt)
        raw.balance = _sorted_cols(t.quarterly_balance_sheet)
        raw.cashflow = _sorted_cols(t.quarterly_cashflow)
        hist = t.history(period="2y", auto_adjust=True)
        if hist is not None and not hist.empty:
            prices = hist["Close"].dropna()
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            raw.prices = prices
        if raw.prices.empty and raw.income.empty:
            raw.error = "no data returned"
    except Exception as exc:  # network errors, delisted tickers, rate limits
        raw.error = f"{type(exc).__name__}: {exc}"

    if raw.error is None:
        _write_cache(raw)
    return raw


def fetch_prices_live(ticker, use_cache=True):
    """Price-only fetch, used for benchmarks (SPY / SOXX)."""
    return fetch_live(ticker, use_cache=use_cache).prices


# --------------------------------------------------------------------------- #
# Demo source (synthetic, deterministic per ticker)
# --------------------------------------------------------------------------- #
def _seed(ticker):
    return int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)


def fetch_demo(ticker, n_quarters=8):
    """Synthetic but internally consistent statements and prices.

    Each ticker gets a random "cycle phase": some are in an MU-style upswing
    (accelerating revenue, expanding margins, falling inventory days), others
    are flat or rolling over, so every screen and chart has something to show.
    Numbers are NOT real and are labeled as such in the dashboard.
    """
    rng = np.random.default_rng(_seed(ticker))
    phase = rng.choice(["upswing", "upswing", "steady", "rollover", "trough"])

    quarters = pd.date_range(end=pd.Timestamp("2026-06-30"), periods=n_quarters, freq="QE")
    base_rev = float(rng.uniform(0.3, 12.0)) * 1e9

    if phase == "upswing":
        growth = np.linspace(rng.uniform(-0.03, 0.02), rng.uniform(0.06, 0.15), n_quarters)
        gm = np.linspace(rng.uniform(0.20, 0.40), rng.uniform(0.45, 0.65), n_quarters)
        dio = np.linspace(rng.uniform(140, 200), rng.uniform(70, 110), n_quarters)
    elif phase == "steady":
        growth = np.full(n_quarters, rng.uniform(0.01, 0.05))
        gm = np.full(n_quarters, rng.uniform(0.35, 0.60))
        dio = np.full(n_quarters, rng.uniform(80, 130))
    elif phase == "rollover":
        growth = np.linspace(rng.uniform(0.08, 0.15), rng.uniform(-0.10, -0.02), n_quarters)
        gm = np.linspace(rng.uniform(0.45, 0.60), rng.uniform(0.25, 0.40), n_quarters)
        dio = np.linspace(rng.uniform(80, 110), rng.uniform(150, 220), n_quarters)
    else:  # trough
        growth = np.linspace(rng.uniform(-0.10, -0.05), rng.uniform(-0.02, 0.04), n_quarters)
        gm = np.linspace(rng.uniform(0.10, 0.25), rng.uniform(0.15, 0.30), n_quarters)
        dio = np.linspace(rng.uniform(180, 240), rng.uniform(150, 200), n_quarters)

    growth = growth + rng.normal(0, 0.015, n_quarters)
    gm = np.clip(gm + rng.normal(0, 0.01, n_quarters), 0.02, 0.9)
    revenue = base_rev * np.cumprod(1 + growth)
    cogs = revenue * (1 - gm)
    opex = revenue[0] * rng.uniform(0.12, 0.25) * np.cumprod(np.full(n_quarters, 1.01))
    op_inc = revenue - cogs - opex
    net_inc = op_inc * 0.85
    inventory = cogs * dio / 91.0
    capex_ratio = np.linspace(rng.uniform(0.06, 0.15), rng.uniform(0.08, 0.30), n_quarters)
    capex = -revenue * capex_ratio
    ocf = net_inc + revenue * rng.uniform(0.08, 0.15)
    fcf = ocf + capex
    deferred = revenue * np.linspace(rng.uniform(0.02, 0.05), rng.uniform(0.03, 0.12), n_quarters)

    cols = quarters[::-1]  # newest first, like the live source
    def stmt(rows):
        return pd.DataFrame({k: v[::-1] for k, v in rows.items()}, index=cols).T

    income = stmt({
        "Total Revenue": revenue, "Cost Of Revenue": cogs, "Gross Profit": revenue - cogs,
        "Operating Income": op_inc, "Net Income": net_inc,
        "EBITDA": op_inc + revenue * 0.08,
    })
    balance = stmt({"Inventory": inventory, "Current Deferred Revenue": deferred})
    cashflow = stmt({"Operating Cash Flow": ocf, "Capital Expenditure": capex, "Free Cash Flow": fcf})

    # Prices: drift follows the fundamental phase, plus noise.
    days = pd.bdate_range(end=pd.Timestamp("2026-09-30"), periods=504)
    drift = {"upswing": 0.0016, "steady": 0.0004, "rollover": -0.0006, "trough": 0.0002}[phase]
    rets = rng.normal(drift, rng.uniform(0.015, 0.035), len(days))
    prices = pd.Series(rng.uniform(20, 300) * np.exp(np.cumsum(rets)), index=days)

    ttm_rev = revenue[-4:].sum()
    ttm_ni = net_inc[-4:].sum()
    shares = ttm_rev * rng.uniform(4, 20) / prices.iloc[-1]
    mcap = prices.iloc[-1] * shares
    trailing_eps = ttm_ni / shares
    fwd_eps = trailing_eps * (1 + growth[-1] * 2.5 * rng.uniform(0.6, 1.4))
    info = {
        "shortName": f"{ticker} (demo)", "sector": "Technology", "industry": "Demo",
        "marketCap": mcap, "enterpriseValue": mcap * rng.uniform(0.9, 1.15),
        "currentPrice": prices.iloc[-1],
        "trailingPE": prices.iloc[-1] / trailing_eps if trailing_eps > 0 else None,
        "forwardPE": prices.iloc[-1] / fwd_eps if fwd_eps > 0 else None,
        "trailingEps": trailing_eps, "forwardEps": fwd_eps,
        "enterpriseToRevenue": mcap / ttm_rev,
        "enterpriseToEbitda": mcap / max(income.loc["EBITDA"].iloc[:4].sum(), 1.0),
        "revenueGrowth": revenue[-1] / revenue[-5] - 1,
        "earningsGrowth": None, "grossMargins": gm[-1], "operatingMargins": op_inc[-1] / revenue[-1],
        "freeCashflow": fcf[-4:].sum(), "totalRevenue": ttm_rev,
        "fiftyTwoWeekHigh": prices.iloc[-252:].max(), "fiftyTwoWeekLow": prices.iloc[-252:].min(),
    }
    return RawData(ticker=ticker, info=info, income=income, balance=balance,
                   cashflow=cashflow, prices=prices, synthetic=True)


def fetch_demo_benchmark(ticker):
    rng = np.random.default_rng(_seed(ticker))
    days = pd.bdate_range(end=pd.Timestamp("2026-09-30"), periods=504)
    rets = rng.normal(0.0005, 0.011, len(days))
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=days)


# --------------------------------------------------------------------------- #
# Batch loader
# --------------------------------------------------------------------------- #
def load_universe(tickers, demo=False, max_workers=8, use_cache=True, progress=None):
    """Fetch every ticker (in parallel for live data). Returns {ticker: RawData}."""
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    if demo:
        return {t: fetch_demo(t) for t in tickers}

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {t: pool.submit(fetch_live, t, CACHE_TTL_HOURS, use_cache) for t in tickers}
        for i, (t, fut) in enumerate(futures.items(), 1):
            out[t] = fut.result()
            if progress:
                progress(i / len(tickers), t)
    return out


def load_benchmark(ticker, demo=False, use_cache=True):
    if demo:
        return fetch_demo_benchmark(ticker)
    return fetch_prices_live(ticker, use_cache=use_cache)
