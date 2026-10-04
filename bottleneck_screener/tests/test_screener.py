import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod  # noqa: E402
import metrics  # noqa: E402
import scoring  # noqa: E402


def _stmt(rows, n):
    cols = pd.date_range(end="2026-06-30", periods=n, freq="QE")[::-1]  # newest first
    return pd.DataFrame({c: [v[i] for v in rows.values()] for i, c in enumerate(cols)},
                        index=list(rows)).astype(float)


@pytest.fixture
def raw():
    # Newest first, five quarters, the way yfinance returns them.
    rev = [200, 160, 130, 110, 100]
    cogs = [80, 80, 78, 72, 70]          # GM 60% now vs 30% a year ago
    inc = _stmt({"Total Revenue": rev, "Cost Of Revenue": cogs,
                 "Gross Profit": [r - c for r, c in zip(rev, cogs)],
                 "Operating Income": [80, 50, 30, 15, 10]}, 5)
    bal = _stmt({"Inventory": [88, 100, 110, 120, 140],
                 "Current Deferred Revenue": [30, 25, 20, 15, 10]}, 5)
    cf = _stmt({"Operating Cash Flow": [70, 50, 40, 30, 20],
                "Capital Expenditure": [-30, -25, -20, -15, -10]}, 5)
    prices = pd.Series(np.linspace(50, 100, 300), index=pd.bdate_range(end="2026-09-30", periods=300))
    info = {"marketCap": 4_000.0, "forwardPE": 12.0, "trailingEps": 2.0, "forwardEps": 4.0,
            "enterpriseToRevenue": 6.0}
    return data_mod.RawData("TEST", info, inc, bal, cf, prices)


def test_growth_and_margins(raw):
    m = metrics.compute_metrics(raw)
    assert m["rev_yoy"] == pytest.approx(100.0)
    assert m["rev_qoq"] == pytest.approx(25.0)
    assert m["rev_accel_pp"] == pytest.approx(25.0 - (160 / 130 - 1) * 100)
    assert m["gross_margin"] == pytest.approx(60.0)
    assert m["gm_change_pp"] == pytest.approx(60.0 - 30.0)
    assert m["op_margin"] == pytest.approx(40.0)
    assert m["eps_fwd_growth"] == pytest.approx(100.0)
    assert m["peg"] == pytest.approx(0.12)


def test_fcf_and_supply_demand(raw):
    m = metrics.compute_metrics(raw)
    # FCF derived from OCF + capex when no "Free Cash Flow" row
    fcf_ttm = (70 - 30) + (50 - 25) + (40 - 20) + (30 - 15)
    assert m["fcf_margin_ttm"] == pytest.approx(fcf_ttm / 600 * 100)
    assert m["fcf_yield"] == pytest.approx(fcf_ttm / 4000 * 100)
    assert m["dio"] == pytest.approx(88 / 80 * 91)
    assert m["dio_change"] == pytest.approx(88 / 80 * 91 - 140 / 70 * 91)
    assert m["rev_minus_inv_growth_pp"] == pytest.approx(100.0 - (88 / 140 - 1) * 100)
    assert m["deferred_rev_growth"] == pytest.approx(200.0)
    assert m["capex_intensity_change_pp"] == pytest.approx(15.0 - 10.0)
    assert {"Pricing power", "Inventory tightening", "Demand > supply",
            "Capacity race", "Customer prepayments"} <= set(m["signals"])


def test_momentum(raw):
    bench = pd.Series(np.linspace(100, 110, 300), index=raw.prices.index)
    m = metrics.compute_metrics(raw, bench)
    assert m["ret_6m"] > 0 and m["pct_vs_200dma"] > 0
    assert m["pct_from_52w_high"] == pytest.approx(0.0)
    assert m["rs_6m_pp"] == pytest.approx(m["ret_6m"] - (110 / bench.iloc[-127] - 1) * 100)
    assert m["rsi14"] == pytest.approx(100.0)


def test_empty_data_is_nan_not_error():
    m = metrics.compute_metrics(data_mod.RawData("EMPTY", error="no data"))
    assert math.isnan(m["rev_yoy"]) and math.isnan(m["dio_change"]) and m["signals"] == []


def test_short_history_uses_oldest_quarter(raw):
    raw.income = raw.income.iloc[:, :3]
    raw.balance = raw.balance.iloc[:, :3]
    m = metrics.compute_metrics(raw)
    assert m["yoy_span_q"] == 2
    assert m["rev_yoy"] == pytest.approx((200 / 130 - 1) * 100)


def test_filters_and_scores():
    df = pd.DataFrame({"rev_yoy": [50, 10, np.nan], "peg": [0.8, 2.0, 1.0],
                       "gm_change_pp": [10, -5, 0]}, index=["A", "B", "C"])
    filters = {"rev_yoy": (True, ">=", 20), "peg": (True, "<=", 1.0)}
    passed, reasons = scoring.apply_filters(df, filters)
    assert passed.to_dict() == {"A": True, "B": False, "C": False}
    assert any("n/a" in r for r in reasons["C"])
    passed2, _ = scoring.apply_filters(df, filters, missing_passes=True)
    assert passed2["C"]

    res = scoring.run_screen(df, filters, scoring.DEFAULT_WEIGHTS)
    assert list(res.index)[0] == "A"
    assert res.loc["A", "score_Valuation"] > res.loc["B", "score_Valuation"]  # lower PEG is better


def test_every_preset_runs_on_demo_data():
    raws = data_mod.load_universe(["MU", "NVDA", "VRT", "CEG", "LITE"], demo=True)
    table = metrics.build_table(raws, data_mod.load_benchmark("SPY", demo=True))
    for name in scoring.PRESETS:
        res = scoring.run_screen(table, scoring.preset_filters(name), scoring.DEFAULT_WEIGHTS)
        assert res["score"].between(0, 100).all()
    assert len(metrics.quarterly_history(raws["MU"])) == 8
