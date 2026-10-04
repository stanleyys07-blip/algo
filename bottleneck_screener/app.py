"""Bottleneck Screener dashboard.

Run:  streamlit run bottleneck_screener/app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import scoring  # noqa: E402
import universe  # noqa: E402

st.set_page_config(page_title="Bottleneck Screener", page_icon="🔍", layout="wide")

# Palette (validated reference categorical slots 1-2 + neutral gray).
PASS_COLOR = "#2a78d6"
REF_COLOR = "#eb6834"
FAIL_COLOR = "#a3a29c"
GRID = "rgba(128,128,128,0.18)"

FMT = {"%": "{:.1f}%", "pp": "{:+.1f} pp", "x": "{:.1f}x", "days": "{:.0f} d", "$B": "${:,.1f}B", "": "{:.0f}"}


def fmt_val(key, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    m = scoring.METRIC_BY_KEY.get(key)
    return FMT[m.unit].format(v) if m else f"{v:.2f}"


def base_layout(fig, height=380, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12), **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# --------------------------------------------------------------------------- #
# Data (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load(tickers: tuple, demo: bool, refresh_token: int):
    use_cache = refresh_token == 0
    raws = data_mod.load_universe(list(tickers), demo=demo, use_cache=use_cache)
    bench = data_mod.load_benchmark(universe.BENCHMARK, demo=demo, use_cache=use_cache)
    table = metrics_mod.build_table(raws, bench, universe.group_of)
    return raws, table


# --------------------------------------------------------------------------- #
# Sidebar: universe, preset, thresholds, weights
# --------------------------------------------------------------------------- #
st.sidebar.title("🔍 Bottleneck Screener")

with st.sidebar.expander("Universe & data", expanded=False):
    demo = st.toggle("Demo mode (synthetic data, offline)", value=os.environ.get("SCREENER_DEMO") == "1",
                     help="Use generated data when Yahoo Finance is unreachable.")
    groups = st.multiselect("Groups", list(universe.UNIVERSE), default=list(universe.UNIVERSE))
    extra = st.text_input("Extra tickers (comma-separated)", "")
    if st.button("Force refresh data"):
        st.session_state["refresh"] = st.session_state.get("refresh", 0) + 1

tickers = []
for g in groups:
    tickers += universe.UNIVERSE[g]
tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
tickers = tuple(dict.fromkeys(tickers))  # de-dupe, keep order
if "MU" not in tickers:
    tickers = ("MU",) + tickers  # MU is always loaded as the reference profile

preset = st.sidebar.selectbox("Preset", list(scoring.PRESETS), index=0)
st.sidebar.caption(scoring.PRESETS[preset]["_desc"])

# Reset widget state when the preset changes so sliders jump to preset values.
if st.session_state.get("_preset") != preset:
    for k in [k for k in st.session_state if k.startswith(("en_", "thr_"))]:
        del st.session_state[k]
    st.session_state["_preset"] = preset

defaults = scoring.preset_filters(preset)
filters = {}
st.sidebar.markdown("### Thresholds")
st.sidebar.caption("Tick a box to turn a filter on; drag to set the cut-off.")
categories = scoring.CATEGORIES + ["Size"]
for cat in categories:
    cat_metrics = [m for m in scoring.METRICS if m.category == cat]
    n_on = sum(defaults[m.key][0] for m in cat_metrics)
    with st.sidebar.expander(f"{cat}" + (f"  ·  {n_on} preset on" if n_on else ""), expanded=n_on > 0):
        for m in cat_metrics:
            en_d, op_d, thr_d = defaults[m.key]
            thr_d = float(min(max(thr_d, m.lo), m.hi))
            c1, c2 = st.columns([1, 3])
            en = c1.checkbox(m.op, value=en_d, key=f"en_{m.key}", help=m.help)
            thr = c2.slider(f"{m.label} ({m.unit})" if m.unit else m.label,
                            float(m.lo), float(m.hi), thr_d, float(m.step),
                            key=f"thr_{m.key}", disabled=not en, help=m.help)
            filters[m.key] = (en, op_d, thr)

missing_passes = st.sidebar.checkbox("Missing data passes filters", value=False,
                                     help="Off: a ticker with no value for an active filter fails it.")

st.sidebar.markdown("### Score weights")
weights = {}
for cat in scoring.CATEGORIES:
    weights[cat] = st.sidebar.slider(cat, 0, 50, scoring.DEFAULT_WEIGHTS[cat], 5, key=f"w_{cat}")

# --------------------------------------------------------------------------- #
# Load + screen
# --------------------------------------------------------------------------- #
with st.spinner(f"Loading {len(tickers)} tickers…"):
    raws, table = load(tickers, demo, st.session_state.get("refresh", 0))

errors = table[table["error"].notna()]
good = table[table["error"].isna()]
if good.empty:
    st.error("No data could be loaded. Yahoo Finance may be unreachable — "
             "switch on **Demo mode** under *Universe & data* to explore the dashboard offline.")
    if not errors.empty:
        st.dataframe(errors[["error"]])
    st.stop()

res = scoring.run_screen(good, filters, weights, missing_passes)
passed = res[res["passes"]]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Next-MU bottleneck screener")
st.caption(
    "Looks for suppliers of scarce inputs showing the Micron pattern: pricing power "
    "(gross-margin expansion), tight supply (falling inventory days, sales outrunning "
    "inventory, prepayments, capex race), cash generation, a multiple that hasn't "
    "caught up, and price confirmation."
)
if demo:
    st.warning("DEMO MODE — numbers are synthetic and do not describe the real companies.")

active = [k for k, v in filters.items() if v[0]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers screened", len(res))
c2.metric("Pass all filters", len(passed))
c3.metric("Active filters", len(active))
if "MU" in res.index:
    c4.metric("MU score (reference)", f"{res.loc['MU', 'score']:.0f}",
              help="MU's composite score on today's data, for calibration.")
if not errors.empty:
    st.caption(f"⚠️ No data for: {', '.join(errors.index)}")

tab_rank, tab_map, tab_detail, tab_method = st.tabs(
    ["Ranked results", "Growth × margin map", "Ticker drill-down", "Method"])

# --------------------------------------------------------------------------- #
# Ranked results
# --------------------------------------------------------------------------- #
with tab_rank:
    show_all = st.toggle("Show tickers that fail filters", value=len(passed) == 0)
    view = res if show_all else passed
    if view.empty:
        st.info("Nothing passes the current thresholds. Loosen a filter, or toggle "
                "'Show tickers that fail filters' to see what's closest (fewest failures).")
    else:
        top = view.head(25).iloc[::-1]
        colors = [REF_COLOR if t == "MU" else (PASS_COLOR if p else FAIL_COLOR)
                  for t, p in zip(top.index, top["passes"])]
        fig = go.Figure(go.Bar(
            x=top["score"], y=top.index, orientation="h", marker_color=colors,
            marker_line_width=0, text=[f"{s:.0f}" for s in top["score"]], textposition="outside",
            customdata=np.stack([top["name"], top["group"], top["signal_count"]], axis=1),
            hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>%{customdata[1]}"
                          "<br>Score %{x:.1f}<br>Bottleneck signals: %{customdata[2]}<extra></extra>",
        ))
        base_layout(fig, height=max(300, 24 * len(top) + 60),
                    title="Composite Bottleneck Score (0–100)",
                    xaxis=dict(range=[0, 105], title=None), yaxis=dict(title=None))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Blue = passes all filters · gray = fails at least one · orange = MU reference.")

        cols = ["name", "group", "score"] + [f"score_{c}" for c in scoring.CATEGORIES] + [
            "rev_yoy", "rev_accel_pp", "gross_margin", "gm_change_pp", "fcf_margin_ttm",
            "fcf_yield", "fwd_pe", "peg", "ev_sales", "dio_change", "rev_minus_inv_growth_pp",
            "deferred_rev_growth", "rs_6m_pp", "pct_vs_200dma", "market_cap_b",
            "signal_count", "signals",
        ]
        if show_all:
            cols.insert(3, "fail_reasons")
        tbl = view[cols].copy()
        tbl["signals"] = tbl["signals"].apply(", ".join)
        if "fail_reasons" in tbl:
            tbl["fail_reasons"] = tbl["fail_reasons"].apply(", ".join)
        col_cfg = {
            "name": "Company", "group": "Group",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "signal_count": st.column_config.NumberColumn("# signals", format="%d"),
            "signals": "Bottleneck signals", "fail_reasons": "Fails on",
        }
        for c in scoring.CATEGORIES:
            col_cfg[f"score_{c}"] = st.column_config.NumberColumn(c, format="%.0f")
        for k in cols:
            m = scoring.METRIC_BY_KEY.get(k)
            if m:
                f = {"%": "%.1f%%", "pp": "%+.1f", "x": "%.1fx", "days": "%+.0f",
                     "$B": "$%.1fB", "": "%.0f"}[m.unit]
                col_cfg[k] = st.column_config.NumberColumn(m.label, format=f, help=m.help)
        st.dataframe(tbl, column_config=col_cfg, use_container_width=True, height=520)
        st.download_button("Download CSV", tbl.to_csv().encode(), "bottleneck_screen.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Growth × margin map
# --------------------------------------------------------------------------- #
with tab_map:
    axis_opts = {m.label: m.key for m in scoring.METRICS}
    a1, a2 = st.columns(2)
    xl = a1.selectbox("X axis", list(axis_opts), index=list(axis_opts.values()).index("rev_yoy"))
    yl = a2.selectbox("Y axis", list(axis_opts), index=list(axis_opts.values()).index("gm_change_pp"))
    xk, yk = axis_opts[xl], axis_opts[yl]
    d = res[[xk, yk, "passes", "name", "score", "market_cap_b"]].dropna(subset=[xk, yk])
    fig = go.Figure()
    for label, mask, color in [
        ("Fails filters", ~d["passes"] & (d.index != "MU"), FAIL_COLOR),
        ("Passes filters", d["passes"] & (d.index != "MU"), PASS_COLOR),
        ("MU (reference)", d.index == "MU", REF_COLOR),
    ]:
        s = d[mask]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s[xk], y=s[yk], mode="markers+text", name=label, text=s.index,
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=11, color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([s["name"], s["score"]], axis=1),
            hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>" + xl + ": %{x:.1f}<br>"
                          + yl + ": %{y:.1f}<br>Score %{customdata[1]:.0f}<extra></extra>",
        ))
    for k, axis in [(xk, "x"), (yk, "y")]:
        en, _, thr = filters[k]
        if en:
            (fig.add_vline if axis == "x" else fig.add_hline)(
                thr, line_dash="dot", line_color=PASS_COLOR, opacity=0.6)
    base_layout(fig, height=560, xaxis_title=xl, yaxis_title=yl,
                legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines mark active thresholds. The MU pattern lives top-right of "
               "revenue growth × gross-margin change.")

# --------------------------------------------------------------------------- #
# Drill-down
# --------------------------------------------------------------------------- #
with tab_detail:
    pick = st.selectbox("Ticker", list(res.index), index=0)
    row = res.loc[pick]
    st.subheader(f"{pick} · {row['name']}")
    st.caption(f"{row['group']} · {row['industry']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Composite score", f"{row['score']:.0f}")
    k2.metric("Passes filters", "Yes" if row["passes"] else "No")
    k3.metric("Bottleneck signals", int(row["signal_count"]))
    k4.metric("Forward P/E", fmt_val("fwd_pe", row["fwd_pe"]))
    if row["signals"]:
        st.success("Signals firing: " + " · ".join(row["signals"]))
    if row["fail_reasons"]:
        st.warning("Fails on: " + " · ".join(row["fail_reasons"]))

    # Category scores vs MU
    cats = scoring.CATEGORIES
    fig = go.Figure()
    fig.add_trace(go.Bar(name=pick, y=cats, x=[row[f"score_{c}"] for c in cats], orientation="h",
                         marker_color=PASS_COLOR, hovertemplate="%{y}: %{x:.0f}<extra>" + pick + "</extra>"))
    if pick != "MU" and "MU" in res.index:
        fig.add_trace(go.Bar(name="MU", y=cats, x=[res.loc["MU", f"score_{c}"] for c in cats],
                             orientation="h", marker_color=REF_COLOR,
                             hovertemplate="%{y}: %{x:.0f}<extra>MU</extra>"))
    base_layout(fig, height=320, title="Category percentile scores", barmode="group",
                bargap=0.3, bargroupgap=0.1, xaxis=dict(range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)

    # Metric table
    rows = []
    for m in scoring.METRICS:
        if m.key in row:
            en, op, thr = filters[m.key]
            v = row[m.key]
            ok = "" if not en else ("✅" if pd.notna(v) and (v >= thr if op == ">=" else v <= thr) else "❌")
            rows.append({"Category": m.category, "Metric": m.label, "Value": fmt_val(m.key, v),
                         "MU": fmt_val(m.key, res.loc["MU", m.key]) if "MU" in res.index else "—",
                         "Filter": f"{op} {thr:g}" if en else "", "Pass": ok})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=420)

    # Quarterly trends: one chart per measure (no dual axes)
    hist = metrics_mod.quarterly_history(raws[pick])
    if not hist.empty:
        st.markdown("##### Quarterly trends")
        cols = st.columns(2)
        for i, col in enumerate(hist.columns):
            s = hist[col].dropna()
            if s.empty:
                continue
            f = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers",
                                     line=dict(width=2, color=PASS_COLOR), marker=dict(size=8),
                                     hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
            base_layout(f, height=240, title=col, showlegend=False)
            cols[i % 2].plotly_chart(f, use_container_width=True)

    p = raws[pick].prices
    if p is not None and len(p):
        f = go.Figure()
        f.add_trace(go.Scatter(x=p.index, y=p.values, name="Close", line=dict(width=2, color=PASS_COLOR),
                               hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Close</extra>"))
        if len(p) >= 200:
            sma = p.rolling(200).mean()
            f.add_trace(go.Scatter(x=sma.index, y=sma.values, name="200-day avg",
                                   line=dict(width=2, color=FAIL_COLOR, dash="dot"),
                                   hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>200-day</extra>"))
        base_layout(f, height=320, title="Price vs 200-day average", hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
        st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
with tab_method:
    st.markdown("""
**What a "MU-type bottleneck" looks like in the numbers**

Micron's 2023–24 run came from owning a scarce input (HBM/DRAM) when AI demand
outran industry capacity. The tells, roughly in the order they appeared:

1. **Inventory days collapse** and revenue grows faster than inventory — the channel empties.
2. **Gross margin expands sharply** — pricing power before volume.
3. **Revenue re-accelerates**, customers **prepay** to lock in supply (deferred revenue ↑),
   and the company **raises capex** to chase demand.
4. **FCF swings positive** while the **forward P/E / PEG still look like a trough**.
5. The **stock reclaims its 200-day** and outperforms the market.

**Filters** are hard pass/fail cut-offs. **Score** ranks every ticker against the universe:
each metric becomes a 0–100 percentile (lower is better for valuation multiples and
inventory-day change), metrics average into six category scores, and your weights combine
them. Missing data scores 0, so thin coverage is penalized rather than rewarded.

**Bottleneck signals** are plain-language checks (e.g. *Pricing power* = gross margin
+5 pp YoY) — a quick read on how many supply-tightness tells are firing at once.

**Data**: Yahoo Finance via `yfinance` (quarterly statements, consensus forward EPS,
2-year prices), cached for 12 hours. Statement coverage varies by company — foreign
filers (ADRs) often report semi-annually, so YoY comparisons may span fewer quarters.

*Screening output is a research starting point, not investment advice.*
""")
