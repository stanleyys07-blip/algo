# Next-MU Bottleneck Screener

Screens a universe of AI / data-center / electrification suppliers for the
**Micron pattern**: a company that owns a scarce, hard-to-expand input where
tight supply shows up as pricing power before the multiple catches up.

## Quick start

```bash
pip install -r bottleneck_screener/requirements.txt
streamlit run bottleneck_screener/app.py          # dashboard
python bottleneck_screener/cli.py                 # terminal output
```

Data comes from Yahoo Finance through `yfinance` and is cached in
`bottleneck_screener/.cache/` for 12 hours. With no internet access, turn on
**Demo mode** under *Universe & data*, or run with `SCREENER_DEMO=1` or
`cli.py --demo`. Demo mode uses synthetic numbers that do not describe the
real companies.

## What it measures

| Category | Metrics |
|---|---|
| Growth | Revenue YoY, revenue acceleration (QoQ change in QoQ growth), forward EPS growth |
| Margins | Gross / operating margin, and their YoY change in pp (pricing power) |
| FCF | TTM FCF margin, FCF margin change YoY, FCF yield |
| Valuation | Forward P/E, PEG, EV/Sales, EV/EBITDA |
| Supply-demand | Inventory-days change, revenue growth − inventory growth, capex-intensity change, deferred-revenue (prepayment) growth |
| Momentum | 6/12-month return, 6-month relative strength vs SPY, % vs 200-day, % from 52-week high, RSI |

**Filters** are hard pass/fail thresholds that you set in the sidebar.
**Score** turns each metric into a 0–100 percentile within the universe,
averages the percentiles into six category scores, and combines those with
the weights you choose. Missing data scores 0. **Bottleneck signals** are
plain checks, for example *Pricing power* means gross margin is up at least
5 pp YoY. MU is always loaded as a reference row.

## Presets

* **MU-type inflection** (default): rev ≥15% YoY and accelerating, GM +3 pp,
  FCF ≥ 0, PEG ≤ 1.5, fwd P/E ≤ 40, inventory days falling, above 200-day.
* **Early turn (pre-momentum)**: fundamentals turning, stock still ≥20% off its high.
* **Quality compounder**: high margins, real FCF, outperforming.
* **Momentum breakout**: strong relative strength, RSI not overbought.
* **Show everything**: no filters, ranked by score only.

## CLI

```bash
python bottleneck_screener/cli.py --preset "Early turn (pre-momentum)"
python bottleneck_screener/cli.py --set "rev_yoy>=25" --set "peg<=1" --off fwd_pe
python bottleneck_screener/cli.py --tickers MU,SNDK,WDC,STX --all --csv out.csv
```

## Files

* `universe.py`: default ticker groups. Edit these to change the universe.
* `data.py`: yfinance loader, disk cache and synthetic demo source.
* `metrics.py`: raw statements → one metrics row per ticker.
* `scoring.py`: metric definitions, presets, filters and percentile scoring.
* `app.py`: Streamlit dashboard. `cli.py`: command-line version.
* `tests/`: run with `python -m pytest bottleneck_screener/tests`.

## Caveats

Yahoo statement coverage varies. Foreign filers often report semi-annually,
so YoY can span fewer quarters (see the `yoy_span_q` column). Inventory
metrics mean little for companies that hold no inventory, and forward
estimates are consensus figures. Use the output as a research starting
point. It is not investment advice.
