# =============================================================================
# Moomoo Algo - Code Strategy
# EMA(9/21) Crossover + RSI(14) Filter  |  US Stocks  |  15-Minute Bars
# -----------------------------------------------------------------------------
# BUY : EMA fast crosses ABOVE EMA slow  AND  RSI < rsi_buy_max (default 60)
# SELL: EMA fast crosses BELOW EMA slow  OR   RSI > rsi_overbought (default 70)
#
# Signals are evaluated on COMPLETED 15-min bars (select=2 vs select=3) so a
# signal cannot "repaint" while the current bar is still forming.
#
# HOW TO SET THE US TICKER:
#   This script does NOT hard-code a ticker. `self.trig_symbol_1` is bound to
#   whatever symbol you pick in the Moomoo UI when you start the strategy:
#     1. Open Moomoo > Algo > Code Strategy, paste this script and save it.
#     2. Click "Run" / "Backtest". In the parameter panel, find the
#        "Trigger Symbol" field and search for the US ticker, e.g. AAPL,
#        TSLA, NVDA, SPY (these appear as US.AAPL, US.TSLA, ...).
#     3. Make sure the account selected is a US stock (USD) account.
#   To run the same logic on another ticker, just start a new instance of the
#   strategy and pick a different Trigger Symbol - no code changes needed.
# =============================================================================

from moomoo import *


class Strategy(StrategyBase):

    # -------------------------------------------------------------------------
    # Initialization (called once when the strategy starts)
    # -------------------------------------------------------------------------
    def initialize(self):
        declare_strategy_type(AlgoStrategyType.SECURITY)  # stock strategy
        self.trigger_symbols()
        self.custom_indicator()
        self.global_variables()

    def trigger_symbols(self):
        # Dynamic ticker: bound to the "Trigger Symbol" chosen in the Moomoo UI.
        # Pick a US ticker there (e.g. US.AAPL). Do not hard-code it here.
        self.trig_symbol_1 = declare_trig_symbol()

    def custom_indicator(self):
        # Only built-in ema() / rsi() are used, nothing to register.
        pass

    # -------------------------------------------------------------------------
    # User-tunable parameters (editable in the Moomoo UI via show_variable)
    # -------------------------------------------------------------------------
    def global_variables(self):
        # Indicator settings
        self.ema_fast_period = show_variable(9, GlobalType.INT)
        self.ema_slow_period = show_variable(21, GlobalType.INT)
        self.rsi_period = show_variable(14, GlobalType.INT)

        # RSI thresholds
        self.rsi_buy_max = show_variable(60.0, GlobalType.FLOAT)     # buy only if RSI below this
        self.rsi_overbought = show_variable(70.0, GlobalType.FLOAT)  # sell if RSI above this

        # Position sizing
        self.buy_pct_of_max = show_variable(100.0, GlobalType.FLOAT)   # % of max affordable qty to buy (1-100)
        self.max_shares_per_order = show_variable(0, GlobalType.INT)   # hard cap per order, 0 = no cap
        self.sell_pct_of_position = show_variable(100.0, GlobalType.FLOAT)  # % of sellable qty to exit (1-100)

        # Cash guard
        self.min_cash_reserve = show_variable(0.0, GlobalType.FLOAT)   # USD to keep untouched

        # Fixed internal settings
        self.bar_type = BarType.M15

    # -------------------------------------------------------------------------
    # Main loop (called by the engine on every update of the trigger symbol)
    # -------------------------------------------------------------------------
    def handle_data(self):
        symbol = self.trig_symbol_1

        # ---- Guard: parameter sanity --------------------------------------
        if self.ema_fast_period <= 0 or self.ema_slow_period <= 0 or self.rsi_period <= 0:
            print("Invalid indicator period(s); all periods must be > 0. Skipping.")
            return
        if self.ema_fast_period >= self.ema_slow_period:
            print("EMA fast period must be smaller than EMA slow period. Skipping.")
            return

        # ---- Indicators on completed 15-min bars ---------------------------
        # select=2 -> last completed bar, select=3 -> the bar before it
        fast_now = ema(symbol, self.ema_fast_period, self.bar_type, 2)
        fast_prev = ema(symbol, self.ema_fast_period, self.bar_type, 3)
        slow_now = ema(symbol, self.ema_slow_period, self.bar_type, 2)
        slow_prev = ema(symbol, self.ema_slow_period, self.bar_type, 3)
        rsi_now = rsi(symbol, self.rsi_period, self.bar_type, 2)

        # ---- Guard: indicator warm-up / missing data -----------------------
        values = (fast_now, fast_prev, slow_now, slow_prev, rsi_now)
        if any(v is None for v in values) or slow_now == 0 or slow_prev == 0:
            return  # not enough history yet

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        holding_qty = position_holding_qty(symbol)
        if holding_qty is None:
            holding_qty = 0

        # ---- SELL logic (checked first, only if we hold shares) ------------
        if holding_qty > 0 and (crossed_down or rsi_now > self.rsi_overbought):
            reason = "EMA cross down" if crossed_down else "RSI overbought"
            self.execute_sell(symbol, reason, rsi_now)
            return

        # ---- BUY logic (only if flat, to avoid stacking on repeat ticks) ---
        if holding_qty == 0 and crossed_up and rsi_now < self.rsi_buy_max:
            self.execute_buy(symbol, rsi_now)

    # -------------------------------------------------------------------------
    # Order helpers with zero-quantity and cash guards
    # -------------------------------------------------------------------------
    def execute_buy(self, symbol, rsi_now):
        cash = total_cash(Currency.USD)
        if cash is None or cash <= 0:
            print("BUY skipped: no USD cash available.")
            return

        spendable = cash - self.min_cash_reserve
        if spendable <= 0:
            print("BUY skipped: cash %.2f is at/below reserve %.2f." % (cash, self.min_cash_reserve))
            return

        price = current_price(symbol)
        if price is None or price <= 0:
            print("BUY skipped: invalid current price.")
            return

        # Max shares the cash balance supports for a market order
        max_qty = max_qty_to_buy_on_cash(symbol, OrdType.MKT)
        if max_qty is None or max_qty <= 0:
            print("BUY skipped: max_qty_to_buy_on_cash returned 0 (insufficient cash).")
            return

        # Scale by the user's % setting
        pct = min(max(self.buy_pct_of_max, 0.0), 100.0)
        qty = int(max_qty * pct / 100.0)

        # Respect the cash reserve
        qty = min(qty, int(spendable // price))

        # Respect the optional per-order hard cap
        if self.max_shares_per_order > 0:
            qty = min(qty, self.max_shares_per_order)

        if qty <= 0:
            print("BUY skipped: computed quantity is 0 (price %.2f, spendable %.2f)." % (price, spendable))
            return

        place_market(symbol, qty, OrderSide.BUY, TimeInForce.DAY)
        print("BUY %d %s @ ~%.2f | RSI %.2f | EMA%d crossed above EMA%d"
              % (qty, symbol, price, rsi_now, self.ema_fast_period, self.ema_slow_period))

    def execute_sell(self, symbol, reason, rsi_now):
        sellable = max_qty_to_sell(symbol)
        if sellable is None or sellable <= 0:
            print("SELL skipped: no sellable quantity.")
            return

        pct = min(max(self.sell_pct_of_position, 0.0), 100.0)
        qty = int(sellable * pct / 100.0)
        if qty <= 0:
            print("SELL skipped: computed quantity is 0.")
            return

        place_market(symbol, qty, OrderSide.SELL, TimeInForce.DAY)
        print("SELL %d %s | reason: %s | RSI %.2f" % (qty, symbol, reason, rsi_now))
