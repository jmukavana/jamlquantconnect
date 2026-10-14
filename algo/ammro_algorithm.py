# region imports

import numpy as np
import pandas as pd

from stubs.AlgorithmImports import *


# endregion


class AMMROAlgorithm(QCAlgorithm):
    """
    AMMRO — Adaptive Multi-Factor Momentum with Risk Overlay
    ─────────────────────────────────────────────────────────
    Equity core  : Momentum (60%) + Quality (30%) + RSI gate (10%)
                   Inverse-vol weighted, top-25, monthly rebalance
    Crypto sat.  : BTC perpetual funding-rate carry (Binance)
    Optional     : Supply & Demand zone entries + Wyckoff phase filter
    Risk overlay : Per-position stops + portfolio circuit breaker
    """

    # ── Universe ──────────────────────────────────────────────────────────────
    UNIVERSE_SIZE       = 500
    MIN_PRICE           = 10.0
    MIN_DOLLAR_VOL      = 10_000_000
    MIN_MARKET_CAP      = 1_000_000_000

    # ── Alpha blending ────────────────────────────────────────────────────────
    MOMENTUM_LOOKBACK   = 252       # trading days (~12 months)
    MOMENTUM_SKIP       = 22        # exclude most-recent month
    QUALITY_DE_PENALTY  = 0.1       # debt/equity penalty coefficient
    RSI_PERIOD          = 14
    RSI_IDEAL_LO        = 40
    RSI_IDEAL_HI        = 65
    W_MOMENTUM          = 0.60
    W_QUALITY           = 0.30
    W_RSI               = 0.10

    # ── Portfolio construction ────────────────────────────────────────────────
    TOP_N               = 25
    EQUITY_SLEEVE       = 0.85
    CRYPTO_SLEEVE       = 0.10
    VOL_LOOKBACK        = 60
    POSITION_CAP        = 0.15
    SECTOR_CAP          = 0.30

    # ── Risk thresholds ───────────────────────────────────────────────────────
    HARD_STOP           = 0.93      # liquidate if price < entry × 0.93 (−7%)
    TRAILING_STOP       = 0.90      # liquidate if price < peak  × 0.90 (−10%)
    TAKE_PROFIT         = 0.25      # trim 50% if unrealised gain > 25%
    DEFENSIVE_DD        = 0.08      # halve weights at next rebalance
    HALT_DD             = 0.12      # liquidate everything, halt trading
    RECOVERY_DD         = 0.05      # allow resume when DD recovers below 5%
    VIX_RECOVERY        = 25.0      # also require VIX < 25 to resume
    MARGIN_EMERGENCY    = 0.05      # liquidate crypto if margin < 5% of PV

    # ── Crypto ────────────────────────────────────────────────────────────────
    FUNDING_HI          = 0.0001    # > 0.01% per 8hr → go SHORT
    FUNDING_LO          = -0.00005  # < −0.005% per 8hr → go LONG
    FUNDING_MIN_PERIODS = 10

    # ── Supply & Demand ───────────────────────────────────────────────────────
    ENABLE_SD           = True
    SD_LOOKBACK         = 120
    SD_RISK_PCT         = 0.01
    SD_RR_RATIO         = 2.5
    SD_MAX_POSITIONS    = 5

    # ── Wyckoff ───────────────────────────────────────────────────────────────
    ENABLE_WYCKOFF      = True
    WY_LOOKBACK         = 150
    WY_RISK_PCT         = 0.01

    # ═════════════════════════════════════════════════════════════════════════
    # INITIALIZE
    # ═════════════════════════════════════════════════════════════════════════

    def initialize(self) -> None:
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)

        self.set_brokerage_model(
            BrokerageModel.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.free_portfolio_value_percentage = 0.05
        self.set_benchmark("SPY")

        # Universe
        self.universe_settings.resolution   = Resolution.DAILY
        self.universe_settings.asynchronous = True
        self.add_universe(self._fundamental_filter)

        # Regime instruments
        self._spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._vix = self.add_index("VIX", Resolution.DAILY).symbol

        # Crypto satellite
        btc                = self.add_crypto_future("BTCUSDT", Resolution.DAILY, Market.BINANCE)
        self._btc          = btc.symbol
        self._btc_lot_size = btc.symbol_properties.lot_size
        self._funding_min  = Minimum(self.FUNDING_MIN_PERIODS)

        # Per-security indicator stores
        self._rsi = {}
        self._atr = {}

        # Position tracking state
        self._entry_price = {}
        self._peak_price  = {}
        self._entry_time  = {}
        self._tp_taken    = set()

        # Portfolio risk state
        self._peak_value      = self.portfolio.total_portfolio_value
        self._trading_halted  = False
        self._defensive_mode  = False
        self._state_restored  = False

        # Score cache (rebuilt monthly)
        self._score_cache      = {}
        self._score_cache_month = None

        # S&D tracking
        self._sd_targets  = {}   # symbol → float target price
        self._sd_stops    = {}   # symbol → stop order id

        # Wyckoff tracking
        self._wy_targets  = {}
        self._wy_stops    = {}

        # Audit trail
        self._audit = []

        self.set_warm_up(280, Resolution.DAILY)

        # Schedules
        self.schedule.on(
            self.date_rules.month_start(self._spy),
            self.time_rules.after_market_open(self._spy, 30),
            self._monthly_rebalance)

        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.after_market_open(self._spy, 15),
            self._daily_risk_check)

        if self.ENABLE_SD:
            self.schedule.on(
                self.date_rules.every_day(self._spy),
                self.time_rules.after_market_open(self._spy, 60),
                self._scan_sd_signals)

        if self.ENABLE_WYCKOFF:
            self.schedule.on(
                self.date_rules.week_start(self._spy),
                self.time_rules.after_market_open(self._spy, 90),
                self._scan_wyckoff_signals)

        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.before_market_close(self._spy, 5),
            self._log_state)

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 0 — UNIVERSE SELECTION
    # ═════════════════════════════════════════════════════════════════════════

    def _fundamental_filter(self, fundamental):
        filtered = [
            f for f in fundamental
            if f.has_fundamental_data
            and f.price > self.MIN_PRICE
            and f.dollar_volume > self.MIN_DOLLAR_VOL
            and f.market_cap > self.MIN_MARKET_CAP
            and f.asset_classification.morningstar_sector_code
                != MorningstarSectorCode.FINANCIAL_SERVICES
        ]
        return [
            f.symbol for f in sorted(
                filtered, key=lambda x: x.dollar_volume, reverse=True
            )[:self.UNIVERSE_SIZE]
        ]

    def on_securities_changed(self, changes) -> None:
        try:
            for sec in changes.added_securities:
                s = sec.symbol
                if s.security_type == SecurityType.EQUITY and s != self._spy:
                    self._rsi[s] = self.rsi(s, self.RSI_PERIOD, Resolution.DAILY)
                    self._atr[s] = self.atr(s, 14, Resolution.DAILY)

            for sec in changes.removed_securities:
                s = sec.symbol
                for store in (self._rsi, self._atr):
                    if s in store:
                        self.deregister_indicator(store.pop(s))
                if self.portfolio[s].invested:
                    self.liquidate(s, tag="exit-universe-removal")
                self._clear_position_state(s)

            self.log(f"Universe change: +{len(changes.added_securities)} "
                     f"−{len(changes.removed_securities)}")
        except Exception as e:
            self.log(f"on_securities_changed error: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 0 — DATA HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _safe_history(self, symbol_or_list, periods, resolution):
        try:
            bars = self.history(symbol_or_list, periods, resolution)
            if bars is None or (hasattr(bars, 'empty') and bars.empty):
                return None
            return bars
        except Exception as e:
            self.log(f"History error ({symbol_or_list}): {e}")
            return None

    def _safe_rsi(self, symbol):
        ind = self._rsi.get(symbol)
        return ind.current.value if (ind and ind.is_ready) else None

    def _safe_price(self, symbol):
        sec = self.securities.get(symbol)
        if sec is None:
            return None
        p = sec.price
        return p if p > 0 else None

    def _record_entry(self, symbol, fill_price) -> None:
        self._entry_price[symbol] = fill_price
        self._peak_price[symbol]  = fill_price
        self._entry_time[symbol]  = self.time
        self._tp_taken.discard(symbol)

    def _clear_position_state(self, symbol) -> None:
        self._entry_price.pop(symbol, None)
        self._peak_price.pop(symbol, None)
        self._entry_time.pop(symbol, None)
        self._tp_taken.discard(symbol)
        self._sd_targets.pop(symbol, None)
        self._sd_stops.pop(symbol, None)
        self._wy_targets.pop(symbol, None)
        self._wy_stops.pop(symbol, None)

    def _audit_event(self, event_type, details) -> None:
        self._audit.append({"time": str(self.time), "type": event_type, **details})
        if len(self._audit) > 500:
            self._audit = self._audit[-500:]

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2 — ALPHA: SIGNAL GENERATION
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(raw: dict) -> dict:
        if not raw:
            return {}
        vals = list(raw.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        return {k: (v - lo) / span for k, v in raw.items()}

    def _compute_scores(self) -> dict:
        """Batch momentum + quality + RSI scores for all active equities."""
        equity_syms = [
            s for s in self.active_securities.keys()
            if s.security_type == SecurityType.EQUITY and s != self._spy
        ]
        if not equity_syms:
            return {}

        # Batch history: one round-trip for all symbols
        hist = self._safe_history(equity_syms, self.MOMENTUM_LOOKBACK + self.MOMENTUM_SKIP + 5,
                                   Resolution.DAILY)
        if hist is None:
            return {}

        momentum_raw = {}
        quality_raw  = {}
        rsi_scores   = {}

        for symbol in equity_syms:
            try:
                sym_data = hist.loc[symbol]
            except KeyError:
                continue

            if len(sym_data) < self.MOMENTUM_LOOKBACK:
                continue

            closes = sym_data["close"].values

            # Momentum: 12-month return, skip most recent month
            momentum_raw[symbol] = closes[-self.MOMENTUM_SKIP] / closes[0] - 1

            # Quality: ROE minus debt penalty
            try:
                op  = self.securities[symbol].fundamentals.operation_ratios
                roe = op.roe.one_year or 0.0
                de  = op.long_term_debt_equity_ratio.value or 0.0
                quality_raw[symbol] = roe - max(0.0, de - 1.0) * self.QUALITY_DE_PENALTY
            except Exception:
                quality_raw[symbol] = 0.0

            # RSI gate
            rsi = self._safe_rsi(symbol)
            if rsi is None:
                rsi_scores[symbol] = 0.5
            elif self.RSI_IDEAL_LO <= rsi <= self.RSI_IDEAL_HI:
                rsi_scores[symbol] = 1.0
            elif rsi < self.RSI_IDEAL_LO:
                rsi_scores[symbol] = 0.5
            else:
                rsi_scores[symbol] = 0.0   # overbought gate

        if not momentum_raw:
            return {}

        m = self._normalize(momentum_raw)
        q = self._normalize({k: quality_raw.get(k, 0.0) for k in m})

        return {
            s: self.W_MOMENTUM * m[s]
               + self.W_QUALITY  * q.get(s, 0.0)
               + self.W_RSI      * rsi_scores.get(s, 0.5)
            for s in m
        }

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 3 — PORTFOLIO CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════════

    def _get_sector(self, symbol):
        try:
            return (self.securities[symbol].fundamentals
                        .asset_classification.morningstar_sector_code)
        except Exception:
            return -1

    def _apply_sector_caps(self, weights: dict) -> dict:
        sector_totals = {}
        for symbol, w in weights.items():
            sec = self._get_sector(symbol)
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w

        adjusted = dict(weights)
        for sector, total in sector_totals.items():
            if total > self.SECTOR_CAP:
                scale = self.SECTOR_CAP / total
                for symbol in adjusted:
                    if self._get_sector(symbol) == sector:
                        adjusted[symbol] *= scale
        return adjusted

    def _build_weights(self, scores: dict, defensive: bool) -> dict:
        if not scores:
            return {}

        top_syms = sorted(scores, key=scores.get, reverse=True)[:self.TOP_N]

        # Batch vol history
        vol_hist = self._safe_history(top_syms, self.VOL_LOOKBACK + 1, Resolution.DAILY)
        vols = {}
        if vol_hist is not None:
            for symbol in top_syms:
                try:
                    sym_data = vol_hist.loc[symbol]
                    if len(sym_data) < 10:
                        continue
                    closes = sym_data["close"].values
                    rets   = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
                    vols[symbol] = float(np.std(rets)) if len(rets) > 0 else 1.0
                except Exception:
                    continue

        if not vols:
            # Equal weight fallback
            n = len(top_syms)
            vols = {s: 1.0 for s in top_syms}

        inv_vol_sum = sum(1.0 / max(v, 1e-9) for v in vols.values())
        weights = {
            s: min((1.0 / max(vols[s], 1e-9)) / inv_vol_sum * self.EQUITY_SLEEVE,
                   self.POSITION_CAP)
            for s in vols
        }

        weights = self._apply_sector_caps(weights)

        if defensive:
            weights = {s: w * 0.5 for s, w in weights.items()}

        return weights

    def _needs_rebalance(self, target_weights: dict) -> bool:
        pv = self.portfolio.total_portfolio_value
        if pv <= 0:
            return False
        for symbol, target in target_weights.items():
            if not self.portfolio[symbol].invested:
                continue
            current = self.portfolio[symbol].holdings_value / pv
            if abs(current - target) > 0.03:
                return True
        return False

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 4 — PRE-TRADE RISK GATE
    # ═════════════════════════════════════════════════════════════════════════

    def _pre_trade_gate(self, weights: dict) -> dict:
        if self._trading_halted or self.is_warming_up:
            self.log("PRE-TRADE GATE: blocked — halted or warming up")
            return {}

        pv = self.portfolio.total_portfolio_value
        drawdown = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0

        if drawdown >= self.HALT_DD:
            self.log(f"PRE-TRADE GATE: blocked — drawdown {drawdown:.1%}")
            return {}

        if drawdown >= self.DEFENSIVE_DD:
            weights = {s: w * 0.5 for s, w in weights.items()}
            self.log(f"PRE-TRADE GATE: weights halved — drawdown {drawdown:.1%}")

        return {s: min(w, self.POSITION_CAP) for s, w in weights.items()}

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 5 — EXECUTION
    # ═════════════════════════════════════════════════════════════════════════

    def _execute_rebalance(self, weights: dict) -> None:
        approved = self._pre_trade_gate(weights)
        if not approved:
            return
        targets = [PortfolioTarget(s, w) for s, w in approved.items()]
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self.log(f"Rebalance: {len(targets)} positions | "
                 f"defensive={self._defensive_mode} | "
                 f"sum_weight={sum(approved.values()):.2%}")

    def _monthly_rebalance(self) -> None:
        if self.is_warming_up:
            return
        if self._trading_halted:
            self.log("Monthly rebalance SKIPPED — trading halted")
            return

        self.log(f"=== MONTHLY REBALANCE {self.time.strftime('%Y-%m')} | "
                 f"PV=${self.portfolio.total_portfolio_value:,.0f} ===")

        # Refresh score cache
        if self._score_cache_month != self.time.month:
            self._score_cache       = self._compute_scores()
            self._score_cache_month = self.time.month

        if not self._score_cache:
            self.log("No scores computed — skipping rebalance")
            return

        weights = self._build_weights(self._score_cache, self._defensive_mode)
        self._execute_rebalance(weights)
        self._audit_event("rebalance", {
            "positions": len(weights),
            "defensive": self._defensive_mode,
        })

    def on_order_event(self, order_event) -> None:
        tag    = order_event.tag or ""
        symbol = order_event.symbol

        if order_event.status == OrderStatus.FILLED:
            self.log(f"FILL [{tag}] {symbol} "
                     f"qty={order_event.fill_quantity:+.4f} "
                     f"@ {order_event.fill_price:.4f} "
                     f"fee={order_event.order_fee}")

            if any(k in tag for k in ("entry", "rebalance", "crypto-funding-entry",
                                       "sd-demand-entry", "wyckoff")):
                if symbol not in self._entry_price:
                    self._record_entry(symbol, order_event.fill_price)

            if any(k in tag for k in ("stop", "exit", "halt", "neutral", "flip",
                                       "universe-removal")):
                self._clear_position_state(symbol)

        elif order_event.status == OrderStatus.INVALID:
            self.log(f"INVALID ORDER: {symbol} — {order_event.message}")

        elif order_event.status == OrderStatus.CANCELED:
            self.log(f"CANCELED [{tag}] {symbol}")

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 6 — ON_DATA: REAL-TIME STOPS
    # ═════════════════════════════════════════════════════════════════════════

    def on_data(self, slice) -> None:
        if self.is_warming_up:
            return

        # Restore position state after live restart (one-time)
        if not self._state_restored:
            self._restore_state_after_restart()
            self._state_restored = True

        pv = self.portfolio.total_portfolio_value
        self._peak_value = max(self._peak_value, pv)

        # Per-position trailing stop, hard stop, take-profit
        for symbol in list(self._peak_price.keys()):
            if not self.portfolio[symbol].invested:
                self._clear_position_state(symbol)
                continue

            price = self._safe_price(symbol)
            if price is None:
                continue

            self._peak_price[symbol] = max(self._peak_price[symbol], price)
            entry = self._entry_price.get(symbol, price)

            if price < self._peak_price[symbol] * self.TRAILING_STOP:
                self.liquidate(symbol, tag="trailing-stop-10pct")
                self._clear_position_state(symbol)
                continue

            if price < entry * self.HARD_STOP:
                self.liquidate(symbol, tag="stop-loss-7pct")
                self._clear_position_state(symbol)
                continue

            # S&D take-profit
            sd_target = self._sd_targets.get(symbol)
            if sd_target and price >= sd_target:
                self.liquidate(symbol, tag="sd-take-profit")
                stop_id = self._sd_stops.pop(symbol, None)
                if stop_id:
                    self.transactions.cancel_order(stop_id)
                self._clear_position_state(symbol)
                continue

            # Wyckoff take-profit
            wy_target = self._wy_targets.get(symbol)
            if wy_target and price >= wy_target:
                self.liquidate(symbol, tag="wyckoff-take-profit")
                self._clear_position_state(symbol)
                continue

            # Momentum partial take-profit (25% gain → trim to 50%)
            if (self.portfolio[symbol].unrealized_profit_percent > self.TAKE_PROFIT
                    and symbol not in self._tp_taken
                    and symbol not in self._sd_targets
                    and symbol not in self._wy_targets):
                current_w = self.portfolio[symbol].holdings_value / max(pv, 1.0)
                self.set_holdings(symbol, current_w * 0.5, tag="take-profit-25pct")
                self._tp_taken.add(symbol)

        self._handle_crypto(slice)

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 6 — CRYPTO SATELLITE
    # ═════════════════════════════════════════════════════════════════════════

    def _handle_crypto(self, slice) -> None:
        if self._trading_halted:
            return
        if not self.is_market_open(self._btc):
            return

        rate_obj = slice.margin_interest_rates.get(self._btc)
        if not rate_obj:
            return

        rate = rate_obj.interest_rate
        self._funding_min.update(self.time, rate)

        invested = self.portfolio[self._btc].invested
        is_short = self.portfolio[self._btc].is_short
        is_long  = self.portfolio[self._btc].is_long

        if rate > self.FUNDING_HI and not is_short:
            self._place_crypto_order(-self.CRYPTO_SLEEVE)

        elif rate < self.FUNDING_LO and not is_long:
            self._place_crypto_order(self.CRYPTO_SLEEVE)

        elif self.FUNDING_LO <= rate <= self.FUNDING_HI and invested:
            self.liquidate(self._btc, tag="crypto-funding-neutral")

    def _place_crypto_order(self, target_weight: float) -> None:
        price = self._safe_price(self._btc)
        if price is None:
            return

        pv           = self.portfolio.total_portfolio_value
        target_value = pv * abs(target_weight)
        quantity     = target_value / price
        quantity     = (quantity // self._btc_lot_size) * self._btc_lot_size

        if quantity < self._btc_lot_size:
            return

        # Liquidate opposite direction before opening new side
        if self.portfolio[self._btc].invested:
            self.liquidate(self._btc, tag="crypto-direction-flip")

        signed_qty = quantity if target_weight > 0 else -quantity
        direction  = "long" if target_weight > 0 else "short"
        self.market_order(self._btc, signed_qty,
                          tag=f"crypto-funding-entry-{direction}")
        self.log(f"Crypto {direction}: qty={signed_qty:.4f} @ {price:.2f} "
                 f"funding_rate={self._funding_min.current.value:.6f}")

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 6 — POST-TRADE RISK: CIRCUIT BREAKER
    # ═════════════════════════════════════════════════════════════════════════

    def _trigger_halt(self, drawdown: float) -> None:
        self.liquidate(tag="drawdown-halt-12pct")
        self._trading_halted = True
        self._defensive_mode = False
        self.log(f"!!! TRADING HALTED — drawdown={drawdown:.1%} breached "
                 f"{self.HALT_DD:.0%} at {self.time} !!!")
        self.notify.email(
            "jamlechmukavana9@gmail.com",
            "AMMRO ALERT: Trading Halted",
            f"Portfolio drawdown {drawdown:.1%} breached {self.HALT_DD:.0%} "
            f"at {self.time}. All positions liquidated. Manual review required.")

    def _attempt_recovery(self, drawdown: float, total: float) -> None:
        vix_bars = self._safe_history(self._vix, 2, Resolution.DAILY)
        if vix_bars is not None and not vix_bars.empty:
            try:
                vix_level = float(vix_bars["close"].iloc[-1])
            except Exception:
                vix_level = 30.0
        else:
            vix_level = 30.0

        if vix_level < self.VIX_RECOVERY:
            self._trading_halted = False
            self._peak_value     = total
            self.log(f"Trading RESUMED — drawdown={drawdown:.1%}, VIX={vix_level:.1f}")

    def _daily_risk_check(self) -> None:
        if self.is_warming_up:
            return

        pv       = self.portfolio.total_portfolio_value
        drawdown = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0

        if drawdown >= self.HALT_DD and not self._trading_halted:
            self._trigger_halt(drawdown)

        elif (drawdown >= self.DEFENSIVE_DD
              and not self._defensive_mode
              and not self._trading_halted):
            self._defensive_mode = True
            self.log(f"DEFENSIVE MODE ON — drawdown={drawdown:.1%}")

        elif self._trading_halted and drawdown < self.RECOVERY_DD:
            self._attempt_recovery(drawdown, pv)

        elif self._defensive_mode and drawdown < self.RECOVERY_DD:
            self._defensive_mode = False
            self.log(f"DEFENSIVE MODE OFF — drawdown={drawdown:.1%}")

        # Margin guard — crypto sleeve
        margin_pct = self.portfolio.margin_remaining / pv if pv > 0 else 1.0
        if margin_pct < self.MARGIN_EMERGENCY and self.portfolio[self._btc].invested:
            self.liquidate(self._btc, tag="margin-reduction-emergency")
            self.log(f"Crypto liquidated — margin_pct={margin_pct:.1%}")

    def _restore_state_after_restart(self) -> None:
        """Repopulate entry/peak price from live holdings on restart."""
        for symbol, holding in self.portfolio.items():
            if holding.invested and symbol not in self._entry_price:
                p = holding.average_price
                if p > 0:
                    self._entry_price[symbol] = p
                    price = self._safe_price(symbol) or p
                    self._peak_price[symbol]  = max(p, price)
                    self._entry_time[symbol]  = self.time

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2 — SUPPLY & DEMAND SIGNALS (optional)
    # ═════════════════════════════════════════════════════════════════════════

    def _identify_demand_zones(self, symbol) -> list:
        bars = self._safe_history(symbol, self.SD_LOOKBACK, Resolution.DAILY)
        if bars is None or bars.empty or len(bars) < 5:
            return []

        # Handle both single-symbol (DatetimeIndex) and multi-symbol (MultiIndex)
        if isinstance(bars.index, pd.MultiIndex):
            try:
                bars = bars.loc[symbol]
            except KeyError:
                return []

        closes  = bars["close"].values
        highs   = bars["high"].values
        lows    = bars["low"].values
        opens   = bars["open"].values
        volumes = bars["volume"].values
        bodies  = np.abs(closes - opens)
        avg_body = np.mean(bodies) if len(bodies) > 0 else 1.0

        zones = []
        for i in range(2, len(closes) - 2):
            if bodies[i] > avg_body * 0.5:
                continue  # not a base candle

            if i + 1 >= len(closes):
                continue

            impulse_pct  = (closes[i + 1] - opens[i + 1]) / max(opens[i + 1], 0.01)
            impulse_body = bodies[i + 1]

            if impulse_pct < 0.015 or impulse_body < avg_body * 1.5:
                continue

            zone_top    = max(opens[i], closes[i], highs[i])
            zone_bottom = min(opens[i], closes[i], lows[i])
            zone_height = zone_top - zone_bottom if zone_top > zone_bottom else 0.001

            if impulse_body < zone_height * 3:
                continue

            # Check freshness — price never re-entered since formation
            future_lows = lows[i + 2:]
            if any(l < zone_top for l in future_lows):
                continue

            zones.append({
                "top":     zone_top,
                "bottom":  zone_bottom,
                "impulse": impulse_pct,
                "age":     len(closes) - i,
            })

        zones.sort(key=lambda z: (-z["impulse"], z["age"]))
        return zones

    def _check_sd_entry(self, symbol) -> dict | None:
        price = self._safe_price(symbol)
        if price is None:
            return None

        zones = self._identify_demand_zones(symbol)
        for zone in zones:
            if zone["bottom"] * 0.99 <= price <= zone["top"] * 1.005:
                # Confirm bounce: last close above zone top
                bars = self._safe_history(symbol, 2, Resolution.DAILY)
                if bars is None or bars.empty:
                    return None
                if isinstance(bars.index, pd.MultiIndex):
                    try:
                        bars = bars.loc[symbol]
                    except KeyError:
                        return None
                last_close = bars["close"].values[-1] if len(bars) > 0 else 0.0
                if last_close <= zone["top"]:
                    continue

                risk   = zone["top"] - zone["bottom"]
                return {
                    "entry":  price,
                    "stop":   zone["bottom"] - risk * 0.15,
                    "target": zone["top"] + risk * self.SD_RR_RATIO,
                    "zone":   zone,
                }
        return None

    def _sd_position_size(self, entry: float, stop: float) -> int:
        pv  = self.portfolio.total_portfolio_value
        rps = abs(entry - stop)
        if rps <= 0:
            return 0
        shares = int(pv * self.SD_RISK_PCT / rps)
        max_sh = int(pv * 0.10 / max(entry, 0.01))
        return min(shares, max_sh)

    def _scan_sd_signals(self) -> None:
        if self.is_warming_up or self._trading_halted:
            return
        if sum(1 for s in self._sd_targets) >= self.SD_MAX_POSITIONS:
            return

        for symbol in list(self.active_securities.keys()):
            if symbol.security_type != SecurityType.EQUITY or symbol == self._spy:
                continue
            if self.portfolio[symbol].invested:
                continue
            if symbol in self._sd_targets:
                continue

            signal = self._check_sd_entry(symbol)
            if not signal:
                continue

            qty = self._sd_position_size(signal["entry"], signal["stop"])
            if qty <= 0:
                continue

            self.market_order(symbol, qty, tag="sd-demand-entry")
            stop_ticket = self.stop_market_order(
                symbol, -qty, signal["stop"], tag="sd-stop-loss")

            self._sd_targets[symbol] = signal["target"]
            self._sd_stops[symbol]   = stop_ticket.order_id
            self.log(f"S&D entry {symbol}: entry={signal['entry']:.2f} "
                     f"stop={signal['stop']:.2f} target={signal['target']:.2f}")

            if sum(1 for s in self._sd_targets) >= self.SD_MAX_POSITIONS:
                break

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2 — WYCKOFF PHASE SIGNALS (optional)
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_sc(self, closes, opens, lows, highs, volumes, lookback=60) -> bool:
        """Selling Climax: wide down bar, very high volume, closes off the low."""
        bodies   = np.abs(closes - opens)
        avg_vol  = np.mean(volumes[-lookback:])
        avg_body = np.mean(bodies[-lookback:])
        for i in range(max(0, len(closes) - 10), len(closes)):
            is_down      = closes[i] < opens[i]
            wide         = bodies[i] > avg_body * 2.0
            hi_vol       = volumes[i] > avg_vol * 2.0
            off_low      = (closes[i] - lows[i]) > bodies[i] * 0.3
            if is_down and wide and hi_vol and off_low:
                return True
        return False

    def _detect_spring(self, closes, lows, volumes, support) -> bool:
        """Price briefly dips below support then recovers by close."""
        avg_vol = np.mean(volumes)
        for i in range(max(0, len(closes) - 15), len(closes)):
            if lows[i] < support * 0.995 and closes[i] > support * 0.995:
                if volumes[i] < avg_vol * 1.5:
                    return True
        return False

    def _detect_sos(self, closes, opens, volumes, resistance, lookback=60) -> bool:
        """Sign of Strength: strong up bar on high volume, breaks resistance."""
        bodies   = np.abs(closes - opens)
        avg_vol  = np.mean(volumes[-lookback:])
        avg_body = np.mean(bodies[-lookback:])
        for i in range(max(0, len(closes) - 10), len(closes)):
            is_up    = closes[i] > opens[i]
            wide     = bodies[i] > avg_body * 1.5
            hi_vol   = volumes[i] > avg_vol * 1.5
            breaks_r = closes[i] > resistance
            if is_up and wide and hi_vol and breaks_r:
                return True
        return False

    def _detect_utad(self, closes, highs, volumes, resistance) -> bool:
        """Upthrust After Distribution: poke above resistance, close back below."""
        avg_vol = np.mean(volumes)
        for i in range(max(0, len(closes) - 15), len(closes)):
            if highs[i] > resistance * 1.005 and closes[i] < resistance:
                if volumes[i] > avg_vol * 0.8:
                    return True
        return False

    def _classify_wyckoff_phase(self, symbol) -> str:
        bars = self._safe_history(symbol, self.WY_LOOKBACK, Resolution.DAILY)
        if bars is None or bars.empty or len(bars) < 50:
            return "UNKNOWN"

        if isinstance(bars.index, pd.MultiIndex):
            try:
                bars = bars.loc[symbol]
            except KeyError:
                return "UNKNOWN"

        closes  = bars["close"].values
        opens   = bars["open"].values
        highs   = bars["high"].values
        lows    = bars["low"].values
        volumes = bars["volume"].values

        recent_hi = np.max(closes[-60:])
        recent_lo = np.min(closes[-60:])
        range_pct = (recent_hi - recent_lo) / max(recent_lo, 0.01)

        if range_pct < 0.05:
            return "TIGHT_RANGE"

        current     = closes[-1]
        pos          = (current - recent_lo) / max(recent_hi - recent_lo, 0.01)
        has_sc       = self._detect_sc(closes, opens, lows, highs, volumes)
        has_spring   = self._detect_spring(closes, lows, volumes, recent_lo)
        has_sos      = self._detect_sos(closes, opens, volumes, recent_hi)
        has_utad     = self._detect_utad(closes, highs, volumes, recent_hi)

        if has_sc and not has_sos:
            if has_spring:
                return "ACCUMULATION_SPRING"
            return "ACCUMULATION_LATE" if pos < 0.35 else "ACCUMULATION_EARLY"

        if has_sos:
            return "MARKUP" if pos > 0.85 else "ACCUMULATION_SOS"

        if has_utad:
            return "DISTRIBUTION_UTAD"

        return "TRENDING_OR_UNKNOWN"

    def _vsa_no_supply(self, bars) -> bool:
        """No Supply: recent down bar with narrow spread and low volume."""
        if isinstance(bars.index, pd.MultiIndex):
            return False
        closes  = bars["close"].values[-5:]
        opens   = bars["open"].values[-5:]
        volumes = bars["volume"].values[-5:]
        avg_vol  = np.mean(bars["volume"].values)
        avg_body = np.mean(np.abs(bars["close"].values - bars["open"].values))
        for i in range(len(closes)):
            is_down = closes[i] < opens[i]
            narrow  = abs(closes[i] - opens[i]) < avg_body * 0.5
            low_vol = volumes[i] < avg_vol * 0.7
            if is_down and narrow and low_vol:
                return True
        return False

    def _wyckoff_signal(self, symbol) -> dict | None:
        phase = self._classify_wyckoff_phase(symbol)
        if phase not in ("ACCUMULATION_SPRING", "ACCUMULATION_SOS",
                         "ACCUMULATION_LATE"):
            return None

        bars = self._safe_history(symbol, self.WY_LOOKBACK, Resolution.DAILY)
        if bars is None or bars.empty:
            return None
        if isinstance(bars.index, pd.MultiIndex):
            try:
                bars = bars.loc[symbol]
            except KeyError:
                return None

        if not self._vsa_no_supply(bars):
            return None

        closes    = bars["close"].values
        recent_hi = np.max(closes[-60:])
        recent_lo = np.min(closes[-60:])

        if phase == "ACCUMULATION_SPRING":
            stop   = recent_lo * 0.97
            target = recent_hi * 1.05
        elif phase == "ACCUMULATION_SOS":
            stop   = (recent_lo + recent_hi) / 2
            target = recent_hi * 1.10
        else:
            stop   = recent_lo * 0.97
            target = recent_hi

        return {"stop": stop, "target": target, "phase": phase}

    def _scan_wyckoff_signals(self) -> None:
        if self.is_warming_up or self._trading_halted:
            return

        for symbol in list(self.active_securities.keys()):
            if symbol.security_type != SecurityType.EQUITY or symbol == self._spy:
                continue
            if self.portfolio[symbol].invested:
                continue
            if symbol in self._wy_targets:
                continue

            signal = self._wyckoff_signal(symbol)
            if not signal:
                continue

            price = self._safe_price(symbol)
            if price is None:
                continue

            rps = abs(price - signal["stop"])
            if rps <= 0:
                continue

            pv   = self.portfolio.total_portfolio_value
            qty  = int(pv * self.WY_RISK_PCT / rps)
            qty  = min(qty, int(pv * 0.08 / max(price, 0.01)))
            if qty <= 0:
                continue

            self.market_order(symbol, qty,
                              tag=f"wyckoff-{signal['phase'].lower()}")
            self._wy_targets[symbol] = signal["target"]
            self._wy_stops[symbol]   = signal["stop"]
            self.log(f"Wyckoff {signal['phase']} {symbol}: "
                     f"entry={price:.2f} stop={signal['stop']:.2f} "
                     f"target={signal['target']:.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 7 — MONITORING
    # ═════════════════════════════════════════════════════════════════════════

    def _log_state(self) -> None:
        if self.is_warming_up:
            return
        pv        = self.portfolio.total_portfolio_value
        cash_pct  = self.portfolio.cash / max(pv, 1.0)
        drawdown  = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0
        n_pos     = sum(1 for h in self.portfolio.values() if h.invested)
        margin_p  = self.portfolio.margin_remaining / max(pv, 1.0)

        self.plot("Risk",      "DrawdownPct",  drawdown  * 100)
        self.plot("Risk",      "CashPct",      cash_pct  * 100)
        self.plot("Risk",      "MarginPct",    margin_p  * 100)
        self.plot("Alpha",     "Positions",    n_pos)
        self.plot("Portfolio", "TotalValue",   pv)
