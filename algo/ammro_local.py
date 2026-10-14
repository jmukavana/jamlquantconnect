"""
ammro_local.py — AMMRO algorithm running locally against MetaTrader 5.

Entry point:
    uv run python algo/ammro_local.py

Configuration:
    Edit the CONFIG dict at the bottom of this file.
    Set MT5 account credentials and your broker's symbol suffix.

Architecture:
    LocalAMMRORunner implements the same algorithm logic as AMMROAlgorithm
    but replaces all LEAN API calls with MT5Bridge equivalents:
        self.history()      → data.history()
        self.market_order() → executor.market_order()
        self.portfolio[s]   → portfolio[s]
        self.log()          → logger.info()
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mt5_bridge import (
    MT5Connection,
    MT5DataFeed,
    MT5Executor,
    MT5Portfolio,
    Scheduler,
    SymbolMap,
)

# ── Default equity watchlist ──────────────────────────────────────────────────
# Replace or extend with symbols available on your MT5 broker.
# For a full S&P 500 universe, load from a CSV instead.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH",  "XOM",  "LLY",  "JPM",  "JNJ",   "V",    "PG",   "MA",
    "HD",   "CVX",  "MRK",  "ABBV", "COST",  "PEP",  "KO",   "ADBE",
    "WMT",  "CRM",  "ACN",  "BAC",  "MCD",   "TMO",  "NFLX", "AVGO",
    "ABT",  "CMCSA","AMD",  "DHR",  "TXN",   "NEE",  "PM",   "UPS",
    "RTX",  "QCOM", "HON",  "INTU", "SBUX",  "CAT",  "IBM",  "GS",
]


class LocalAMMRORunner:
    """
    AMMRO algorithm adapted for local execution via MT5.

    Same parameters and logic as AMMROAlgorithm.
    All LEAN-specific calls replaced with MT5Bridge equivalents.
    """

    # ── Parameters (mirrors AMMROAlgorithm class constants) ───────────────────
    MOMENTUM_LOOKBACK   = 252
    MOMENTUM_SKIP       = 22
    QUALITY_DE_PENALTY  = 0.1
    RSI_PERIOD          = 14
    RSI_IDEAL_LO        = 40
    RSI_IDEAL_HI        = 65
    W_MOMENTUM          = 0.60
    W_QUALITY           = 0.30
    W_RSI               = 0.10

    TOP_N               = 25
    EQUITY_SLEEVE       = 0.85
    CRYPTO_SLEEVE       = 0.10
    VOL_LOOKBACK        = 60
    POSITION_CAP        = 0.15
    SECTOR_CAP          = 0.30

    HARD_STOP           = 0.93
    TRAILING_STOP       = 0.90
    TAKE_PROFIT         = 0.25
    DEFENSIVE_DD        = 0.08
    HALT_DD             = 0.12
    RECOVERY_DD         = 0.05
    VIX_RECOVERY        = 25.0
    MARGIN_EMERGENCY    = 0.05

    FUNDING_HI          = 0.0001
    FUNDING_LO          = -0.00005

    ENABLE_SD           = False    # disable for initial live testing
    ENABLE_WYCKOFF      = False    # disable for initial live testing

    SD_LOOKBACK         = 120
    SD_RISK_PCT         = 0.01
    SD_RR_RATIO         = 2.5
    SD_MAX_POSITIONS    = 5

    WY_LOOKBACK         = 150
    WY_RISK_PCT         = 0.01

    def __init__(
        self,
        data:       MT5DataFeed,
        executor:   MT5Executor,
        portfolio:  MT5Portfolio,
        watchlist:  list[str] = DEFAULT_WATCHLIST,
        btc_ticker: str = "BTCUSD",
    ):
        self.data      = data
        self.executor  = executor
        self.portfolio = portfolio
        self.watchlist = watchlist
        self.btc       = btc_ticker

        # Position tracking state
        self._entry_price  : dict[str, float]         = {}
        self._peak_price   : dict[str, float]         = {}
        self._entry_time   : dict[str, datetime.datetime] = {}
        self._tp_taken     : set[str]                 = set()

        # Portfolio risk state
        self._peak_value      = self.portfolio.total_portfolio_value
        self._trading_halted  = False
        self._defensive_mode  = False

        # Score cache
        self._score_cache      : dict[str, float] = {}
        self._score_cache_month: int | None        = None

        # S&D tracking
        self._sd_targets: dict[str, float] = {}
        self._sd_stops  : dict[str, int]   = {}

        # Wyckoff tracking
        self._wy_targets: dict[str, float] = {}

        logger.info(f"LocalAMMRORunner initialised | watchlist={len(watchlist)} symbols")

    # ═════════════════════════════════════════════════════════════════════════
    # SCHEDULED HANDLERS  (called by Scheduler)
    # ═════════════════════════════════════════════════════════════════════════

    def monthly_rebalance(self) -> None:
        if self._trading_halted:
            logger.info("Monthly rebalance SKIPPED — trading halted")
            return

        now = datetime.datetime.now()
        logger.info(f"=== MONTHLY REBALANCE {now.strftime('%Y-%m')} | "
                    f"PV=${self.portfolio.total_portfolio_value:,.0f} ===")

        # Refresh score cache
        if self._score_cache_month != now.month:
            self.data.clear_fundamental_cache()
            self._score_cache       = self._compute_scores()
            self._score_cache_month = now.month

        if not self._score_cache:
            logger.warning("No scores — skipping rebalance")
            return

        weights = self._build_weights(self._score_cache, self._defensive_mode)
        self._execute_rebalance(weights)

    def daily_risk_check(self) -> None:
        pv       = self.portfolio.total_portfolio_value
        self._peak_value = max(self._peak_value, pv)
        drawdown = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0

        logger.debug(f"Daily risk | PV={pv:,.0f} DD={drawdown:.1%} "
                     f"halted={self._trading_halted} defensive={self._defensive_mode}")

        if drawdown >= self.HALT_DD and not self._trading_halted:
            self._trigger_halt(drawdown)

        elif (drawdown >= self.DEFENSIVE_DD
              and not self._defensive_mode
              and not self._trading_halted):
            self._defensive_mode = True
            logger.warning(f"DEFENSIVE MODE ON — drawdown={drawdown:.1%}")

        elif self._trading_halted and drawdown < self.RECOVERY_DD:
            vix = self.data.vix_level()
            if vix < self.VIX_RECOVERY:
                self._trading_halted = False
                self._peak_value = pv
                logger.info(f"Trading RESUMED — drawdown={drawdown:.1%}, VIX={vix:.1f}")

        elif self._defensive_mode and drawdown < self.RECOVERY_DD:
            self._defensive_mode = False
            logger.info(f"DEFENSIVE MODE OFF — drawdown={drawdown:.1%}")

        # Margin guard for crypto
        margin_pct = self.portfolio.margin_remaining / max(pv, 1.0)
        if margin_pct < self.MARGIN_EMERGENCY:
            logger.warning(f"Low margin {margin_pct:.1%} — liquidating crypto")
            self.executor.liquidate(self.btc, tag="margin-reduction-emergency")

    def on_tick(self) -> None:
        """
        Called every scheduler tick (default 60s).
        Checks per-position stops and crypto funding signal.
        """
        if self._trading_halted:
            return

        pv = self.portfolio.total_portfolio_value
        self._peak_value = max(self._peak_value, pv)

        # Per-position trailing stop / hard stop / take-profit
        for ticker in list(self._peak_price.keys()):
            holding = self.portfolio[ticker]
            if not holding.invested:
                self._clear_state(ticker)
                continue

            price = self.data.current_price(ticker)
            if price is None:
                continue

            self._peak_price[ticker] = max(self._peak_price[ticker], price)
            entry = self._entry_price.get(ticker, price)

            if price < self._peak_price[ticker] * self.TRAILING_STOP:
                logger.info(f"TRAILING STOP {ticker} @ {price:.4f}")
                self.executor.liquidate(ticker, tag="trailing-stop-10pct")
                self._clear_state(ticker)
                continue

            if price < entry * self.HARD_STOP:
                logger.info(f"HARD STOP {ticker} @ {price:.4f}")
                self.executor.liquidate(ticker, tag="stop-loss-7pct")
                self._clear_state(ticker)
                continue

            # S&D take-profit
            sd_target = self._sd_targets.get(ticker)
            if sd_target and price >= sd_target:
                logger.info(f"S&D TAKE PROFIT {ticker} @ {price:.4f}")
                self.executor.liquidate(ticker, tag="sd-take-profit")
                stop_id = self._sd_stops.pop(ticker, None)
                if stop_id:
                    self.executor.cancel_order(stop_id)
                self._clear_state(ticker)
                continue

            # Momentum partial take-profit
            if (holding.unrealized_profit_percent > self.TAKE_PROFIT
                    and ticker not in self._tp_taken
                    and ticker not in self._sd_targets):
                current_w = holding.holdings_value / max(pv, 1.0)
                target_w  = current_w * 0.5
                logger.info(f"TAKE PROFIT {ticker} — trimming to {target_w:.1%}")
                self.executor.set_holdings(
                    ticker, target_w, pv, price, tag="take-profit-25pct")
                self._tp_taken.add(ticker)

    def log_state(self) -> None:
        pv        = self.portfolio.total_portfolio_value
        cash_pct  = self.portfolio.cash / max(pv, 1.0)
        drawdown  = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0
        n_pos     = sum(1 for h in self.portfolio.values() if h.invested)
        margin_p  = self.portfolio.margin_remaining / max(pv, 1.0)

        logger.info(
            f"STATE | PV={pv:,.2f} | DD={drawdown:.1%} | Cash={cash_pct:.1%} | "
            f"Margin={margin_p:.1%} | Positions={n_pos} | "
            f"Halted={self._trading_halted} | Defensive={self._defensive_mode}")

    # ═════════════════════════════════════════════════════════════════════════
    # ALPHA LAYER
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(raw: dict) -> dict:
        if not raw:
            return {}
        vals = list(raw.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        return {k: (v - lo) / span for k, v in raw.items()}

    def _compute_scores(self) -> dict[str, float]:
        logger.info(f"Computing scores for {len(self.watchlist)} symbols ...")

        # Batch history (one call per symbol — yfinance fallback if MT5 insufficient)
        hist = self.data.history_batch(
            self.watchlist,
            self.MOMENTUM_LOOKBACK + self.MOMENTUM_SKIP + 5,
            resolution="Daily")

        momentum_raw: dict[str, float] = {}
        quality_raw : dict[str, float] = {}
        rsi_scores  : dict[str, float] = {}

        for ticker in self.watchlist:
            df = hist.get(ticker)
            if df is None or len(df) < self.MOMENTUM_LOOKBACK:
                continue

            closes = df["close"].values

            # Momentum: 12-month return, skip most-recent month
            momentum_raw[ticker] = closes[-self.MOMENTUM_SKIP] / closes[0] - 1

            # Quality from yfinance
            fund = self.data.fundamentals(ticker)
            roe  = fund.get("roe", 0.0)
            de   = fund.get("debt_to_equity", 0.0)
            quality_raw[ticker] = roe - max(0.0, de - 1.0) * self.QUALITY_DE_PENALTY

            # RSI (computed from closes — no LEAN indicator object available)
            rsi_val = self._compute_rsi(closes, self.RSI_PERIOD)
            if rsi_val is None:
                rsi_scores[ticker] = 0.5
            elif self.RSI_IDEAL_LO <= rsi_val <= self.RSI_IDEAL_HI:
                rsi_scores[ticker] = 1.0
            elif rsi_val < self.RSI_IDEAL_LO:
                rsi_scores[ticker] = 0.5
            else:
                rsi_scores[ticker] = 0.0

        if not momentum_raw:
            logger.warning("No momentum scores computed — watchlist may be empty")
            return {}

        m = self._normalize(momentum_raw)
        q = self._normalize({k: quality_raw.get(k, 0.0) for k in m})

        composite = {
            t: self.W_MOMENTUM * m[t]
               + self.W_QUALITY  * q.get(t, 0.0)
               + self.W_RSI      * rsi_scores.get(t, 0.5)
            for t in m
        }

        top5 = sorted(composite, key=composite.get, reverse=True)[:5]
        logger.info(f"Top 5 scores: {[(t, f'{composite[t]:.3f}') for t in top5]}")
        return composite

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
        """Wilder RSI computed from close prices."""
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = np.mean(gains[:period])
        avg_l  = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_g = (avg_g * (period - 1) + gains[i])  / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100.0 - 100.0 / (1.0 + rs)

    # ═════════════════════════════════════════════════════════════════════════
    # PORTFOLIO CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════════

    def _build_weights(self, scores: dict, defensive: bool) -> dict[str, float]:
        if not scores:
            return {}

        top_syms = sorted(scores, key=scores.get, reverse=True)[:self.TOP_N]

        # Vol weighting from 60-day history
        vol_hist = self.data.history_batch(top_syms, self.VOL_LOOKBACK + 1)
        vols: dict[str, float] = {}
        for ticker in top_syms:
            df = vol_hist.get(ticker)
            if df is None or len(df) < 10:
                continue
            closes = df["close"].values
            rets   = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
            vols[ticker] = float(np.std(rets)) if len(rets) > 0 else 1.0

        if not vols:
            vols = {t: 1.0 for t in top_syms}

        inv_sum = sum(1.0 / max(v, 1e-9) for v in vols.values())
        weights = {
            t: min((1.0 / max(vols[t], 1e-9)) / inv_sum * self.EQUITY_SLEEVE,
                   self.POSITION_CAP)
            for t in vols
        }

        # Sector cap
        weights = self._apply_sector_caps(weights)

        if defensive:
            weights = {t: w * 0.5 for t, w in weights.items()}

        return weights

    def _apply_sector_caps(self, weights: dict) -> dict:
        sector_totals: dict[str, float] = {}
        sectors: dict[str, str] = {}
        for ticker in weights:
            fund = self.data.fundamentals(ticker)
            sec  = fund.get("sector", "Unknown")
            sectors[ticker] = sec
            sector_totals[sec] = sector_totals.get(sec, 0.0) + weights[ticker]

        adjusted = dict(weights)
        for sector, total in sector_totals.items():
            if total > self.SECTOR_CAP:
                scale = self.SECTOR_CAP / total
                for ticker in adjusted:
                    if sectors.get(ticker) == sector:
                        adjusted[ticker] *= scale
        return adjusted

    # ═════════════════════════════════════════════════════════════════════════
    # PRE-TRADE GATE & EXECUTION
    # ═════════════════════════════════════════════════════════════════════════

    def _pre_trade_gate(self, weights: dict) -> dict:
        if self._trading_halted:
            logger.info("PRE-TRADE GATE: blocked — halted")
            return {}

        pv = self.portfolio.total_portfolio_value
        dd = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0

        if dd >= self.HALT_DD:
            logger.info(f"PRE-TRADE GATE: blocked — DD {dd:.1%}")
            return {}
        if dd >= self.DEFENSIVE_DD:
            weights = {t: w * 0.5 for t, w in weights.items()}

        return {t: min(w, self.POSITION_CAP) for t, w in weights.items()}

    def _execute_rebalance(self, weights: dict) -> None:
        approved = self._pre_trade_gate(weights)
        if not approved:
            return

        pv = self.portfolio.total_portfolio_value

        # Liquidate positions not in new target set
        for ticker, holding in self.portfolio.items():
            if holding.invested and ticker not in approved:
                logger.info(f"EXIT {ticker} (not in new weights)")
                self.executor.liquidate(ticker, tag="exit-rotation-monthly")
                self._clear_state(ticker)

        # Rebalance / enter new positions
        for ticker, target_w in approved.items():
            price = self.data.current_price(ticker)
            if price is None:
                continue

            # Skip if spread too wide (> 0.5%)
            if self.data.spread_pct(ticker) > 0.005:
                logger.debug(f"Skip {ticker} — spread too wide")
                continue

            result = self.executor.set_holdings(
                ticker, target_w, pv, price, tag="entry-rebalance-monthly")

            if result.success and result.volume > 0:
                if ticker not in self._entry_price:
                    self._record_entry(ticker, price)

        logger.info(f"Rebalance done: {len(approved)} targets | "
                    f"defensive={self._defensive_mode}")

    # ═════════════════════════════════════════════════════════════════════════
    # RISK: CIRCUIT BREAKER
    # ═════════════════════════════════════════════════════════════════════════

    def _trigger_halt(self, drawdown: float) -> None:
        self.executor.liquidate_all(tag="drawdown-halt-12pct")
        self._trading_halted = True
        self._defensive_mode = False
        logger.critical(
            f"!!! TRADING HALTED — drawdown={drawdown:.1%} breached {self.HALT_DD:.0%} !!!")

    # ═════════════════════════════════════════════════════════════════════════
    # STATE HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _record_entry(self, ticker: str, price: float) -> None:
        self._entry_price[ticker] = price
        self._peak_price[ticker]  = price
        self._entry_time[ticker]  = datetime.datetime.now()
        self._tp_taken.discard(ticker)

    def _clear_state(self, ticker: str) -> None:
        self._entry_price.pop(ticker, None)
        self._peak_price.pop(ticker, None)
        self._entry_time.pop(ticker, None)
        self._tp_taken.discard(ticker)
        self._sd_targets.pop(ticker, None)
        self._sd_stops.pop(ticker, None)
        self._wy_targets.pop(ticker, None)

    def status(self) -> dict:
        """Return a JSON-serialisable status snapshot."""
        pv = self.portfolio.total_portfolio_value
        return {
            "time":           datetime.datetime.now().isoformat(),
            "portfolio_value": round(pv, 2),
            "cash":           round(self.portfolio.cash, 2),
            "margin_free":    round(self.portfolio.margin_remaining, 2),
            "drawdown_pct":   round((self._peak_value - pv) / max(self._peak_value, 1) * 100, 2),
            "halted":         self._trading_halted,
            "defensive":      self._defensive_mode,
            "open_positions": [h.symbol for h in self.portfolio.values() if h.invested],
            "score_cache_size": len(self._score_cache),
        }


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── MT5 connection ────────────────────────────────────────────────────────
    "mt5_account":  0,              # your MT5 account number
    "mt5_password": "",             # your MT5 password
    "mt5_server":   "",             # broker server name (e.g. "ICMarkets-Demo01")
    "mt5_host":     "127.0.0.1",   # Linux only (mt5linux rpyc host)
    "mt5_port":     18812,          # Linux only (mt5linux rpyc port)

    # ── Broker symbol config ──────────────────────────────────────────────────
    "equity_suffix": "",            # e.g. ".US" for Pepperstone, "" for ICMarkets
    "btc_ticker":    "BTCUSD",

    # ── Watchlist ─────────────────────────────────────────────────────────────
    "watchlist": DEFAULT_WATCHLIST,

    # ── Scheduler ─────────────────────────────────────────────────────────────
    "tick_interval_sec": 60,        # on_tick fires every N seconds
}


def main():
    logger.add("logs/ammro_local_{time}.log", rotation="1 day", retention="30 days")

    conn = MT5Connection(
        account  = CONFIG["mt5_account"],
        password = CONFIG["mt5_password"],
        server   = CONFIG["mt5_server"],
        host     = CONFIG["mt5_host"],
        port     = CONFIG["mt5_port"],
    )

    with conn:
        mt5 = conn.mt5
        sym_map   = SymbolMap(equity_suffix=CONFIG["equity_suffix"])
        data      = MT5DataFeed(mt5, sym_map)
        executor  = MT5Executor(mt5, sym_map)
        portfolio = MT5Portfolio(mt5, sym_map)

        runner = LocalAMMRORunner(
            data      = data,
            executor  = executor,
            portfolio = portfolio,
            watchlist = CONFIG["watchlist"],
            btc_ticker= CONFIG["btc_ticker"],
        )

        scheduler = Scheduler()
        scheduler.on_month_start(runner.monthly_rebalance, offset_min=30)
        scheduler.on_every_day(runner.daily_risk_check,    offset_min=15)
        scheduler.before_market_close(runner.log_state,    offset_min=5)

        logger.info("Starting AMMRO local runner ...")
        logger.info(f"Status: {json.dumps(runner.status(), indent=2)}")

        scheduler.run(
            tick_fn           = runner.on_tick,
            tick_interval_sec = CONFIG["tick_interval_sec"],
        )


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    main()
