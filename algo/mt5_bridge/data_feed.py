"""
MT5DataFeed — historical OHLCV + fundamental data for AMMRO.

OHLCV  : fetched from MT5 terminal (real-time broker feed)
Fundamentals: fetched from yfinance (ROE, D/E — no MT5 fundamental API)
"""
from __future__ import annotations

import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from .symbol_map import SymbolMap


class MT5DataFeed:
    """
    Provides OHLCV history and fundamental data to the local AMMRO runner.

    All history methods return pandas DataFrames with a DatetimeIndex and
    columns [open, high, low, close, volume] — same shape as LEAN history().
    """

    def __init__(self, mt5, symbol_map: SymbolMap):
        self._mt5 = mt5
        self._sym = symbol_map
        self._fund_cache: dict[str, dict] = {}   # ticker → fundamental dict

    # ── OHLCV history ─────────────────────────────────────────────────────────

    def history(self, ticker: str, periods: int,
                resolution: str = "Daily") -> pd.DataFrame | None:
        """
        Fetch OHLCV history from MT5.

        Args:
            ticker    : strategy ticker (e.g. "AAPL")
            periods   : number of bars
            resolution: "Daily", "Hour", "Minute"

        Returns:
            DataFrame with DatetimeIndex, columns [open, high, low, close, volume]
            or None on error.
        """
        mt5_sym = self._sym.to_mt5(ticker)
        tf      = SymbolMap.lean_resolution_to_mt5_tf(resolution, self._mt5)

        try:
            rates = self._mt5.copy_rates_from_pos(mt5_sym, tf, 0, periods)
        except Exception as e:
            logger.warning(f"MT5 history error {mt5_sym}: {e}")
            return None

        if rates is None or len(rates) == 0:
            logger.debug(f"No data for {mt5_sym} ({periods} bars, {resolution})")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").rename(columns={
            "open":  "open",
            "high":  "high",
            "low":   "low",
            "close": "close",
            "tick_volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        if len(df) < int(periods * 0.8):
            logger.debug(f"Insufficient bars for {mt5_sym}: got {len(df)}/{periods}")
            return None

        return df

    def history_batch(self, tickers: list[str], periods: int,
                      resolution: str = "Daily") -> dict[str, pd.DataFrame]:
        """Fetch history for multiple tickers. Returns dict[ticker → DataFrame]."""
        result = {}
        for ticker in tickers:
            df = self.history(ticker, periods, resolution)
            if df is not None:
                result[ticker] = df
        return result

    def current_price(self, ticker: str) -> float | None:
        """Live bid/ask midpoint from MT5."""
        mt5_sym = self._sym.to_mt5(ticker)
        try:
            tick = self._mt5.symbol_info_tick(mt5_sym)
            if tick is None:
                return None
            return (tick.bid + tick.ask) / 2.0
        except Exception as e:
            logger.warning(f"Price error {mt5_sym}: {e}")
            return None

    def spread_pct(self, ticker: str) -> float:
        """Bid/ask spread as a fraction of mid-price. Returns 1.0 on error."""
        mt5_sym = self._sym.to_mt5(ticker)
        try:
            tick = self._mt5.symbol_info_tick(mt5_sym)
            if tick is None or tick.ask == 0:
                return 1.0
            mid = (tick.bid + tick.ask) / 2.0
            return (tick.ask - tick.bid) / max(mid, 0.01)
        except Exception:
            return 1.0

    def lot_size(self, ticker: str) -> float:
        """Minimum lot size / contract size for the symbol."""
        mt5_sym = self._sym.to_mt5(ticker)
        try:
            info = self._mt5.symbol_info(mt5_sym)
            return info.volume_min if info else 1.0
        except Exception:
            return 1.0

    # ── Fundamental data (via yfinance) ───────────────────────────────────────

    def fundamentals(self, ticker: str) -> dict:
        """
        Return fundamental data for ticker from yfinance.
        Cached — expensive, only called during monthly rebalance.

        Returns dict with keys: roe, debt_to_equity, market_cap, sector
        """
        if ticker in self._fund_cache:
            return self._fund_cache[ticker]

        data = {
            "roe":            0.0,
            "debt_to_equity": 0.0,
            "market_cap":     0.0,
            "sector":         "Unknown",
        }
        try:
            info = yf.Ticker(ticker).info
            data["roe"]            = float(info.get("returnOnEquity") or 0.0)
            data["debt_to_equity"] = float(info.get("debtToEquity")   or 0.0) / 100.0
            data["market_cap"]     = float(info.get("marketCap")      or 0.0)
            data["sector"]         = info.get("sector", "Unknown")
        except Exception as e:
            logger.debug(f"yfinance fundamentals error {ticker}: {e}")

        self._fund_cache[ticker] = data
        return data

    def clear_fundamental_cache(self) -> None:
        """Call monthly — forces fresh fundamental data on next rebalance."""
        self._fund_cache.clear()

    # ── VIX level ─────────────────────────────────────────────────────────────

    def vix_level(self) -> float:
        """Latest VIX close. Falls back to yfinance if MT5 VIX feed unavailable."""
        # Try MT5 first
        price = self.current_price("VIX")
        if price and price > 0:
            return price

        # Fallback: yfinance
        try:
            df = yf.download("^VIX", period="2d", auto_adjust=True, progress=False)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 20.0   # safe default
