"""
SymbolMap — translates between strategy ticker names and MT5 broker symbol names.

Different brokers name symbols differently:
  IC Markets : "AAPL"      Pepperstone : "AAPL.US"    OANDA : "AAPL_USD"
  Binance MT5: "BTCUSD"    IC Markets  : "BTCUSD"

Configure BROKER_PREFIX and EQUITY_SUFFIX to match your broker.
"""
from __future__ import annotations


# ── Default symbol map (extend for your broker) ───────────────────────────────
_DEFAULT_MAP: dict[str, str] = {
    # Crypto
    "BTCUSD":   "BTCUSD",
    "BTCUSDT":  "BTCUSD",
    "ETHUSD":   "ETHUSD",
    # Indices (for regime)
    "VIX":      "VIX.F",
    "SPY":      "SPY",
    # Common equities — add more as needed
    "AAPL":     "AAPL",
    "MSFT":     "MSFT",
    "GOOGL":    "GOOGL",
    "AMZN":     "AMZN",
    "NVDA":     "NVDA",
    "META":     "META",
    "TSLA":     "TSLA",
    "BRK.B":    "BRKB",
}


class SymbolMap:
    """
    Translates strategy tickers to broker-specific MT5 symbol names.

    Args:
        custom_map  : override/extend the default symbol map
        equity_suffix: broker suffix appended to all equity symbols (e.g. ".US")
    """

    def __init__(
        self,
        custom_map:    dict[str, str] | None = None,
        equity_suffix: str = "",
    ):
        self._map    = dict(_DEFAULT_MAP)
        self._suffix = equity_suffix
        if custom_map:
            self._map.update(custom_map)
        self._reverse: dict[str, str] = {v: k for k, v in self._map.items()}

    def to_mt5(self, ticker: str) -> str:
        """strategy ticker → MT5 symbol name."""
        if ticker in self._map:
            return self._map[ticker]
        # Apply broker suffix for unknown equities
        return ticker + self._suffix

    def from_mt5(self, mt5_symbol: str) -> str:
        """MT5 symbol name → strategy ticker (best-effort reverse lookup)."""
        return self._reverse.get(mt5_symbol, mt5_symbol.replace(self._suffix, ""))

    def register(self, ticker: str, mt5_symbol: str) -> None:
        """Add or override a mapping at runtime."""
        self._map[ticker]        = mt5_symbol
        self._reverse[mt5_symbol] = ticker

    # ── MT5 timeframe helpers ─────────────────────────────────────────────────

    @staticmethod
    def lean_resolution_to_mt5_tf(resolution: str, mt5):
        """Convert a LEAN Resolution string to an MT5 TIMEFRAME constant."""
        mapping = {
            "Daily":   mt5.TIMEFRAME_D1,
            "Hour":    mt5.TIMEFRAME_H1,
            "Minute":  mt5.TIMEFRAME_M1,
        }
        tf = mapping.get(resolution)
        if tf is None:
            raise ValueError(f"Unsupported resolution for MT5: {resolution}")
        return tf
