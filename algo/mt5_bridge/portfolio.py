"""
MT5Portfolio — reads account state and open positions from MT5.
Mirrors the LEAN portfolio API used by AMMRO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Holding:
    """Mirrors LEAN's PortfolioHolding for a single position."""
    symbol:                    str
    quantity:                  float = 0.0
    average_price:             float = 0.0
    market_price:              float = 0.0
    holdings_value:            float = 0.0
    unrealized_profit:         float = 0.0
    unrealized_profit_percent: float = 0.0
    is_long:                   bool  = False
    is_short:                  bool  = False

    @property
    def invested(self) -> bool:
        return abs(self.quantity) > 0.0


class MT5Portfolio:
    """
    Reads live account data and positions from MT5.

    Provides the same attributes/methods that AMMRO uses on self.portfolio:
        total_portfolio_value, cash, margin_remaining
        portfolio[symbol].invested / .holdings_value / .unrealized_profit_percent
    """

    MAGIC = 20250503   # EA magic number — all AMMRO orders tagged with this

    def __init__(self, mt5, symbol_map):
        self._mt5 = mt5
        self._sym = symbol_map

    # ── Account state ─────────────────────────────────────────────────────────

    @property
    def total_portfolio_value(self) -> float:
        info = self._account_info()
        return float(info.equity) if info else 0.0

    @property
    def cash(self) -> float:
        info = self._account_info()
        return float(info.balance) if info else 0.0

    @property
    def margin_remaining(self) -> float:
        info = self._account_info()
        return float(info.margin_free) if info else 0.0

    @property
    def currency(self) -> str:
        info = self._account_info()
        return info.currency if info else "USD"

    # ── Position access ───────────────────────────────────────────────────────

    def __getitem__(self, ticker: str) -> Holding:
        """Returns a Holding for ticker (strategy name), or empty Holding if flat."""
        mt5_sym  = self._sym.to_mt5(ticker)
        position = self._get_position(mt5_sym)
        if position is None:
            return Holding(symbol=ticker)

        price = float(position.price_current)
        avg   = float(position.price_open)
        qty   = float(position.volume)
        value = qty * price
        pnl   = float(position.profit)
        pnl_p = pnl / (qty * avg) if qty * avg != 0 else 0.0

        return Holding(
            symbol                    = ticker,
            quantity                  = qty if position.type == 0 else -qty,
            average_price             = avg,
            market_price              = price,
            holdings_value            = value,
            unrealized_profit         = pnl,
            unrealized_profit_percent = pnl_p,
            is_long                   = position.type == 0,
            is_short                  = position.type == 1,
        )

    def values(self):
        """Iterate all open positions — mirrors LEAN portfolio.values()."""
        try:
            positions = self._mt5.positions_get()
        except Exception:
            return []
        if not positions:
            return []
        return [self[self._sym.from_mt5(p.symbol)] for p in positions]

    def items(self):
        """Iterate (ticker, Holding) pairs for all open positions."""
        try:
            positions = self._mt5.positions_get()
        except Exception:
            return []
        if not positions:
            return []
        return [
            (self._sym.from_mt5(p.symbol), self[self._sym.from_mt5(p.symbol)])
            for p in positions
        ]

    def invested_symbols(self) -> list[str]:
        """Return list of tickers with open positions."""
        return [ticker for ticker, h in self.items() if h.invested]

    def summary(self) -> str:
        info = self._account_info()
        if info is None:
            return "MT5 account info unavailable"
        return (f"Balance={info.balance:.2f} | Equity={info.equity:.2f} | "
                f"Margin={info.margin:.2f} | Free={info.margin_free:.2f} | "
                f"Positions={len(self._mt5.positions_get() or [])}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _account_info(self):
        try:
            return self._mt5.account_info()
        except Exception as e:
            logger.warning(f"account_info error: {e}")
            return None

    def _get_position(self, mt5_symbol: str):
        """Return the first open position for mt5_symbol, or None."""
        try:
            positions = self._mt5.positions_get(symbol=mt5_symbol)
            if positions:
                return positions[0]
        except Exception as e:
            logger.debug(f"positions_get error {mt5_symbol}: {e}")
        return None
