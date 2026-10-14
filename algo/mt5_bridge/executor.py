"""
MT5Executor — places, modifies, and cancels orders via MT5.
Maps AMMRO order concepts (market, stop, limit, liquidate) to MT5 trade requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

from .symbol_map import SymbolMap


MAGIC = 20250503       # unique identifier for all AMMRO orders
DEVIATION = 20         # max slippage in points


@dataclass
class OrderResult:
    success:    bool
    order_id:   int    = 0
    fill_price: float  = 0.0
    volume:     float  = 0.0
    comment:    str    = ""
    error:      str    = ""


class MT5Executor:
    """
    Executes trades through MT5.

    Methods mirror the LEAN execution API used by AMMRO:
        market_order(ticker, qty, tag)
        limit_order(ticker, qty, price, tag)
        stop_market_order(ticker, qty, stop_price, tag)
        liquidate(ticker, tag)
        cancel_order(order_id)
        set_holdings(ticker, target_weight, portfolio, data_feed, tag)
    """

    def __init__(self, mt5, symbol_map: SymbolMap):
        self._mt5 = mt5
        self._sym = symbol_map

    # ── Core order methods ────────────────────────────────────────────────────

    def market_order(self, ticker: str, qty: float, tag: str = "") -> OrderResult:
        """
        Place a market order.
        qty > 0 = BUY, qty < 0 = SELL.
        If a position exists on the opposite side, it is closed first.
        """
        mt5_sym = self._sym.to_mt5(ticker)
        if qty == 0:
            return OrderResult(False, comment="qty=0, skipped")

        action   = self._mt5.ORDER_TYPE_BUY  if qty > 0 else self._mt5.ORDER_TYPE_SELL
        tick     = self._mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return OrderResult(False, error=f"No tick for {mt5_sym}")

        price    = tick.ask if qty > 0 else tick.bid
        volume   = round(abs(qty), 2)

        request = {
            "action":       self._mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_sym,
            "volume":       volume,
            "type":         action,
            "price":        price,
            "deviation":    DEVIATION,
            "magic":        MAGIC,
            "comment":      tag[:31],   # MT5 comment max 31 chars
            "type_time":    self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(self._mt5.last_error())
            logger.error(f"market_order FAILED {ticker} qty={qty:+.2f} | {err}")
            return OrderResult(False, error=err)

        logger.info(f"FILLED [{tag}] {ticker} qty={qty:+.2f} @ {result.price:.4f}")
        return OrderResult(True, order_id=result.order,
                           fill_price=result.price, volume=volume)

    def stop_market_order(self, ticker: str, qty: float,
                          stop_price: float, tag: str = "") -> OrderResult:
        """
        Place a stop order (pending).
        qty < 0 = SELL_STOP (stop loss on a long position).
        """
        mt5_sym  = self._sym.to_mt5(ticker)
        vol      = round(abs(qty), 2)
        order_type = (self._mt5.ORDER_TYPE_BUY_STOP
                      if qty > 0 else self._mt5.ORDER_TYPE_SELL_STOP)

        request = {
            "action":       self._mt5.TRADE_ACTION_PENDING,
            "symbol":       mt5_sym,
            "volume":       vol,
            "type":         order_type,
            "price":        stop_price,
            "deviation":    DEVIATION,
            "magic":        MAGIC,
            "comment":      tag[:31],
            "type_time":    self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_RETURN,
        }

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(self._mt5.last_error())
            logger.warning(f"stop_order FAILED {ticker} stop={stop_price} | {err}")
            return OrderResult(False, error=err)

        logger.info(f"STOP PLACED [{tag}] {ticker} qty={qty:+.2f} @ {stop_price:.4f}")
        return OrderResult(True, order_id=result.order, volume=vol)

    def limit_order(self, ticker: str, qty: float,
                    limit_price: float, tag: str = "") -> OrderResult:
        """Place a pending limit order."""
        mt5_sym    = self._sym.to_mt5(ticker)
        vol        = round(abs(qty), 2)
        order_type = (self._mt5.ORDER_TYPE_BUY_LIMIT
                      if qty > 0 else self._mt5.ORDER_TYPE_SELL_LIMIT)

        request = {
            "action":       self._mt5.TRADE_ACTION_PENDING,
            "symbol":       mt5_sym,
            "volume":       vol,
            "type":         order_type,
            "price":        limit_price,
            "deviation":    DEVIATION,
            "magic":        MAGIC,
            "comment":      tag[:31],
            "type_time":    self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_RETURN,
        }

        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(self._mt5.last_error())
            logger.warning(f"limit_order FAILED {ticker} | {err}")
            return OrderResult(False, error=err)

        logger.info(f"LIMIT [{tag}] {ticker} qty={qty:+.2f} @ {limit_price:.4f}")
        return OrderResult(True, order_id=result.order, volume=vol)

    def liquidate(self, ticker: str, tag: str = "") -> OrderResult:
        """Close all open positions for ticker at market price."""
        mt5_sym   = self._sym.to_mt5(ticker)
        positions = self._mt5.positions_get(symbol=mt5_sym) or []
        if not positions:
            return OrderResult(True, comment="no position to liquidate")

        last_result = OrderResult(True)
        for pos in positions:
            close_type = (self._mt5.ORDER_TYPE_SELL
                          if pos.type == 0 else self._mt5.ORDER_TYPE_BUY)
            tick  = self._mt5.symbol_info_tick(mt5_sym)
            price = tick.bid if pos.type == 0 else tick.ask

            request = {
                "action":       self._mt5.TRADE_ACTION_DEAL,
                "symbol":       mt5_sym,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     pos.ticket,
                "price":        price,
                "deviation":    DEVIATION,
                "magic":        MAGIC,
                "comment":      tag[:31],
                "type_time":    self._mt5.ORDER_TIME_GTC,
                "type_filling": self._mt5.ORDER_FILLING_IOC,
            }
            result = self._mt5.order_send(request)
            if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
                err = result.comment if result else str(self._mt5.last_error())
                logger.error(f"liquidate FAILED {ticker} ticket={pos.ticket} | {err}")
                last_result = OrderResult(False, error=err)
            else:
                logger.info(f"LIQUIDATED [{tag}] {ticker} "
                            f"vol={pos.volume:.2f} @ {result.price:.4f}")

        return last_result

    def liquidate_all(self, tag: str = "halt") -> None:
        """Close every open position. Used by the circuit breaker halt."""
        try:
            positions = self._mt5.positions_get() or []
        except Exception:
            return
        for pos in positions:
            ticker = self._sym.from_mt5(pos.symbol)
            self.liquidate(ticker, tag=tag)

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order by ticket ID."""
        request = {
            "action": self._mt5.TRADE_ACTION_REMOVE,
            "order":  order_id,
        }
        result = self._mt5.order_send(request)
        if result and result.retcode == self._mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order {order_id} cancelled")
            return True
        err = result.comment if result else str(self._mt5.last_error())
        logger.warning(f"cancel_order {order_id} FAILED | {err}")
        return False

    def set_holdings(self, ticker: str, target_weight: float,
                     portfolio_value: float, current_price: float,
                     tag: str = "") -> OrderResult:
        """
        Rebalance a position to target_weight of portfolio_value.
        Buys or sells the difference between current and target exposure.
        """
        if current_price <= 0:
            return OrderResult(False, error="price <= 0")

        mt5_sym     = self._sym.to_mt5(ticker)
        target_val  = portfolio_value * target_weight
        target_qty  = target_val / current_price

        positions   = self._mt5.positions_get(symbol=mt5_sym) or []
        current_qty = sum(p.volume if p.type == 0 else -p.volume
                          for p in positions)

        delta = round(target_qty - current_qty, 2)
        if abs(delta) < self._min_lot(mt5_sym):
            return OrderResult(True, comment=f"delta {delta:.2f} below min lot, skipped")

        return self.market_order(ticker, delta, tag=tag)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _min_lot(self, mt5_symbol: str) -> float:
        try:
            info = self._mt5.symbol_info(mt5_symbol)
            return info.volume_min if info else 0.01
        except Exception:
            return 0.01
