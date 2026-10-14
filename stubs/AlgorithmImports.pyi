# Stub file for AlgorithmImports — mirrors LEAN's Python namespace
# Allows local type-checking and IDE completion without the full LEAN runtime.
# When running on QuantConnect Cloud, this file is ignored.
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

# ── Enums ────────────────────────────────────────────────────────────────────
class Resolution:
    TICK = "Tick"
    SECOND = "Second"
    MINUTE = "Minute"
    HOUR = "Hour"
    DAILY = "Daily"

class AccountType:
    CASH = "Cash"
    MARGIN = "Margin"

class SecurityType:
    EQUITY = "Equity"
    CRYPTO = "Crypto"
    CRYPTO_FUTURE = "CryptoFuture"
    FOREX = "Forex"
    FUTURE = "Future"
    INDEX = "Index"

class OrderStatus:
    FILLED = "Filled"
    PARTIALLY_FILLED = "PartiallyFilled"
    CANCELED = "Canceled"
    INVALID = "Invalid"
    SUBMITTED = "Submitted"
    NONE = "None"

class OrderDirection:
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"

class InsightDirection:
    UP = "Up"
    DOWN = "Down"
    FLAT = "Flat"

class Market:
    USA = "usa"
    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    OANDA = "oanda"

class BrokerageModel:
    INTERACTIVE_BROKERS_BROKERAGE = "InteractiveBrokersBrokerage"
    BINANCE_BROKERAGE = "BinanceBrokerage"
    PAPER_BROKERAGE = "PaperBrokerage"

class MorningstarSectorCode:
    FINANCIAL_SERVICES = 103
    TECHNOLOGY = 311
    HEALTHCARE = 206
    ENERGY = 309
    UTILITIES = 207
    CONSUMER_CYCLICAL = 102
    CONSUMER_DEFENSIVE = 205
    INDUSTRIALS = 310
    BASIC_MATERIALS = 101
    COMMUNICATION_SERVICES = 308
    REAL_ESTATE = 104

# ── Data structures ───────────────────────────────────────────────────────────
class Symbol:
    value: str
    security_type: str
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class TradeBar:
    symbol: Symbol
    time: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    end_time: Any

class Slice:
    time: Any
    margin_interest_rates: Dict[Symbol, Any]
    def __contains__(self, symbol: Symbol) -> bool: ...
    def __getitem__(self, symbol: Symbol) -> TradeBar: ...

class OrderEvent:
    symbol: Symbol
    status: str
    fill_price: float
    fill_quantity: float
    order_id: int
    order_fee: Any
    tag: str
    message: str

class OrderTicket:
    order_id: int
    quantity: float
    quantity_filled: float
    tag: str
    status: str
    def cancel(self, tag: str = "") -> None: ...
    def update_quantity(self, qty: float, tag: str = "") -> None: ...
    def update_limit_price(self, price: float, tag: str = "") -> None: ...

class PortfolioTarget:
    symbol: Symbol
    quantity: float
    def __init__(self, symbol: Symbol, percent: float) -> None: ...

class Fundamental:
    symbol: Symbol
    price: float
    dollar_volume: float
    market_cap: float
    has_fundamental_data: bool
    asset_classification: Any
    operation_ratios: Any

class SecurityChanges:
    added_securities: List[Any]
    removed_securities: List[Any]

class Security:
    symbol: Symbol
    price: float
    fundamentals: Any
    def __getattr__(self, name: str) -> Any: ...

class Portfolio:
    total_portfolio_value: float
    cash: float
    margin_remaining: float
    def __getitem__(self, symbol: Symbol) -> Any: ...
    def values(self) -> Iterable[Any]: ...
    def items(self) -> Iterable[Any]: ...

# ── Indicators ────────────────────────────────────────────────────────────────
class IndicatorBase:
    is_ready: bool
    current: Any
    def update(self, time: Any, value: float) -> bool: ...

class RelativeStrengthIndex(IndicatorBase): ...
class AverageTrueRange(IndicatorBase): ...
class Minimum(IndicatorBase):
    def __init__(self, period: int) -> None: ...
class Maximum(IndicatorBase):
    def __init__(self, period: int) -> None: ...
class SimpleMovingAverage(IndicatorBase): ...
class ExponentialMovingAverage(IndicatorBase): ...

# ── Base Algorithm class ──────────────────────────────────────────────────────
class QCAlgorithm:
    time: Any
    portfolio: Portfolio
    securities: Dict[Symbol, Security]
    active_securities: Dict[Symbol, Security]
    transactions: Any
    schedule: Any
    date_rules: Any
    time_rules: Any
    universe_settings: Any
    settings: Any
    notify: Any
    is_warming_up: bool

    def set_start_date(self, year: int, month: int, day: int) -> None: ...
    def set_end_date(self, year: int, month: int, day: int) -> None: ...
    def set_cash(self, amount: float) -> None: ...
    def set_brokerage_model(self, model: str, account_type: str = ...) -> None: ...
    def set_benchmark(self, ticker: str) -> None: ...
    def set_warm_up(self, period: int, resolution: str = ...) -> None: ...

    def add_equity(self, ticker: str, resolution: str = ...) -> Security: ...
    def add_index(self, ticker: str, resolution: str = ...) -> Security: ...
    def add_crypto(self, ticker: str, resolution: str = ..., market: str = ...) -> Security: ...
    def add_crypto_future(self, ticker: str, resolution: str = ..., market: str = ...) -> Security: ...
    def add_universe(self, filter_func: Callable) -> Any: ...

    def market_order(self, symbol: Symbol, quantity: float, tag: str = "") -> OrderTicket: ...
    def limit_order(self, symbol: Symbol, quantity: float, limit_price: float, tag: str = "") -> OrderTicket: ...
    def stop_market_order(self, symbol: Symbol, quantity: float, stop_price: float, tag: str = "") -> OrderTicket: ...
    def set_holdings(self, symbol_or_targets: Any, percentage: float = ..., liquidate_existing_holdings: bool = False, tag: str = "") -> None: ...
    def liquidate(self, symbol: Optional[Symbol] = None, tag: str = "") -> None: ...

    def history(self, symbol_or_type: Any, periods: int = ..., resolution: str = ...) -> Any: ...

    def rsi(self, symbol: Symbol, period: int, resolution: str = ...) -> RelativeStrengthIndex: ...
    def atr(self, symbol: Symbol, period: int, resolution: str = ...) -> AverageTrueRange: ...
    def sma(self, symbol: Symbol, period: int, resolution: str = ...) -> SimpleMovingAverage: ...
    def ema(self, symbol: Symbol, period: int, resolution: str = ...) -> ExponentialMovingAverage: ...
    def deregister_indicator(self, indicator: IndicatorBase) -> None: ...

    def is_market_open(self, symbol: Symbol) -> bool: ...
    def log(self, message: str) -> None: ...
    def plot(self, chart: str, series: str, value: float) -> None: ...
    def debug(self, message: str) -> None: ...
