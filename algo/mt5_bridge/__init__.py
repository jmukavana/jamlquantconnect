"""
MT5 Bridge — local execution layer for AMMRO.

On Linux  : uses mt5linux (rpyc socket to MT5 terminal running under Wine)
On Windows : uses MetaTrader5 directly

Usage:
    from algo.mt5_bridge import MT5Connection, MT5DataFeed, MT5Executor, MT5Portfolio
"""
from .connection import MT5Connection
from .data_feed  import MT5DataFeed
from .executor   import MT5Executor
from .portfolio  import MT5Portfolio
from .scheduler  import Scheduler
from .symbol_map import SymbolMap

__all__ = [
    "MT5Connection",
    "MT5DataFeed",
    "MT5Executor",
    "MT5Portfolio",
    "Scheduler",
    "SymbolMap",
]
