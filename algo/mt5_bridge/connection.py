"""
MT5Connection — manages the connection to MetaTrader 5.

Linux  : mt5linux wraps the rpyc socket to a Wine-hosted MT5 terminal.
Windows: MetaTrader5 package connects directly to the locally running terminal.

Setup (Linux):
  1. Install Wine: sudo dnf install wine
  2. Download MT5 installer, run under Wine
  3. Inside MT5 terminal: Tools → Options → Expert Advisors → allow DLL imports
  4. Copy metatrader5_server.py (from mt5linux docs) into MT5 terminal's MQL5/Scripts/
  5. Run the script from MT5: it starts an rpyc server on port 18812
  6. Connect from Python: MT5Connection(host="127.0.0.1", port=18812)
"""
from __future__ import annotations

import sys
from loguru import logger


def _import_mt5():
    """Import MT5 API — mt5linux on Linux, MetaTrader5 on Windows."""
    if sys.platform == "win32":
        try:
            import MetaTrader5 as mt5
            return mt5
        except ImportError:
            raise RuntimeError(
                "MetaTrader5 package not found. "
                "Install with: pip install MetaTrader5")
    else:
        try:
            from mt5linux import MetaTrader5 as mt5
            return mt5
        except ImportError:
            raise RuntimeError(
                "mt5linux package not found. "
                "Install with: uv add mt5linux\n"
                "Then start the MT5 rpyc server inside Wine/Windows MT5 terminal.")


class MT5Connection:
    """
    Context-manager wrapper around the MT5 connection lifecycle.

    Example:
        with MT5Connection(account=123456, password="pass", server="ICMarkets-Demo") as conn:
            mt5 = conn.mt5
            ticks = mt5.copy_rates_from_pos("AAPL", mt5.TIMEFRAME_D1, 0, 100)
    """

    def __init__(
        self,
        account:  int  = 0,
        password: str  = "",
        server:   str  = "",
        host:     str  = "127.0.0.1",   # only used on Linux (mt5linux rpyc host)
        port:     int  = 18812,          # only used on Linux (mt5linux rpyc port)
        timeout:  int  = 60_000,         # ms
    ):
        self._account  = account
        self._password = password
        self._server   = server
        self._host     = host
        self._port     = port
        self._timeout  = timeout
        self._mt5      = None

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def mt5(self):
        if self._mt5 is None:
            raise RuntimeError("Not connected. Use 'with MT5Connection(...) as conn:'")
        return self._mt5

    def connect(self) -> "MT5Connection":
        mt5 = _import_mt5()

        # Linux: pass host/port to mt5linux; Windows: no extra args
        if sys.platform != "win32":
            mt5.initialize(host=self._host, port=self._port)
        else:
            if not mt5.initialize():
                raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

        if self._account:
            ok = mt5.login(self._account, password=self._password,
                           server=self._server, timeout=self._timeout)
            if not ok:
                mt5.shutdown()
                raise ConnectionError(
                    f"MT5 login failed for account {self._account}: {mt5.last_error()}")

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise ConnectionError("MT5 connected but account_info() returned None")

        logger.info(
            f"MT5 connected | account={info.login} | "
            f"server={info.server} | balance={info.balance:.2f} {info.currency}")

        self._mt5 = mt5
        return self

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        if self._mt5 is None:
            return False
        try:
            return self._mt5.terminal_info() is not None
        except Exception:
            return False

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MT5Connection":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.disconnect()
