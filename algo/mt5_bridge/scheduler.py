"""
Scheduler — fires AMMRO's time-based handlers locally.

Replaces LEAN's self.schedule.on() system.
Tracks which events have already fired today/this month to prevent duplicates.
"""
from __future__ import annotations

import datetime
import time
from typing import Callable

import schedule as sched
from loguru import logger


class Scheduler:
    """
    Time-based scheduler for the local AMMRO runner.

    Registered events:
        monthly   — fires once on the first weekday of each month after market open
        daily     — fires once per trading day after market open
        weekly    — fires once on the first weekday of each week

    All times in the broker's local timezone (default: US/Eastern for equities).
    Adjust MARKET_OPEN_OFFSET_MIN to match your broker's market open time.
    """

    MARKET_OPEN_HOUR   = 9    # 09:30 ET
    MARKET_OPEN_MIN    = 30

    def __init__(self):
        self._monthly_handlers:  list[tuple[int, Callable]] = []  # (offset_min, fn)
        self._daily_handlers:    list[tuple[int, Callable]] = []
        self._weekly_handlers:   list[tuple[int, Callable]] = []
        self._eod_handlers:      list[tuple[int, Callable]] = []  # before close

        self._last_monthly_fire: datetime.date | None = None
        self._last_daily_fire:   dict[Callable, datetime.date] = {}
        self._last_weekly_fire:  dict[Callable, datetime.date] | None = None
        self._running = False

    # ── Registration ─────────────────────────────────────────────────────────

    def on_month_start(self, fn: Callable, offset_min: int = 30) -> None:
        """Fire fn once on the first trading day of each month, offset_min after open."""
        self._monthly_handlers.append((offset_min, fn))

    def on_every_day(self, fn: Callable, offset_min: int = 15) -> None:
        """Fire fn every trading day, offset_min after market open."""
        self._daily_handlers.append((offset_min, fn))

    def on_week_start(self, fn: Callable, offset_min: int = 90) -> None:
        """Fire fn on the first trading day of each week, offset_min after open."""
        self._weekly_handlers.append((offset_min, fn))

    def before_market_close(self, fn: Callable, offset_min: int = 5) -> None:
        """Fire fn offset_min before market close."""
        self._eod_handlers.append((offset_min, fn))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, tick_fn: Callable | None = None,
            tick_interval_sec: int = 60) -> None:
        """
        Start the event loop. Runs indefinitely until KeyboardInterrupt.

        Args:
            tick_fn           : optional function called every tick (e.g. on_data)
            tick_interval_sec : seconds between tick calls (default 60s = 1 minute)
        """
        self._running = True
        logger.info("Scheduler started — press Ctrl+C to stop")

        try:
            while self._running:
                now  = datetime.datetime.now()
                date = now.date()
                tod  = now.time()

                if self._is_trading_day(date):
                    self._fire_daily_events(date, tod)
                    self._fire_weekly_events(date, tod)
                    self._fire_monthly_events(date, tod)
                    self._fire_eod_events(date, tod)

                    if tick_fn is not None:
                        try:
                            tick_fn()
                        except Exception as e:
                            logger.error(f"tick_fn error: {e}")

                time.sleep(tick_interval_sec)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self._running = False

    def stop(self) -> None:
        self._running = False

    # ── Event firing logic ────────────────────────────────────────────────────

    def _fire_daily_events(self, date: datetime.date, tod: datetime.time) -> None:
        for offset_min, fn in self._daily_handlers:
            if self._last_daily_fire.get(fn) == date:
                continue
            fire_time = self._open_plus(offset_min)
            if tod >= fire_time:
                logger.debug(f"Firing daily handler: {fn.__name__}")
                try:
                    fn()
                except Exception as e:
                    logger.error(f"Daily handler {fn.__name__} error: {e}")
                self._last_daily_fire[fn] = date

    def _fire_weekly_events(self, date: datetime.date, tod: datetime.time) -> None:
        week_start = date - datetime.timedelta(days=date.weekday())  # Monday
        for offset_min, fn in self._weekly_handlers:
            fired_key = f"{fn.__name__}_{week_start}"
            if not hasattr(self, "_weekly_fired"):
                self._weekly_fired: set = set()
            if fired_key in self._weekly_fired:
                continue
            if date.weekday() == 0 or date == self._first_trading_day_of_week(date):
                fire_time = self._open_plus(offset_min)
                if tod >= fire_time:
                    logger.debug(f"Firing weekly handler: {fn.__name__}")
                    try:
                        fn()
                    except Exception as e:
                        logger.error(f"Weekly handler {fn.__name__} error: {e}")
                    self._weekly_fired.add(fired_key)

    def _fire_monthly_events(self, date: datetime.date, tod: datetime.time) -> None:
        for offset_min, fn in self._monthly_handlers:
            fired_key = f"{fn.__name__}_{date.year}_{date.month}"
            if not hasattr(self, "_monthly_fired"):
                self._monthly_fired: set = set()
            if fired_key in self._monthly_fired:
                continue
            if date == self._first_trading_day_of_month(date):
                fire_time = self._open_plus(offset_min)
                if tod >= fire_time:
                    logger.debug(f"Firing monthly handler: {fn.__name__}")
                    try:
                        fn()
                    except Exception as e:
                        logger.error(f"Monthly handler {fn.__name__} error: {e}")
                    self._monthly_fired.add(fired_key)

    def _fire_eod_events(self, date: datetime.date, tod: datetime.time) -> None:
        for offset_min, fn in self._eod_handlers:
            fired_key = f"{fn.__name__}_{date}"
            if not hasattr(self, "_eod_fired"):
                self._eod_fired: set = set()
            if fired_key in self._eod_fired:
                continue
            close_time = datetime.time(16, 0)
            fire_time  = datetime.time(15, 60 - offset_min)
            if tod >= fire_time:
                logger.debug(f"Firing EOD handler: {fn.__name__}")
                try:
                    fn()
                except Exception as e:
                    logger.error(f"EOD handler {fn.__name__} error: {e}")
                self._eod_fired.add(fired_key)

    # ── Calendar helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_trading_day(date: datetime.date) -> bool:
        """Simple check: Monday–Friday, non-holiday. Extend with exchange calendar."""
        return date.weekday() < 5

    @staticmethod
    def _open_plus(offset_min: int) -> datetime.time:
        total = (Scheduler.MARKET_OPEN_HOUR * 60
                 + Scheduler.MARKET_OPEN_MIN
                 + offset_min)
        return datetime.time(total // 60, total % 60)

    @staticmethod
    def _first_trading_day_of_month(date: datetime.date) -> datetime.date:
        d = date.replace(day=1)
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        return d

    @staticmethod
    def _first_trading_day_of_week(date: datetime.date) -> datetime.date:
        d = date - datetime.timedelta(days=date.weekday())
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        return d
