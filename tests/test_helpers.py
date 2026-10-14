"""
Smoke tests for pure helper functions that don't need the LEAN runtime.
Run with: uv run pytest
"""
import numpy as np
import pytest


# ── Helpers copied from algo layer for local unit testing ─────────────────────

def normalize(raw_dict: dict) -> dict:
    if not raw_dict:
        return {}
    vals = list(raw_dict.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    return {k: (v - lo) / span for k, v in raw_dict.items()}


def inverse_vol_weights(vols: dict, equity_sleeve: float = 0.85,
                        position_cap: float = 0.15) -> dict:
    inv_vol_sum = sum(1.0 / max(v, 1e-9) for v in vols.values())
    raw = {s: (1.0 / max(v, 1e-9)) / inv_vol_sum * equity_sleeve
           for s, v in vols.items()}
    return {s: min(w, position_cap) for s, w in raw.items()}


def compute_drawdown(peak: float, current: float) -> float:
    return (peak - current) / peak if peak > 0 else 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_normalize_basic():
    scores = {"A": 10, "B": 20, "C": 30}
    result = normalize(scores)
    assert result["A"] == pytest.approx(0.0)
    assert result["C"] == pytest.approx(1.0)
    assert 0.0 <= result["B"] <= 1.0


def test_normalize_empty():
    assert normalize({}) == {}


def test_normalize_all_same():
    result = normalize({"A": 5, "B": 5, "C": 5})
    assert all(v == pytest.approx(0.0) for v in result.values())


def test_inverse_vol_weights_sum():
    vols = {"A": 0.10, "B": 0.20, "C": 0.15, "D": 0.05}
    weights = inverse_vol_weights(vols, equity_sleeve=0.85, position_cap=0.15)
    assert sum(weights.values()) <= 0.85 + 1e-9


def test_inverse_vol_weights_cap():
    # Single stock should be capped at position_cap even with very low vol
    vols = {"A": 0.001}
    weights = inverse_vol_weights(vols, equity_sleeve=0.85, position_cap=0.15)
    assert weights["A"] <= 0.15 + 1e-9


def test_inverse_vol_higher_weight_for_lower_vol():
    vols = {"LOW": 0.05, "HIGH": 0.25}
    weights = inverse_vol_weights(vols, equity_sleeve=0.85, position_cap=0.50)
    assert weights["LOW"] > weights["HIGH"]


def test_drawdown_zero_at_peak():
    assert compute_drawdown(peak=100_000, current=100_000) == pytest.approx(0.0)


def test_drawdown_ten_percent():
    assert compute_drawdown(peak=100_000, current=90_000) == pytest.approx(0.10)


def test_drawdown_zero_peak_guard():
    assert compute_drawdown(peak=0, current=50_000) == 0.0
