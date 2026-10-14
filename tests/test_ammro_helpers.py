"""
Unit tests for AMMRO pure-logic helpers.
Tests run against extracted helper functions that require no LEAN runtime.
Run with: uv run pytest -v
"""
import numpy as np
import pytest


# ── Helpers extracted verbatim from AMMROAlgorithm ───────────────────────────

def normalize(raw: dict) -> dict:
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    return {k: (v - lo) / span for k, v in raw.items()}


def inverse_vol_weights(vols: dict, equity_sleeve=0.85, position_cap=0.15,
                        sector_cap=0.30) -> dict:
    inv_vol_sum = sum(1.0 / max(v, 1e-9) for v in vols.values())
    weights = {
        s: min((1.0 / max(v, 1e-9)) / inv_vol_sum * equity_sleeve, position_cap)
        for s, v in vols.items()
    }
    return weights


def compute_drawdown(peak: float, current: float) -> float:
    return (peak - current) / peak if peak > 0 else 0.0


def blend_scores(m: dict, q: dict, r: dict,
                 w_m=0.60, w_q=0.30, w_r=0.10) -> dict:
    syms = set(m) & set(q) & set(r)
    return {s: w_m * m[s] + w_q * q.get(s, 0.0) + w_r * r.get(s, 0.5) for s in syms}


def apply_sector_caps(weights: dict, sector_map: dict, cap: float = 0.30) -> dict:
    totals = {}
    for s, w in weights.items():
        sec = sector_map.get(s, -1)
        totals[sec] = totals.get(sec, 0.0) + w
    adjusted = dict(weights)
    for sector, total in totals.items():
        if total > cap:
            scale = cap / total
            for s in adjusted:
                if sector_map.get(s, -1) == sector:
                    adjusted[s] *= scale
    return adjusted


def pre_trade_gate(weights: dict, peak: float, current: float,
                   position_cap=0.15, defensive_dd=0.08, halt_dd=0.12,
                   halted=False) -> dict:
    if halted:
        return {}
    dd = compute_drawdown(peak, current)
    if dd >= halt_dd:
        return {}
    if dd >= defensive_dd:
        weights = {s: w * 0.5 for s, w in weights.items()}
    return {s: min(w, position_cap) for s, w in weights.items()}


# ── normalize ────────────────────────────────────────────────────────────────

def test_normalize_range():
    result = normalize({"A": 0, "B": 50, "C": 100})
    assert result["A"] == pytest.approx(0.0)
    assert result["C"] == pytest.approx(1.0)
    assert result["B"] == pytest.approx(0.5)

def test_normalize_empty():
    assert normalize({}) == {}

def test_normalize_all_same():
    result = normalize({"A": 7, "B": 7})
    assert all(v == pytest.approx(0.0) for v in result.values())

def test_normalize_negative_values():
    result = normalize({"A": -10, "B": 0, "C": 10})
    assert result["A"] == pytest.approx(0.0)
    assert result["C"] == pytest.approx(1.0)

def test_normalize_single():
    result = normalize({"X": 42})
    assert result["X"] == pytest.approx(0.0)


# ── inverse_vol_weights ──────────────────────────────────────────────────────

def test_weights_sum_below_sleeve():
    vols = {"A": 0.10, "B": 0.20, "C": 0.15, "D": 0.05}
    w = inverse_vol_weights(vols)
    assert sum(w.values()) <= 0.85 + 1e-9

def test_position_cap_enforced():
    # One very-low-vol symbol would dominate without cap
    vols = {"A": 0.001, "B": 0.5, "C": 0.5}
    w = inverse_vol_weights(vols, position_cap=0.15)
    assert w["A"] <= 0.15 + 1e-9

def test_lower_vol_gets_higher_weight():
    vols = {"LO": 0.05, "HI": 0.30}
    w = inverse_vol_weights(vols, position_cap=0.50)
    assert w["LO"] > w["HI"]

def test_zero_vol_guard():
    # Should not raise ZeroDivisionError
    vols = {"A": 0.0, "B": 0.10}
    w = inverse_vol_weights(vols)
    assert all(v >= 0 for v in w.values())


# ── drawdown ─────────────────────────────────────────────────────────────────

def test_drawdown_zero_at_peak():
    assert compute_drawdown(100_000, 100_000) == pytest.approx(0.0)

def test_drawdown_twelve_percent():
    assert compute_drawdown(100_000, 88_000) == pytest.approx(0.12)

def test_drawdown_zero_peak_guard():
    assert compute_drawdown(0, 50_000) == 0.0

def test_drawdown_full_loss():
    assert compute_drawdown(100_000, 0) == pytest.approx(1.0)


# ── score blending ───────────────────────────────────────────────────────────

def test_blend_intersection_only():
    m = {"A": 0.8, "B": 0.6, "C": 0.9}
    q = {"A": 0.7, "B": 0.5}           # C missing
    r = {"A": 1.0, "B": 0.5, "C": 0.0}
    result = blend_scores(m, q, r)
    assert "A" in result
    assert "B" in result
    assert "C" not in result            # C excluded from intersection

def test_blend_weights_sum_to_one():
    w_m, w_q, w_r = 0.60, 0.30, 0.10
    assert w_m + w_q + w_r == pytest.approx(1.0)

def test_blend_full_score_of_one():
    m = {"A": 1.0}; q = {"A": 1.0}; r = {"A": 1.0}
    result = blend_scores(m, q, r)
    assert result["A"] == pytest.approx(1.0)

def test_blend_zero_score():
    m = {"A": 0.0}; q = {"A": 0.0}; r = {"A": 0.0}
    result = blend_scores(m, q, r)
    assert result["A"] == pytest.approx(0.0)


# ── sector caps ──────────────────────────────────────────────────────────────

def test_sector_cap_scales_overweight_sector():
    weights    = {"A": 0.20, "B": 0.20, "C": 0.10}
    sector_map = {"A": 1, "B": 1, "C": 2}   # A+B in sector 1 = 0.40 > 0.30 cap
    result = apply_sector_caps(weights, sector_map, cap=0.30)
    sector1_total = result["A"] + result["B"]
    assert sector1_total <= 0.30 + 1e-9

def test_sector_cap_does_not_touch_compliant():
    weights    = {"A": 0.10, "B": 0.08, "C": 0.15}
    sector_map = {"A": 1, "B": 2, "C": 3}   # all sectors well under 30%
    result = apply_sector_caps(weights, sector_map, cap=0.30)
    assert result == pytest.approx(weights)


# ── pre-trade gate ───────────────────────────────────────────────────────────

def test_gate_passes_when_no_breach():
    w = {"A": 0.10, "B": 0.12}
    result = pre_trade_gate(w, peak=100_000, current=95_000)
    assert result == pytest.approx(w)

def test_gate_blocks_when_halted():
    w = {"A": 0.10}
    result = pre_trade_gate(w, peak=100_000, current=95_000, halted=True)
    assert result == {}

def test_gate_blocks_at_halt_dd():
    w = {"A": 0.10}
    result = pre_trade_gate(w, peak=100_000, current=87_000)  # 13% DD
    assert result == {}

def test_gate_halves_at_defensive_dd():
    w = {"A": 0.10, "B": 0.14}
    # 9% drawdown → defensive mode → halve weights
    result = pre_trade_gate(w, peak=100_000, current=91_000)
    assert result["A"] == pytest.approx(0.05)
    assert result["B"] == pytest.approx(0.07)

def test_gate_enforces_position_cap():
    w = {"A": 0.20}   # exceeds 0.15 cap
    result = pre_trade_gate(w, peak=100_000, current=100_000, position_cap=0.15)
    assert result["A"] == pytest.approx(0.15)
