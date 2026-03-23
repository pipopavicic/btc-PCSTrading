"""
BTC Production Cost Model v2
==============================
Grounded in:
- Hayes (2018): "Bitcoin price and its marginal cost of production"
  arXiv:1805.07610
- Gottschalk (2022): "Digital currency price formation: A production cost perspective"
  QFE 6(4): 669-695, DOI:10.3934/QFE.2022030

Key inputs:
  (1) Time-varying hardware efficiency E_t  — step function from best ASIC at each date
  (2) Time-varying electricity price U_t    — regime-based on mining geography

Smoothing columns produced:
  production_cost    — raw daily P* (noisy)
  production_cost_7d — 7-day rolling mean  ← primary input for OU model
  production_cost_smooth — 365-day EWM (legacy, do not use for signals)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# =============================================================================
# Hardware Efficiency Table — Gottschalk (2022) Table 1, extended post-2022
# Converted from Mhash/J → J/TH via: J/TH = 1_000_000 / (Mhash/J)
# =============================================================================
_EFFICIENCY_TABLE: list[tuple[str, float]] = [
    ("2009-01-03", 1_000_000.0),
    ("2010-09-01",   500_000.0),
    ("2011-08-18",    66_667.0),
    ("2013-01-30",     8_547.0),
    ("2013-04-16",     7_692.0),
    ("2013-05-31",     1_420.0),
    ("2013-09-13",     1_100.0),
    ("2014-04-12",       850.0),
    ("2014-09-01",       613.0),
    ("2014-12-27",       511.0),
    ("2015-07-01",       249.0),
    ("2016-06-01",       107.0),
    ("2017-08-01",        98.2),
    ("2017-12-20",        90.0),
    ("2018-03-01",        89.5),
    ("2018-07-01",        86.2),
    ("2018-08-01",        82.5),
    ("2018-09-01",        63.6),
    ("2018-10-01",        45.0),
    ("2019-04-01",        39.5),
    ("2020-04-01",        38.0),
    ("2020-05-01",        29.5),
    ("2021-07-01",        29.5),
    ("2022-05-01",        27.5),
    ("2022-07-01",        21.5),
    ("2023-06-01",        17.5),
    ("2024-04-01",        15.0),
    ("2025-01-01",        13.5),
]

_EFF_INDEX  = pd.to_datetime([r[0] for r in _EFFICIENCY_TABLE], utc=True)
_EFF_VALUES = np.array([r[1] for r in _EFFICIENCY_TABLE])
EFFICIENCY_SERIES = pd.Series(_EFF_VALUES, index=_EFF_INDEX, name="efficiency_J_TH")


def build_efficiency_series(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorised efficiency look-up (merge_asof step function)."""
    idx_utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    table   = pd.DataFrame({"date": _EFF_INDEX, "value": _EFF_VALUES}).sort_values("date")
    target  = pd.DataFrame({"date": idx_utc}).sort_values("date")
    merged  = pd.merge_asof(target, table, on="date", direction="backward")
    merged["value"] = merged["value"].fillna(table["value"].iloc[0])
    merged.index    = idx_utc
    merged          = merged.reindex(idx_utc)
    return pd.Series(merged["value"].to_numpy(), index=index, name="efficiency_J_TH")


# =============================================================================
# Electricity Price Regimes — Gottschalk (2022) Section 3.3
# =============================================================================
_ELEC_REGIMES: list[tuple[str, float]] = [
    ("2009-01-01", 0.050),
    ("2012-01-01", 0.050),
    ("2014-01-01", 0.050),
    ("2021-07-01", 0.055),
    ("2022-01-01", 0.071),
]
_ELEC_IDX  = pd.to_datetime([r[0] for r in _ELEC_REGIMES], utc=True)
_ELEC_VALS = np.array([r[1] for r in _ELEC_REGIMES])


def build_structural_elec_series(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorised electricity rate look-up."""
    idx_utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    table   = pd.DataFrame({"date": _ELEC_IDX, "value": _ELEC_VALS}).sort_values("date")
    target  = pd.DataFrame({"date": idx_utc}).sort_values("date")
    merged  = pd.merge_asof(target, table, on="date", direction="backward")
    merged["value"] = merged["value"].fillna(table["value"].iloc[0])
    merged.index    = idx_utc
    merged          = merged.reindex(idx_utc)
    return pd.Series(merged["value"].to_numpy(), index=index, name="structural_elec_rate")


# =============================================================================
# Block reward schedule
# =============================================================================
BLOCK_REWARD_SCHEDULE: list[tuple[str, float]] = [
    ("2009-01-03", 50.0),
    ("2012-11-28", 25.0),
    ("2016-07-09", 12.5),
    ("2020-05-11",  6.25),
    ("2024-04-20",  3.125),
]
_REWARD_IDX  = pd.to_datetime([r[0] for r in BLOCK_REWARD_SCHEDULE], utc=True)
_REWARD_VALS = np.array([r[1] for r in BLOCK_REWARD_SCHEDULE])


def get_block_reward(date: pd.Timestamp) -> float:
    if date.tzinfo is None:
        date = date.tz_localize("UTC")
    mask = _REWARD_IDX <= date
    return float(_REWARD_VALS[mask][-1]) if mask.any() else 50.0


SECONDS_PER_DAY = 86_400
J_PER_KWH       = 3_600_000
PUE             = 1.12
POOL_FEE        = 0.01
SMOOTHING_7D    = 7
SMOOTHING_LEGACY = 365


# =============================================================================
# Core computation
# =============================================================================
def compute_production_cost_v2(
    hashrate_df: pd.DataFrame,
    dynamic_energy_rate: float | None = None,
    dynamic_blend_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Compute daily BTC production cost using Hayes (2018) + Gottschalk (2022) inputs.

    Formula:
        power_W          = hashrate_TH_s * efficiency_J_TH * PUE
        energy_kWh_day   = power_W * 86400 / 3_600_000
        btc_per_day      = block_reward * 6 * 24
        P*               = energy_kWh_day * electricity_rate / btc_per_day / (1 - pool_fee)
    """
    df = hashrate_df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")

    df["hashrate_TH"]          = df["hashrate"] * 1_000_000
    df["efficiency_J_TH"]      = build_efficiency_series(df.index).values
    df["structural_elec_rate"] = build_structural_elec_series(df.index).values

    if dynamic_energy_rate is not None:
        df["effective_elec_rate"] = (
            (1 - dynamic_blend_weight) * df["structural_elec_rate"]
            + dynamic_blend_weight * dynamic_energy_rate
        )
    else:
        df["effective_elec_rate"] = df["structural_elec_rate"]

    df["block_reward"] = [get_block_reward(d) for d in df.index]

    power_w        = df["hashrate_TH"] * df["efficiency_J_TH"] * PUE
    energy_kwh_day = power_w * SECONDS_PER_DAY / J_PER_KWH
    btc_per_day    = df["block_reward"] * 6 * 24

    df["production_cost"]    = energy_kwh_day * df["effective_elec_rate"] / btc_per_day / (1 - POOL_FEE)
    df["production_cost_7d"] = df["production_cost"].rolling(window=SMOOTHING_7D, min_periods=1).mean()
    df["production_cost_smooth"] = df["production_cost"].ewm(span=SMOOTHING_LEGACY, min_periods=30).mean()

    return df


# Alias used by backtest scripts
def compute_production_cost_dynamic_v2(
    hashrate_df: pd.DataFrame,
    dynamic_energy_rate: float | None = None,
    dynamic_blend_weight: float = 0.50,
) -> pd.DataFrame:
    """Drop-in alias for compute_production_cost_v2."""
    return compute_production_cost_v2(hashrate_df, dynamic_energy_rate, dynamic_blend_weight)
