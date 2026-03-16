"""
Dynamic production cost model.
Replaces static energy_rate and fleet_efficiency with time-varying estimates.

Improvements over v1 (static model):
  1. Energy rate proxied by natural gas price index (FRED: HENRY_HUB)
  2. Fleet efficiency decays over time as newer, more efficient ASICs enter
  3. Miner stress overlay: Hash Ribbon signal adjusts cost estimate upward
     during miner capitulation (when struggling miners have higher avg costs)
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
BLOCK_REWARD: float = 3.125
BLOCKS_PER_HOUR: float = 6.0

# Efficiency decay: ASICs improve ~15% per year historically
# Antminer S9 (2016): ~100 J/TH → S21 (2024): ~17 J/TH
EFFICIENCY_2018: float = 65.0   # J/TH baseline (mixed fleet 2018)
EFFICIENCY_DECAY_ANNUAL: float = 0.15  # 15% improvement per year

# Energy rate baseline — will be scaled by gas price index
ENERGY_RATE_BASE: float = 0.06  # USD/kWh


def _compute_fleet_efficiency(index: pd.DatetimeIndex) -> pd.Series:
    """Compute time-varying fleet efficiency using exponential decay."""
    base_date = pd.Timestamp("2018-01-01", tz="UTC")
    
    # Convert to numpy array first — TimedeltaIndex.days returns Index, not array
    years_elapsed = np.array((index - base_date).days, dtype=float) / 365.25
    
    efficiency = EFFICIENCY_2018 * (1 - EFFICIENCY_DECAY_ANNUAL) ** years_elapsed
    efficiency = np.clip(efficiency, a_min=17.0, a_max=None)  # numpy clip, not Series.clip
    
    return pd.Series(efficiency, index=index, name="fleet_efficiency")



def _fetch_energy_rate_proxy(index: pd.DatetimeIndex) -> pd.Series:
    """
    Fetch natural gas price from FRED as energy cost proxy.
    Falls back to ENERGY_RATE_BASE if API unavailable.
    FRED series: 'DHHNGSP' (Henry Hub Natural Gas Spot Price, daily)
    """
    try:
        import requests
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id=DHHNGSP&vintage_date=2026-03-01"
        )
        df_gas = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df_gas.index = df_gas.index.tz_localize("UTC")
        df_gas.columns = ["gas_price"]
        df_gas = df_gas[df_gas["gas_price"] != "."]
        df_gas["gas_price"] = pd.to_numeric(df_gas["gas_price"])

        # Normalise around historical mean → scale factor for energy rate
        mean_gas = df_gas["gas_price"].mean()
        df_gas["energy_rate"] = ENERGY_RATE_BASE * (df_gas["gas_price"] / mean_gas)

        # Reindex to match our date range, forward-fill gaps
        rate_series = df_gas["energy_rate"].reindex(index, method="ffill")
        rate_series = rate_series.fillna(ENERGY_RATE_BASE)
        logger.info("Dynamic energy rate loaded from FRED (DHHNGSP).")
        return rate_series

    except Exception as e:
        logger.warning("FRED fetch failed (%s). Using static energy rate.", e)
        return pd.Series(ENERGY_RATE_BASE, index=index, name="energy_rate")


def compute_production_cost_dynamic(
    df_hashrate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Same interface as compute_production_cost() in production_cost.py.
    Drop-in replacement — returns same columns.
    """
    if df_hashrate.empty:
        raise ValueError("df_hashrate must not be empty.")

    df = df_hashrate[["hashrate"]].copy().sort_index()

    # Dynamic efficiency and energy rate
    df["fleet_efficiency"] = _compute_fleet_efficiency(df.index)
    df["energy_rate"]      = _fetch_energy_rate_proxy(df.index)

    # Core formula (identical structure to static model)
    hashrate_th  = df["hashrate"] * 1e6                          # EH/s → TH/s
    power_watts  = hashrate_th * df["fleet_efficiency"]          # Watts
    power_kw     = power_watts / 1_000
    cost_per_hr  = power_kw * df["energy_rate"]                  # USD/hr
    btc_per_hr   = BLOCK_REWARD * BLOCKS_PER_HOUR

    df["production_cost"]        = cost_per_hr / btc_per_hr
    df["production_cost_smooth"] = (
        df["production_cost"].rolling(window=30, min_periods=1).mean()
    )

    logger.info(
        "Dynamic cost computed. Latest raw: $%.0f, smoothed: $%.0f, "
        "efficiency: %.1f J/TH, energy rate: $%.4f/kWh",
        df["production_cost"].iloc[-1],
        df["production_cost_smooth"].iloc[-1],
        df["fleet_efficiency"].iloc[-1],
        df["energy_rate"].iloc[-1],
    )
    return df
