"""
Bitcoin mining production cost model.

Estimates the dynamic USD/BTC production cost using the formula:

    Cost ($/BTC) = (fleet_efficiency × energy_rate × hashrate_eh_per_s × 1e18)
                   / (block_reward × blocks_per_hour × watts_per_kw × seconds_per_hour)

A 30-day rolling mean is applied to smooth out short-term volatility.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibrated constants (Hayes 2018, updated for post-halving 2024 parameters)
# ---------------------------------------------------------------------------

FLEET_EFFICIENCY: float = 25.0     # J/TH – average fleet energy efficiency
ENERGY_RATE: float = 0.06          # USD/kWh – average electricity cost
BLOCK_REWARD: float = 3.125        # BTC per block (post-April 2024 halving)
BLOCKS_PER_HOUR: float = 6.0       # average blocks mined per hour
WATTS_PER_KW: float = 1_000.0      # unit conversion
SECONDS_PER_HOUR: float = 3_600.0  # unit conversion
TH_PER_EH: float = 1e6             # 1 EH = 1 × 10^6 TH
ROLLING_WINDOW: int = 30           # days for smoothing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_production_cost(df_hashrate: pd.DataFrame) -> pd.DataFrame:
    """Estimate the USD/BTC production cost from network hashrate.

    The formula converts the network-level electricity cost (in $/hour) to a
    per-BTC cost using the block reward schedule:

        hashrate_th   = hashrate_eh × TH_PER_EH
        power_tw      = hashrate_th × fleet_efficiency   [Watts]
        power_kw      = power_tw / WATTS_PER_KW
        energy_per_h  = power_kw × 1 h                   [kWh/h]
        cost_per_h    = energy_per_h × energy_rate        [USD/h]
        btc_per_h     = block_reward × blocks_per_hour    [BTC/h]
        cost_per_btc  = cost_per_h / btc_per_h            [USD/BTC]

    A ``ROLLING_WINDOW``-day rolling mean is applied to smooth the cost curve.

    Args:
        df_hashrate: DataFrame with a column named ``"hashrate"`` (EH/s) and a
            DatetimeIndex.  Produced by :func:`data.fetch_onchain.fetch_hashrate`.

    Returns:
        DataFrame with columns:

        - ``hashrate``         – original hashrate in EH/s
        - ``production_cost``  – raw instantaneous cost in USD/BTC
        - ``production_cost_smooth`` – 30-day rolling mean of production cost

        The index is preserved from *df_hashrate*.

    Raises:
        KeyError: If *df_hashrate* does not contain a ``"hashrate"`` column.
        ValueError: If *df_hashrate* is empty.
    """
    if df_hashrate.empty:
        raise ValueError("df_hashrate must not be empty.")
    if "hashrate" not in df_hashrate.columns:
        raise KeyError("df_hashrate must contain a 'hashrate' column.")

    df = df_hashrate[["hashrate"]].copy()
    df = df.sort_index()

    # ------------------------------------------------------------------
    # Step 1: Convert EH/s → TH/s
    # ------------------------------------------------------------------
    hashrate_th_per_s = df["hashrate"] * TH_PER_EH

    # ------------------------------------------------------------------
    # Step 2: Network-wide power consumption in Watts
    # ------------------------------------------------------------------
    power_watts = hashrate_th_per_s * FLEET_EFFICIENCY  # J/s = W

    # ------------------------------------------------------------------
    # Step 3: Energy cost per hour (USD/hour)
    # ------------------------------------------------------------------
    power_kw = power_watts / WATTS_PER_KW
    energy_kwh_per_hour = power_kw * 1.0          # 1 hour of operation
    cost_usd_per_hour = energy_kwh_per_hour * ENERGY_RATE

    # ------------------------------------------------------------------
    # Step 4: BTC mined per hour
    # ------------------------------------------------------------------
    btc_per_hour = BLOCK_REWARD * BLOCKS_PER_HOUR  # 3.125 × 6 = 18.75 BTC/h

    # ------------------------------------------------------------------
    # Step 5: USD cost per BTC mined
    # ------------------------------------------------------------------
    df["production_cost"] = cost_usd_per_hour / btc_per_hour

    # ------------------------------------------------------------------
    # Step 6: 30-day rolling smooth
    # ------------------------------------------------------------------
    df["production_cost_smooth"] = (
        df["production_cost"]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    logger.info(
        "Production cost computed for %d rows.  "
        "Latest raw cost: $%.0f/BTC, smoothed: $%.0f/BTC.",
        len(df),
        df["production_cost"].iloc[-1],
        df["production_cost_smooth"].iloc[-1],
    )
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_onchain import fetch_hashrate  # noqa: E402

    hr = fetch_hashrate()
    result = compute_production_cost(hr)
    print(result.tail(10))
