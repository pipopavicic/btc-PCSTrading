"""
Signal generation engine for the BTC production cost strategy.

Computes the Price-to-Cost Ratio (PCR) and applies RSI confirmation filters
to produce actionable trading signals.

Signal zones:
    PCR < 0.9          → STRONG_BUY
    0.9 ≤ PCR < 1.1    → WEAK_BUY
    1.1 ≤ PCR < 2.0    → HOLD
    2.0 ≤ PCR < 3.5    → SELL
    PCR ≥ 3.5          → STRONG_SELL

RSI confirmation:
    Buy signals  : RSI < 45
    Sell signals : RSI > 55
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal zone thresholds
# ---------------------------------------------------------------------------

PCR_STRONG_BUY: float = 0.9
PCR_WEAK_BUY: float = 1.1
PCR_HOLD: float = 2.0
PCR_STRONG_SELL: float = 3.5

RSI_BUY_THRESHOLD: float = 45.0
RSI_SELL_THRESHOLD: float = 55.0
RSI_PERIOD: int = 14


# ---------------------------------------------------------------------------
# Signal labels
# ---------------------------------------------------------------------------

class Signal:
    """Enumeration of signal string labels."""
    STRONG_BUY: str = "STRONG_BUY"
    WEAK_BUY: str = "WEAK_BUY"
    HOLD: str = "HOLD"
    SELL: str = "SELL"
    STRONG_SELL: str = "STRONG_SELL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_signals(
    df_price: pd.DataFrame,
    df_cost: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    use_smooth_cost: bool = True,
) -> pd.DataFrame:
    """Compute PCR-based signals with RSI confirmation on merged price/cost data.

    Args:
        df_price: OHLCV DataFrame with at minimum a ``"close"`` column, indexed
            by UTC datetime.  Produced by
            :func:`data.fetch_price.fetch_price_data`.
        df_cost: Production cost DataFrame with columns
            ``["production_cost", "production_cost_smooth"]``, indexed by UTC
            datetime.  Produced by
            :func:`models.production_cost.compute_production_cost`.
        rsi_period: Lookback window for RSI calculation (default 14).
        use_smooth_cost: If ``True`` (default), use the 30-day smoothed cost;
            otherwise use the raw instantaneous cost.

    Returns:
        DataFrame with the following columns appended to a daily close series:

        - ``close``               – BTC/USDT close price
        - ``production_cost``     – smoothed (or raw) production cost
        - ``pcr``                 – Price-to-Cost Ratio
        - ``rsi``                 – RSI(14)
        - ``signal_zone``         – one of Signal.* string labels
        - ``entry_signal``        – boolean, True on confirmed buy entry
        - ``exit_signal``         – boolean, True on confirmed sell exit

    Raises:
        ValueError: If the merged DataFrame is empty after alignment.
    """
    cost_col = "production_cost_smooth" if use_smooth_cost else "production_cost"

    # ------------------------------------------------------------------
    # Merge price and cost on daily timestamps, forward-fill cost gaps
    # (on-chain data has irregular timestamps)
    # ------------------------------------------------------------------
    price_daily = df_price[["close"]].resample("1D").last().dropna()
    cost_daily = df_cost[[cost_col]].resample("1D").last().dropna()

    merged = price_daily.join(cost_daily, how="inner")
    merged = merged.rename(columns={cost_col: "production_cost"})

    if merged.empty:
        raise ValueError(
            "Merged price/cost DataFrame is empty.  Check that the date ranges overlap."
        )

    # ------------------------------------------------------------------
    # Price-to-Cost Ratio
    # ------------------------------------------------------------------
    merged["pcr"] = merged["close"] / merged["production_cost"]

    # ------------------------------------------------------------------
    # RSI (computed on close prices using pandas-ta)
    # ------------------------------------------------------------------
    rsi_series = ta.rsi(merged["close"], length=rsi_period)
    merged["rsi"] = rsi_series

    # ------------------------------------------------------------------
    # Signal zones
    # ------------------------------------------------------------------
    merged["signal_zone"] = _classify_signal_zone(merged["pcr"])

    # ------------------------------------------------------------------
    # Confirmed entry / exit signals
    # ------------------------------------------------------------------
    merged["entry_signal"] = (
        (merged["pcr"] < PCR_STRONG_BUY) & (merged["rsi"] < RSI_BUY_THRESHOLD)
    )
    merged["exit_signal"] = (
        (merged["pcr"] > PCR_HOLD) & (merged["rsi"] > RSI_SELL_THRESHOLD)
    )

    logger.info(
        "Signals computed.  Entry bars: %d  Exit bars: %d  "
        "Latest PCR: %.3f  Latest RSI: %.1f",
        merged["entry_signal"].sum(),
        merged["exit_signal"].sum(),
        merged["pcr"].iloc[-1],
        merged["rsi"].iloc[-1] if not merged["rsi"].isna().all() else float("nan"),
    )
    return merged


def _classify_signal_zone(pcr: pd.Series) -> pd.Series:
    """Map PCR values to discrete signal zone labels.

    Args:
        pcr: Series of Price-to-Cost Ratio values.

    Returns:
        String series with one of the :class:`Signal` labels.
    """
    conditions = [
        pcr < PCR_STRONG_BUY,
        pcr < PCR_WEAK_BUY,
        pcr < PCR_HOLD,
        pcr < PCR_STRONG_SELL,
    ]
    choices = [
        Signal.STRONG_BUY,
        Signal.WEAK_BUY,
        Signal.HOLD,
        Signal.SELL,
    ]
    return pd.Series(
        np.select(conditions, choices, default=Signal.STRONG_SELL),
        index=pcr.index,
        dtype="object",
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_price import fetch_price_data        # noqa: E402
    from data.fetch_onchain import fetch_hashrate        # noqa: E402
    from models.production_cost import compute_production_cost  # noqa: E402

    price = fetch_price_data()
    hr = fetch_hashrate()
    cost = compute_production_cost(hr)
    signals = compute_signals(price, cost)
    print(signals[["close", "production_cost", "pcr", "rsi", "signal_zone"]].tail(10))
