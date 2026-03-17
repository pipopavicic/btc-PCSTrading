"""
Risk management layer for the BTC production cost strategy.

Implements three risk layers:

1. **Trade Level**   – maximum position size = 10% of portfolio value
2. **Position Level** – ATR-based trailing stop: stop = entry − 2 × ATR(14)
3. **Portfolio Level** – 25% USDC cash buffer, rolling 30-day 99% VaR,
                         circuit-breaker if portfolio drops 20% from monthly peak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_POSITION_FRACTION: float = 0.1   # 10% of portfolio per trade
MIN_CASH_BUFFER: float = 0.25         # 25% USDC minimum
ATR_PERIOD: int = 14
ATR_MULTIPLIER: float = 2.0
VAR_CONFIDENCE: float = 2.33          # z-score for 99% VaR
CIRCUIT_BREAKER_THRESHOLD: float = 0.20  # 20% drawdown from monthly peak
VAR_WINDOW: int = 30                  # 30-day rolling window for VaR


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskParameters:
    """Collection of configurable risk parameters.

    Attributes:
        max_position_fraction: Max fraction of portfolio for a single trade.
        min_cash_buffer: Minimum cash fraction to maintain.
        atr_period: Lookback period for ATR calculation.
        atr_multiplier: Multiplier applied to ATR for stop distance.
        var_confidence_z: z-score used for VaR (2.33 ≈ 99%).
        circuit_breaker_threshold: Monthly drawdown threshold to halt entries.
        var_window: Rolling window (days) for VaR calculation.
    """

    max_position_fraction: float = MAX_POSITION_FRACTION
    min_cash_buffer: float = MIN_CASH_BUFFER
    atr_period: int = ATR_PERIOD
    atr_multiplier: float = ATR_MULTIPLIER
    var_confidence_z: float = VAR_CONFIDENCE
    circuit_breaker_threshold: float = CIRCUIT_BREAKER_THRESHOLD
    var_window: int = VAR_WINDOW


@dataclass
class RiskReport:
    """Summary of computed risk metrics for a given date.

    Attributes:
        date: The evaluation date.
        portfolio_value: Total portfolio value in USD.
        position_value: Current BTC position value in USD.
        cash_value: Cash (USDC) portion in USD.
        position_fraction: ``position_value / portfolio_value``.
        cash_fraction: ``cash_value / portfolio_value``.
        atr_stop: ATR-based trailing stop price.
        var_99: 99% 1-day Value-at-Risk in USD.
        monthly_peak: Rolling monthly portfolio peak value.
        drawdown_from_peak: Current drawdown fraction from monthly peak.
        circuit_breaker_active: Whether the circuit breaker is triggered.
        entries_allowed: Whether new entries are permitted.
    """

    date: pd.Timestamp
    portfolio_value: float
    position_value: float
    cash_value: float
    position_fraction: float
    cash_fraction: float
    atr_stop: Optional[float]
    var_99: float
    monthly_peak: float
    drawdown_from_peak: float
    circuit_breaker_active: bool
    entries_allowed: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_atr_stop(
    df_ohlcv: pd.DataFrame,
    entry_price: float,
    params: RiskParameters = RiskParameters(),
) -> float:
    """Compute the ATR-based trailing stop price for a long position.

    Stop is set below entry by ``atr_multiplier × ATR(atr_period)``.

    Args:
        df_ohlcv: OHLCV DataFrame with columns [high, low, close].
        entry_price: The price at which the position was entered.
        params: Risk parameters.

    Returns:
        Stop price (float).  Returns 0.0 if ATR cannot be computed.
    """
    if len(df_ohlcv) < params.atr_period:
        logger.warning("Not enough data to compute ATR (%d rows).", len(df_ohlcv))
        return 0.0

    atr = _compute_atr(df_ohlcv, params.atr_period)
    if atr is None or np.isnan(atr):
        return 0.0

    stop = entry_price - params.atr_multiplier * atr
    logger.debug("ATR=%.2f  entry=%.2f  stop=%.2f", atr, entry_price, stop)
    return float(stop)


def compute_position_size(
    portfolio_value: float,
    current_price: float,
    params: RiskParameters = RiskParameters(),
) -> float:
    """Compute the maximum allowed position size in BTC for a new trade.

    Respects both the ``max_position_fraction`` trade-level limit and the
    ``min_cash_buffer`` portfolio-level constraint.

    Args:
        portfolio_value: Total portfolio value in USD.
        current_price: Current BTC price in USD.
        params: Risk parameters.

    Returns:
        Maximum BTC quantity to buy (float).
    """
    max_trade_usd = portfolio_value * params.max_position_fraction
    max_cash_deploy = portfolio_value * (1.0 - params.min_cash_buffer)
    allowed_usd = min(max_trade_usd, max_cash_deploy)
    btc_qty = allowed_usd / current_price if current_price > 0 else 0.0
    logger.debug(
        "Position size: $%.0f / $%.0f = %.6f BTC",
        allowed_usd,
        current_price,
        btc_qty,
    )
    return float(btc_qty)

def compute_kelly_position_size(
    portfolio_value: float,
    current_price: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    params: RiskParameters = RiskParameters(),
) -> float:
    """Half-Kelly position sizing based on historical win/loss statistics."""
    if avg_loss_pct <= 0 or avg_win_pct <= 0:
        return compute_position_size(portfolio_value, current_price, params)
    
    loss_rate = 1.0 - win_rate
    kelly_f = (win_rate / avg_loss_pct) - (loss_rate / avg_win_pct)
    kelly_f = max(0.0, kelly_f * 0.5)                    # half-Kelly
    kelly_f = min(kelly_f, params.max_position_fraction)  # cap at limit
    
    allowed_usd = portfolio_value * kelly_f
    btc_qty = allowed_usd / current_price if current_price > 0 else 0.0
    logger.info("Kelly f*=%.3f → $%.0f → %.6f BTC", kelly_f, allowed_usd, btc_qty)
    return float(btc_qty)


# ---------------------------------------------------------------------------
# Tranche / accumulation logic (Option B sophisticated sizing)
# ---------------------------------------------------------------------------

TRANCHE_SIZE: float = 0.10          # 10% NAV per tranche
MAX_TRANCHES: int   = 3             # max 3 tranches = 30% max exposure

@dataclass
class TrancheState:
    """Tracks open tranches for accumulation strategy."""
    tranches: list = field(default_factory=list)
    # Each tranche: {"entry_price": float, "pcr_at_entry": float, "size_usd": float}

    @property
    def total_tranches(self) -> int:
        return len(self.tranches)

    @property
    def total_invested_usd(self) -> float:
        return sum(t["size_usd"] for t in self.tranches)


def get_tranche_action(
    pcr: float,
    state: TrancheState,
    portfolio_value: float,
    params: RiskParameters = RiskParameters(),
) -> dict:
    """
    Decide whether to add a tranche, remove a tranche, or hold.

    Entry thresholds (PCR-based):
        Tranche 1: PCR < 0.90
        Tranche 2: PCR < 0.80  (only if tranche 1 already open)
        Tranche 3: PCR < 0.70  (only if tranches 1+2 already open)

    Exit thresholds (FIFO — first tranche in, first out):
        Tranche 1 exits: PCR > 1.40
        Tranche 2 exits: PCR > 1.60
        Tranche 3 exits: PCR > 1.80

    Returns dict with keys:
        "action"     : "BUY_TRANCHE" | "SELL_TRANCHE" | "HOLD"
        "tranche_n"  : which tranche number (1, 2, or 3)
        "size_usd"   : USD amount for this tranche
        "reason"     : human-readable explanation
    """
    entry_thresholds = [0.90, 0.80, 0.70]   # PCR below = add tranche N
    exit_thresholds  = [1.40, 1.60, 1.80]   # PCR above = close tranche N

    n = state.total_tranches

    # ── Check exits first (FIFO) ────────────────────────────────────────────
    if n > 0 and pcr > exit_thresholds[0]:
        # Close the oldest open tranche
        tranche_to_close = state.tranches[0]
        return {
            "action":    "SELL_TRANCHE",
            "tranche_n": 1,
            "size_usd":  tranche_to_close["size_usd"],
            "reason":    f"PCR {pcr:.2f} > exit threshold {exit_thresholds[n-1]:.2f}",
        }

    # ── Check entries ────────────────────────────────────────────────────────
    if n < MAX_TRANCHES and pcr < entry_thresholds[n]:
        # Can only add tranche N+1 if tranche N is already open
        # (n==0 means no tranches yet → eligible for tranche 1 if PCR < 0.90)
        size_usd = compute_kelly_position_size(
        portfolio_value=portfolio_value,
        current_price=1.0,          # dummy — we want USD not BTC qty here
        win_rate=1.0,               # from your backtest: 100% win rate
        avg_win_pct=5.0,            # avg winning trade return (update after backtest)
        avg_loss_pct=1.0,           # avg losing trade return
        params=params,
        ) 
        return {
            "action":    "BUY_TRANCHE",
            "tranche_n": n + 1,
            "size_usd":  size_usd,
            "reason":    f"PCR {pcr:.2f} < entry threshold {entry_thresholds[n]:.2f}",
        }

    return {
        "action":    "HOLD",
        "tranche_n": n,
        "size_usd":  0.0,
        "reason":    f"PCR {pcr:.2f} in neutral zone, {n} tranches open",
    }


def compute_var(
    portfolio_value: float,
    returns: pd.Series,
    params: RiskParameters = RiskParameters(),
) -> float:
    """Compute the rolling 30-day 99% Value-at-Risk in USD.

    Uses a parametric (normal distribution) VaR:

        VaR = position_size × σ_30d × z_{99%}

    where σ_30d is the standard deviation of the last 30 daily returns.

    Args:
        portfolio_value: Current portfolio value invested in BTC (USD).
        returns: Series of daily returns (fractional, e.g. 0.02 for +2%).
        params: Risk parameters.

    Returns:
        1-day 99% VaR in USD (positive number = potential loss).
    """
    window = returns.iloc[-params.var_window:]
    sigma = float(window.std(ddof=1)) if len(window) >= 2 else 0.0
    var = portfolio_value * sigma * params.var_confidence_z
    logger.debug("VaR99: $%.0f  (σ=%.4f  n=%d)", var, sigma, len(window))
    return float(var)


def is_circuit_breaker_active(
    equity_curve: pd.Series,
    params: RiskParameters = RiskParameters(),
) -> bool:
    """Determine whether the circuit breaker should halt new entries.

    Triggers if the portfolio has dropped by more than
    ``circuit_breaker_threshold`` from its rolling 21-day peak (approx. one
    calendar month of trading days).

    Args:
        equity_curve: Series of portfolio values indexed by date.
        params: Risk parameters.

    Returns:
        ``True`` if new entries should be halted; ``False`` otherwise.
    """
    if len(equity_curve) < 2:
        return False

    monthly_peak = equity_curve.rolling(window=21, min_periods=1).max().iloc[-1]
    current = equity_curve.iloc[-1]

    if monthly_peak <= 0:
        return False

    drawdown = (monthly_peak - current) / monthly_peak
    triggered = drawdown >= params.circuit_breaker_threshold
    if triggered:
        logger.warning(
            "Circuit breaker ACTIVE: drawdown %.1f%% from monthly peak $%.0f",
            drawdown * 100,
            monthly_peak,
        )
    return triggered


def evaluate_risk(
    df_ohlcv: pd.DataFrame,
    equity_curve: pd.Series,
    btc_price: float,
    btc_position: float,
    params: RiskParameters = RiskParameters(),
    entry_price: Optional[float] = None,
) -> RiskReport:
    """Generate a comprehensive risk report for the current portfolio state.

    Args:
        df_ohlcv: OHLCV DataFrame used for ATR calculation.
        equity_curve: Historical portfolio value series indexed by date.
        btc_price: Current BTC price in USD.
        btc_position: Current BTC holdings (quantity, not USD).
        params: Risk parameters.
        entry_price: Optional entry price for ATR stop computation.  If
            ``None``, ``btc_price`` is used.

    Returns:
        A :class:`RiskReport` summarising all risk metrics.
    """
    position_value = btc_position * btc_price
    portfolio_value = float(equity_curve.iloc[-1]) if not equity_curve.empty else position_value
    cash_value = max(portfolio_value - position_value, 0.0)
    position_fraction = position_value / portfolio_value if portfolio_value > 0 else 0.0
    cash_fraction = cash_value / portfolio_value if portfolio_value > 0 else 1.0

    atr_stop = compute_atr_stop(df_ohlcv, entry_price or btc_price, params)

    # Daily returns from equity curve
    returns = equity_curve.pct_change().dropna()
    var_99 = compute_var(position_value, returns, params)

    monthly_peak = float(equity_curve.rolling(window=21, min_periods=1).max().iloc[-1])
    drawdown = (monthly_peak - portfolio_value) / monthly_peak if monthly_peak > 0 else 0.0

    circuit_active = is_circuit_breaker_active(equity_curve, params)
    entries_allowed = (
        not circuit_active
        and cash_fraction >= params.min_cash_buffer
        and position_fraction <= params.max_position_fraction
    )

    return RiskReport(
        date=equity_curve.index[-1] if not equity_curve.empty else pd.Timestamp.now(tz="UTC"),
        portfolio_value=portfolio_value,
        position_value=position_value,
        cash_value=cash_value,
        position_fraction=position_fraction,
        cash_fraction=cash_fraction,
        atr_stop=atr_stop if atr_stop > 0 else None,
        var_99=var_99,
        monthly_peak=monthly_peak,
        drawdown_from_peak=drawdown,
        circuit_breaker_active=circuit_active,
        entries_allowed=entries_allowed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int) -> Optional[float]:
    """Compute the most recent ATR value using True Range.

    Args:
        df: OHLCV DataFrame with columns [high, low, close].
        period: Lookback window.

    Returns:
        ATR value as a float, or ``None`` if computation fails.
    """
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()
        return float(atr.iloc[-1])
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("ATR computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

"""if __name__ == "__main__": #OLD - not in use anymore
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_price import fetch_price_data  # noqa: E402

    price_df = fetch_price_data()

    # Simulate a simple equity curve for demonstration
    equity = (price_df["close"] / price_df["close"].iloc[0]) * 10_000
    last_price = float(price_df["close"].iloc[-1])
    btc_held = 0.1

    report = evaluate_risk(
        df_ohlcv=price_df,
        equity_curve=equity,
        btc_price=last_price,
        btc_position=btc_held,
    )
    print(report)"""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.fetch_price import fetch_price_data
    from data.fetch_onchain import fetch_hashrate
    from models.production_cost import compute_production_cost
    from signals.signal_engine import compute_signals

    price_df = fetch_price_data()
    hr_df    = fetch_hashrate()
    cost_df  = compute_production_cost(hr_df)
    sig_df   = compute_signals(price_df, cost_df)

    equity = (price_df["close"] / price_df["close"].iloc[0]) * 10_000
    state  = TrancheState()

    # Simulate last 10 days of tranche decisions
    print("\n── Last 10 days of tranche decisions ──")
    for date, row in sig_df.tail(10).iterrows():
        action = get_tranche_action(row["pcr"], state, 10_000)
        print(f"{date.date()}  PCR={row['pcr']:.3f}  → {action['action']:15s}  {action['reason']}")
        if action["action"] == "BUY_TRANCHE":
            state.tranches.append({"entry_price": row["close"],
                                   "pcr_at_entry": row["pcr"],
                                   "size_usd": action["size_usd"]})
        elif action["action"] == "SELL_TRANCHE" and state.tranches:
            state.tranches.pop(0)

    # Full risk report on current state
    report = evaluate_risk(
        df_ohlcv=price_df,
        equity_curve=equity,
        btc_price=float(price_df["close"].iloc[-1]),
        btc_position=state.total_invested_usd / float(price_df["close"].iloc[-1]),
    )
    print(f"\n{report}")

