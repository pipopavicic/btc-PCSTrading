"""
Vectorbt backtest for the BTC production cost strategy.

Runs an end-to-end simulation and prints key performance metrics:

    - Sharpe Ratio
    - Max Drawdown
    - Calmar Ratio
    - Win Rate
    - Equity Curve (saved as PNG)

Usage:
    python backtest/run_backtest.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a top-level script or as a module
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.fetch_price import fetch_price_data          # noqa: E402
from data.fetch_onchain import fetch_hashrate          # noqa: E402
# Engine 1 — Static (current)
#from models.production_cost import compute_production_cost  # noqa: E402
# Engine 2 — Dynamic (new)
from models.production_cost_dynamic import compute_production_cost_dynamic as compute_production_cost  # noqa: E402

from signals.signal_engine import compute_signals      # noqa: E402


from risk.risk_manager import (                        # noqa: E402
    RiskParameters,
    TrancheState,
    get_tranche_action,
    evaluate_risk,
    is_circuit_breaker_active,
    compute_kelly_position_size,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Portfolio settings
# ---------------------------------------------------------------------------

INIT_CASH: float = 10_000.0
FEES: float = 0.001     # 0.1% per side
SLIPPAGE: float = 0.001  # 0.1% per trade

MAX_TRANCHES: int = 3

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

OUTPUT_DIR = ROOT / "backtest"
EQUITY_CURVE_PATH = OUTPUT_DIR / "equity_curve.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_backtest() -> dict:
    """Execute the full vectorbt backtest and return performance metrics.

    Steps:
        1. Fetch BTC price and hashrate data.
        2. Compute production cost model.
        3. Generate signals (entries and exits).
        4. Run vectorbt portfolio simulation.
        5. Print and return performance metrics.

    Returns:
        Dictionary with keys: sharpe, max_drawdown, calmar, win_rate,
        total_return, n_trades.
    """
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("Fetching price data …")
    price_df = fetch_price_data()

    logger.info("Fetching hashrate data …")
    hr_df = fetch_hashrate()

    # ------------------------------------------------------------------
    # 2. Production cost
    # ------------------------------------------------------------------
    logger.info("Computing production cost …")
    cost_df = compute_production_cost(hr_df)

    # ------------------------------------------------------------------
    # 3. Signals
    # ------------------------------------------------------------------
    logger.info("Computing signals …")
    signals_df = compute_signals(price_df, cost_df)

    close   = signals_df["close"]
    raw_entries = signals_df["entry_signal"].fillna(False).astype(bool)
    raw_exits   = signals_df["exit_signal"].fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Risk Layer 1: Circuit Breaker — block entries during drawdown
    # ------------------------------------------------------------------
    risk_params  = RiskParameters()
    equity_proxy = (close / close.iloc[0]) * INIT_CASH
    monthly_peak = equity_proxy.rolling(window=21, min_periods=1).max()
    monthly_dd   = (equity_proxy - monthly_peak) / monthly_peak
    cb_mask      = monthly_dd < -risk_params.circuit_breaker_threshold

    entries = raw_entries & ~cb_mask   # gate: no new entries when halted
    exits   = raw_exits.copy()

    cb_blocked = (raw_entries & cb_mask).sum()
    if cb_blocked > 0:
        logger.info("Circuit breaker blocked %d entry signals.", cb_blocked)

    # ------------------------------------------------------------------
    # Risk Layer 2: Tranche Entry Thresholds
    # PCR must cross progressively deeper thresholds to add tranches
    # Tranche 1: PCR < 0.90  (already in entries from signal_engine)
    # Tranche 2: PCR < 0.80  (only if we've already had a T1 signal)
    # Tranche 3: PCR < 0.70
    # ------------------------------------------------------------------
    pcr = signals_df["pcr"]

    tranche_1 = raw_entries & (pcr < 0.90) & ~cb_mask
    tranche_2 = raw_entries & (pcr < 0.80) & ~cb_mask
    tranche_3 = raw_entries & (pcr < 0.70) & ~cb_mask

    # Combine: fire entry if ANY tranche threshold is crossed
    entries = (tranche_1 | tranche_2 | tranche_3)
    exits   = raw_exits.copy()

    logger.info(
        "Tranche signals — T1: %d  T2: %d  T3: %d  Exits: %d",
        tranche_1.sum(), tranche_2.sum(), tranche_3.sum(), exits.sum(),
    )


    logger.info(
        "Signal summary: %d entries, %d exits over %d trading days.",
        entries.sum(),
        exits.sum(),
        len(close),
    )

    # ------------------------------------------------------------------
    # 4. Vectorbt simulation
    # ------------------------------------------------------------------
    try:
        import vectorbt as vbt  # type: ignore[import]
    except ImportError:
        logger.error(
            "vectorbt is not installed.  Run: pip install vectorbt"
        )
        sys.exit(1)

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=INIT_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
        accumulate=True,          # ← allows multiple buys without full exit
        size=0.333,               # ← deploy 33% of available cash per tranche
        size_type="percent",      # ← interpret size as fraction of available cash
    )


    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    stats = portfolio.stats()
    sharpe = float(stats.get("Sharpe Ratio", np.nan))
    max_dd = float(stats.get("Max Drawdown [%]", np.nan)) / 100.0
    total_return = float(stats.get("Total Return [%]", np.nan)) / 100.0
    n_trades = int(stats.get("Total Trades", 0))

    # Calmar ratio = annualised return / |max drawdown|
    years = len(close) / 365.0
    annualised_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = annualised_return / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate from trades
    win_rate = _compute_win_rate(portfolio)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  BTC Production Cost Strategy – Backtest Results")
    print("=" * 60)
    print(f"  Period          : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Trading days    : {len(close)}")
    print(f"  Total trades    : {n_trades}")
    print(f"  Total return    : {total_return * 100:.1f}%")
    print(f"  Annualised ret  : {annualised_return * 100:.1f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd * 100:.1f}%")
    print(f"  Calmar Ratio    : {calmar:.3f}" if not np.isnan(calmar) else "  Calmar Ratio    : N/A")
    print(f"  Win Rate        : {win_rate * 100:.1f}%")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Buy-and-hold comparison
    # ------------------------------------------------------------------
    bh_return = (close.iloc[-1] / close.iloc[0] - 1)
    print(f"  Buy-and-Hold return over same period: {bh_return * 100:.1f}%\n")
    # ------------------------------------------------------------------
    # Risk Report — current state
    # ------------------------------------------------------------------
    try:
        final_nav    = pd.Series(portfolio.value().values,
                                index=portfolio.value().index)
        last_price   = float(close.iloc[-1])
                # Infer open position from portfolio final state
        final_btc_value = float(portfolio.value().iloc[-1]) - float(portfolio.cash().iloc[-1])
        btc_held        = max(final_btc_value, 0.0) / last_price
        open_tranches   = int(entries.iloc[-30:].sum())  # proxy: entries in last 30 days


        risk_report = evaluate_risk(
            df_ohlcv=price_df,
            equity_curve=final_nav,
            btc_price=last_price,
            btc_position=btc_held,
            params=risk_params,
        )

        print("  Risk Report (end of backtest period):")
        print(f"    Open tranches      : {open_tranches} / {MAX_TRANCHES}")
        print(f"    ATR Stop Price     : ${risk_report.atr_stop:,.0f}" if risk_report.atr_stop else "    ATR Stop Price     : N/A")
        print(f"    99% 1-day VaR      : ${risk_report.var_99:,.0f}")
        print(f"    Drawdown from peak : {risk_report.drawdown_from_peak * 100:.1f}%")
        print(f"    Circuit breaker    : {'ACTIVE ⛔' if risk_report.circuit_breaker_active else 'Inactive ✅'}")
        print(f"    New entries allowed: {'Yes ✅' if risk_report.entries_allowed else 'No ⛔'}")
        print("=" * 60 + "\n")
    except Exception as exc:
        logger.warning("Risk report failed: %s", exc)

    # ------------------------------------------------------------------
    # Save equity curve
    # ------------------------------------------------------------------
    _save_equity_curve(portfolio, close, OUTPUT_DIR, EQUITY_CURVE_PATH)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_return": total_return,
        "n_trades": n_trades,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_win_rate(portfolio) -> float:
    """Compute win rate from a vectorbt portfolio.

    Args:
        portfolio: Vectorbt Portfolio object.

    Returns:
        Win rate as a fraction in [0, 1].
    """
    try:
        trades = portfolio.trades.records_readable
        if trades.empty:
            return float("nan")
        wins = (trades["PnL"] > 0).sum()
        return float(wins / len(trades))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not compute win rate: %s", exc)
        return float("nan")


def _save_equity_curve(portfolio, close: pd.Series, output_dir: Path, path: Path) -> None:
    """Save the equity curve and buy-and-hold benchmark as a PNG.

    Args:
        portfolio: Vectorbt Portfolio object.
        close: Close price series (for buy-and-hold normalisation).
        output_dir: Directory in which to save the file.
        path: Full output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt  # noqa: PLC0415

        output_dir.mkdir(parents=True, exist_ok=True)
        equity = portfolio.value()
        bh = (close / close.iloc[0]) * INIT_CASH

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)
        ax.plot(bh.index, bh.values, label="Buy & Hold", linewidth=1.5, linestyle="--")
        ax.set_title("BTC Production Cost Strategy – Equity Curve")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Equity curve saved to %s", path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not save equity curve: %s", exc)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_backtest()
