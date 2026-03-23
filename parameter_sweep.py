from __future__ import annotations

"""
Grid search over cost variant, lag, OU window, and z-score thresholds.

Run:
    python parameter_sweep.py

Outputs:
    data/derived/inspect/rolling_ou_parameter_sweep.csv
    data/derived/inspect/rolling_ou_best_backtest.csv
    data/derived/inspect/rolling_ou_backtest_50davg_50dlag.csv
"""

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch_onchain import fetch_hashrate
from data.fetch_price import fetch_price_data
from models.production_cost_v2 import compute_production_cost_dynamic_v2
from models.engine_rolling_ou import RollingOUConfig, generate_rolling_ou_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INITIAL_CAPITAL = 40_000.0
TRADING_DAYS    = 252.0
DAYS_PER_YEAR   = 365.0


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()[~df.index.duplicated(keep="last")]


def load_price_and_costs() -> pd.DataFrame:
    hr       = fetch_hashrate()
    cost_df  = compute_production_cost_dynamic_v2(hr)
    price_df = fetch_price_data()

    price = _prep(price_df[["close"]].rename(columns={"close": "price_usd"}))
    cost  = _prep(cost_df[["production_cost", "production_cost_7d"]].copy())
    cost["production_cost_50d"] = cost["production_cost"].rolling(50, min_periods=1).mean()

    df = pd.concat([price, cost], axis=1).sort_index().dropna()
    df = df[df.index >= pd.Timestamp("2021-01-01", tz="UTC")]
    if df.empty:
        raise ValueError("No aligned price/cost rows found")
    log.info("Loaded %d rows", len(df))
    return df


def run_backtest(price: pd.Series, cost: pd.Series, cfg: RollingOUConfig) -> pd.DataFrame:
    sig = generate_rolling_ou_signals(price, cost, cfg)
    bt  = sig.copy()
    bt["ret_underlying"]  = bt["price_usd"].pct_change().fillna(0.0)
    bt["position_for_pnl"] = bt["position"].shift(1).fillna(0.0)
    in_usdc            = 1.0 - bt["position_for_pnl"].clip(lower=0.0, upper=1.0)
    bt["carry_ret"]    = in_usdc * (cfg.apy_usdc / DAYS_PER_YEAR)
    bt["strategy_ret"] = bt["position_for_pnl"] * bt["ret_underlying"] + bt["carry_ret"]
    bt["equity_curve"] = (1.0 + bt["strategy_ret"]).cumprod()
    return bt


def summarize(bt: pd.DataFrame) -> dict:
    total_return = float(bt["equity_curve"].iloc[-1] - 1.0)
    daily_mean   = float(bt["strategy_ret"].mean())
    daily_std    = float(bt["strategy_ret"].std(ddof=1))
    sharpe       = (daily_mean / daily_std) * np.sqrt(TRADING_DAYS) if daily_std > 0 else 0.0
    drawdown     = bt["equity_curve"] / bt["equity_curve"].cummax() - 1.0
    return {
        "total_return":  total_return,
        "sharpe":        float(sharpe),
        "max_drawdown":  float(drawdown.min()),
        "avg_position":  float(bt["position_for_pnl"].mean()),
        "final_equity":  float(bt["equity_curve"].iloc[-1]),
    }


def run_parameter_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    base  = load_price_and_costs()
    price = base["price_usd"]

    cost_variants  = ["production_cost_7d", "production_cost_50d"]
    lag_grid       = [7, 30, 50]
    window_grid    = [126, 252, 365]
    threshold_grid = [(0.50, 1.00, 1.50), (0.25, 0.75, 1.25), (0.75, 1.25, 1.75)]

    rows       = []
    best_bt    = None
    best_score = -np.inf

    for cost_name, lag, window, thresh in itertools.product(
        cost_variants, lag_grid, window_grid, threshold_grid
    ):
        cfg = RollingOUConfig(
            cost_lag_days=lag, min_fit_window=window,
            entry_z_1=thresh[0], entry_z_2=thresh[1], entry_z_3=thresh[2],
            exit_z_1=thresh[0],  exit_z_2=thresh[1],  exit_z_3=thresh[2],
            step_size=1.0 / 3.0, apy_usdc=0.04, trading_start="2022-01-01",
        )
        try:
            bt      = run_backtest(price, base[cost_name], cfg)
            metrics = summarize(bt)
        except Exception as exc:
            rows.append({"cost_variant": cost_name, "cost_lag_days": lag,
                         "fit_window": window, "thresholds": str(thresh), "error": str(exc)})
            continue

        row = {"cost_variant": cost_name, "cost_lag_days": lag, "fit_window": window,
               "entry_z_1": thresh[0], "entry_z_2": thresh[1], "entry_z_3": thresh[2],
               **metrics}
        rows.append(row)

        score = metrics["sharpe"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_bt    = bt.copy()

    results = pd.DataFrame(rows)
    if "sharpe" in results.columns:
        results = results.sort_values(["sharpe", "total_return"], ascending=[False, False])
    if best_bt is None:
        raise ValueError("Sweep produced no successful backtests")
    return results, best_bt


def main() -> None:
    out_dir = Path("data/derived/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results, best_bt = run_parameter_sweep()
    sweep_results.to_csv(out_dir / "rolling_ou_parameter_sweep.csv", index=False)
    best_bt.to_csv(out_dir / "rolling_ou_best_backtest.csv")
    log.info("Top 10 configs:\n%s", sweep_results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
