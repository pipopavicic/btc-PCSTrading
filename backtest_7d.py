from __future__ import annotations

"""
Backtest: 7-day smoothed production cost + 7-day lag.

Loads BTC price and 7-day smoothed production cost, builds rolling-OU
allocation signals, and runs a long-only BTC/USDC backtest.

Run:
    python backtest_7d.py

Outputs:
    data/derived/inspect/rolling_ou_backtest.csv
    data/derived/inspect/rolling_ou_z_vs_forward50d.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fetch_onchain import fetch_hashrate
from data.fetch_price import fetch_price_data
from models.production_cost_v2 import compute_production_cost_dynamic_v2
from models.engine_rolling_ou import RollingOUConfig, generate_rolling_ou_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INITIAL_CAPITAL = 20_500
TRADING_DAYS    = 252.0
DAYS_PER_YEAR   = 365.0


def load_price_and_cost() -> pd.DataFrame:
    hr       = fetch_hashrate()
    cost_df  = compute_production_cost_dynamic_v2(hr)
    if "production_cost_7d" not in cost_df.columns:
        raise KeyError("Expected 'production_cost_7d' from compute_production_cost_dynamic_v2")

    price_df = fetch_price_data()
    price    = price_df[["close"]].rename(columns={"close": "price_usd"})
    cost     = cost_df[["production_cost_7d"]].rename(columns={"production_cost_7d": "cost_usd"})

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()[~df.index.duplicated(keep="last")]

    price = _prep(price)
    cost  = _prep(cost)
    df    = pd.concat([price, cost], axis=1).sort_index().dropna()
    df    = df[df.index >= pd.Timestamp("2021-01-01", tz="UTC")]
    if df.empty:
        raise ValueError("No aligned price/cost rows found")
    log.info("Loaded %d rows from %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def run_backtest(cfg: RollingOUConfig | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = RollingOUConfig(
            cost_lag_days  = 7,
            min_fit_window = 126,
            entry_z_1=0.75, entry_z_2=1.25, entry_z_3=1.75,
            exit_z_1=0.75,  exit_z_2=1.25,  exit_z_3=1.75,
            step_size      = 1.0 / 3.0,
            apy_usdc       = 0.04,
            trading_start  = "2022-01-01",
        )

    base = load_price_and_cost()
    sig  = generate_rolling_ou_signals(base["price_usd"], base["cost_usd"], cfg)

    bt = sig.copy()
    bt["ret_underlying"]  = bt["price_usd"].pct_change().fillna(0.0)
    bt["position_for_pnl"] = bt["position"].shift(1).fillna(0.0)

    in_usdc            = 1.0 - bt["position_for_pnl"].clip(lower=0.0, upper=1.0)
    bt["carry_ret"]    = in_usdc * (cfg.apy_usdc / DAYS_PER_YEAR)
    bt["strategy_ret"] = bt["position_for_pnl"] * bt["ret_underlying"] + bt["carry_ret"]
    bt["equity_curve"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"] = (1.0 + bt["ret_underlying"]).cumprod()
    bt["value"]        = bt["equity_curve"] * INITIAL_CAPITAL
    bt["forward_50d_ret"] = bt["price_usd"].shift(-50) / bt["price_usd"] - 1.0
    return bt


def summarize_backtest(bt: pd.DataFrame) -> None:
    total_return = bt["equity_curve"].iloc[-1] - 1.0
    daily_mean   = bt["strategy_ret"].mean()
    daily_std    = bt["strategy_ret"].std(ddof=1)
    sharpe       = (daily_mean / daily_std) * np.sqrt(TRADING_DAYS) if daily_std > 0 else 0.0
    drawdown     = bt["equity_curve"] / bt["equity_curve"].cummax() - 1.0
    max_drawdown = float(drawdown.min())

    log.info("=========== Rolling OU PCR Backtest (7d) ===========")
    log.info("Total return:      %.2f%%", total_return * 100)
    log.info("Annualized Sharpe: %.2f",   sharpe)
    log.info("Max drawdown:      %.2f%%", max_drawdown * 100)
    log.info("Final equity:      %.3f",   bt["equity_curve"].iloc[-1])


def main() -> None:
    bt      = run_backtest()
    out_dir = Path("data/derived/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out_dir / "rolling_ou_backtest.csv")
    log.info("Saved backtest to data/derived/inspect/rolling_ou_backtest.csv")
    summarize_backtest(bt)


if __name__ == "__main__":
    main()
