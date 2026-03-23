from __future__ import annotations

"""
Backtest: 50-day smoothed production cost + 50-day lag + risk overlay.

Run:
    python backtest_50d.py

Outputs:
    data/derived/inspect/rolling_ou_backtest_50davg_50dlag.csv
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
from models.risk import RiskConfig, apply_risk_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INITIAL_CAPITAL = 17_200
TRADING_DAYS    = 252.0
DAYS_PER_YEAR   = 365.0


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()[~df.index.duplicated(keep="last")]


def load_price_and_cost_50d() -> pd.DataFrame:
    hr      = fetch_hashrate()
    cost_df = compute_production_cost_dynamic_v2(hr)
    if "production_cost" not in cost_df.columns:
        raise KeyError("Expected 'production_cost' from compute_production_cost_dynamic_v2")

    price_df = fetch_price_data()
    price    = _prep(price_df[["close"]].rename(columns={"close": "price_usd"}))
    cost     = _prep(cost_df[["production_cost"]].copy())
    cost["production_cost_50d"] = cost["production_cost"].rolling(50, min_periods=1).mean()

    df = pd.concat([price, cost[["production_cost_50d"]]], axis=1).sort_index().dropna()
    df = df[df.index >= pd.Timestamp("2021-01-01", tz="UTC")]
    if df.empty:
        raise ValueError("No aligned price/cost rows found")
    log.info("Loaded %d rows from %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def run_backtest() -> pd.DataFrame:
    base = load_price_and_cost_50d()
    cfg  = RollingOUConfig(
        cost_lag_days  = 50,
        min_fit_window = 126,
        entry_z_1=0.75, entry_z_2=1.25, entry_z_3=1.75,
        exit_z_1=0.75,  exit_z_2=1.25,  exit_z_3=1.75,
        step_size      = 1.0 / 3.0,
        apy_usdc       = 0.04,
        trading_start  = "2022-01-01",
    )

    bt = generate_rolling_ou_signals(base["price_usd"], base["production_cost_50d"], cfg)
    bt["production_cost_50d"] = base["production_cost_50d"].reindex(bt.index)
    bt["ret_underlying"]      = bt["price_usd"].pct_change().fillna(0.0)
    bt["position_for_pnl"]   = bt["position"].shift(1).fillna(0.0)

    in_usdc            = 1.0 - bt["position_for_pnl"].clip(lower=0.0, upper=1.0)
    bt["carry_ret"]    = in_usdc * (cfg.apy_usdc / DAYS_PER_YEAR)
    bt["strategy_ret"] = bt["position_for_pnl"] * bt["ret_underlying"] + bt["carry_ret"]

    trade_start = pd.Timestamp(cfg.trading_start, tz="UTC")
    bt["trade_active"] = bt.index >= trade_start
    bt.loc[~bt["trade_active"], "strategy_ret"] = 0.0

    bt["equity_curve"]   = (1.0 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"] = (1.0 + bt["ret_underlying"]).cumprod()
    bt["value"]          = bt["equity_curve"] * INITIAL_CAPITAL
    bt["forward_50d_ret"] = bt["price_usd"].shift(-50) / bt["price_usd"] - 1.0

    risk_cfg = RiskConfig()
    pcr      = bt["price_usd"] / bt["production_cost_50d"]
    bt       = apply_risk_overlay(bt, bt["price_usd"], risk_cfg, pcr_series=pcr)
    return bt


def summarize(bt: pd.DataFrame) -> None:
    total_return = bt["equity_curve"].iloc[-1] - 1.0
    daily_mean   = bt["strategy_ret"].mean()
    daily_std    = bt["strategy_ret"].std(ddof=1)
    sharpe       = (daily_mean / daily_std) * np.sqrt(TRADING_DAYS) if daily_std > 0 else 0.0
    drawdown     = bt["equity_curve"] / bt["equity_curve"].cummax() - 1.0
    max_drawdown = float(drawdown.min())

    log.info("=========== Rolling OU PCR Backtest (50d avg / 50d lag) ===========")
    log.info("Total return:      %.2f%%", total_return * 100)
    log.info("Annualized Sharpe: %.2f",   sharpe)
    log.info("Max drawdown:      %.2f%%", max_drawdown * 100)
    log.info("Final equity:      %.3f",   bt["equity_curve"].iloc[-1])


def main() -> None:
    bt      = run_backtest()
    out_dir = Path("data/derived/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out_dir / "rolling_ou_backtest_50davg_50dlag.csv")
    log.info("Saved backtest to data/derived/inspect/rolling_ou_backtest_50davg_50dlag.csv")
    summarize(bt)


if __name__ == "__main__":
    main()
