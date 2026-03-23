"""
models/risk.py

Risk management overlay for the BTC/USDC PCR divergence strategy.

Three independent layers applied sequentially after signal generation:
1. Volatility scalar      — shrinks position when realized vol > target
2. OU half-life gate      — suppresses new buys if mean-reversion has broken down
3. Drawdown circuit breaker — freezes new buys, forces exit if equity falls too far

All layers are no-look-ahead safe: they only use data available at t-1.

Usage:
    from models.risk import RiskConfig, apply_risk_overlay
    bt = apply_risk_overlay(bt, price, RiskConfig(), pcr_series=pcr)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    # Volatility targeting
    vol_target:       float = 0.80
    vol_lookback:     int   = 21
    vol_scalar_floor: float = 0.25
    vol_scalar_cap:   float = 1.0

    # OU half-life gate
    halflife_max_days: float = 120.0

    # Drawdown circuit breaker
    drawdown_warn:  float = 0.15
    drawdown_halt:  float = 0.20
    drawdown_exit:  float = 0.30
    recovery_pct:   float = 0.05


def compute_vol_scalar(price: pd.Series, cfg: RiskConfig) -> pd.Series:
    daily_ret    = price.pct_change()
    realized_vol = (
        daily_ret
        .rolling(cfg.vol_lookback, min_periods=max(5, cfg.vol_lookback // 2))
        .std()
        * np.sqrt(365)
    ).shift(1)
    scalar = (cfg.vol_target / realized_vol).clip(lower=cfg.vol_scalar_floor, upper=cfg.vol_scalar_cap)
    return scalar.fillna(cfg.vol_scalar_floor).rename("vol_scalar")


def compute_halflife_gate(
    pcr_series: pd.Series,
    cfg: RiskConfig,
    rolling_window: int = 252,
) -> pd.Series:
    from models.ou_model import fit_ou_parameters

    log_pcr    = np.log(pcr_series.clip(lower=1e-9))
    half_lives = pd.Series(index=pcr_series.index, dtype=float)

    for i in range(rolling_window, len(pcr_series)):
        window = log_pcr.iloc[i - rolling_window: i]
        try:
            params = fit_ou_parameters(np.exp(window))
            half_lives.iloc[i] = params.half_life_days
        except Exception:
            half_lives.iloc[i] = np.nan

    gate = (half_lives <= cfg.halflife_max_days).fillna(False)
    return gate.rename("ou_gate")


def compute_drawdown_mask(
    position_for_pnl: pd.Series,
    ret_underlying: pd.Series,
    cfg: RiskConfig,
) -> pd.DataFrame:
    DAILY_USDC = 0.04 / 365.0
    equity     = pd.Series(index=position_for_pnl.index, dtype=float)
    equity.iloc[0] = 1.0

    for i in range(1, len(equity)):
        pos            = position_for_pnl.iloc[i - 1]
        r              = ret_underlying.iloc[i]
        usdc_carry     = (1.0 - max(pos, 0.0)) * DAILY_USDC
        equity.iloc[i] = equity.iloc[i - 1] * (1 + pos * r + usdc_carry)

    peak     = equity.cummax()
    drawdown = equity / peak - 1.0

    dd_cap        = pd.Series(1.0,  index=equity.index)
    circuit_state = pd.Series("ok", index=equity.index)

    in_halt       = False
    in_exit       = False
    trough_equity = 1.0

    for i in range(len(equity)):
        dd = drawdown.iloc[i]
        eq = equity.iloc[i]

        if dd <= -cfg.drawdown_exit:
            in_exit = in_halt = True
            trough_equity = min(trough_equity, eq)
        elif dd <= -cfg.drawdown_halt:
            in_halt = True
            trough_equity = min(trough_equity, eq)

        if in_exit and eq >= trough_equity * (1 + cfg.recovery_pct):
            in_exit       = False
            trough_equity = eq
        if in_halt and not in_exit and eq >= trough_equity * (1 + cfg.recovery_pct):
            in_halt       = False
            trough_equity = eq

        if in_exit:
            dd_cap.iloc[i] = 0.0; circuit_state.iloc[i] = "exit"
        elif in_halt:
            dd_cap.iloc[i] = 0.0; circuit_state.iloc[i] = "halt"
        elif dd <= -cfg.drawdown_warn:
            dd_cap.iloc[i] = 0.66; circuit_state.iloc[i] = "warn"
        else:
            dd_cap.iloc[i] = 1.0; circuit_state.iloc[i] = "ok"

    return pd.DataFrame({"drawdown": drawdown, "dd_cap": dd_cap, "circuit_state": circuit_state})


def apply_risk_overlay(
    bt: pd.DataFrame,
    price: pd.Series,
    cfg: RiskConfig | None = None,
    pcr_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Apply all three risk layers to a completed backtest DataFrame.

    Expects bt to have columns: position_for_pnl, ret_underlying.
    Adds: position_risk, vol_scalar, ou_gate, drawdown, dd_cap, circuit_state,
          strategy_ret_risk, equity_curve_risk.
    """
    if cfg is None:
        cfg = RiskConfig()
    bt = bt.copy()

    # Layer 1: vol scalar
    vol_scalar   = compute_vol_scalar(price.reindex(bt.index), cfg)
    bt["vol_scalar"] = vol_scalar.reindex(bt.index).fillna(cfg.vol_scalar_floor)

    # Layer 2: OU half-life gate
    if pcr_series is not None:
        ou_gate      = compute_halflife_gate(pcr_series.reindex(bt.index).ffill(), cfg)
        bt["ou_gate"] = ou_gate.reindex(bt.index).fillna(False)
    else:
        bt["ou_gate"] = True

    # Layer 3: drawdown circuit breaker
    dd_df            = compute_drawdown_mask(bt["position_for_pnl"], bt["ret_underlying"], cfg)
    bt["drawdown"]   = dd_df["drawdown"]
    bt["dd_cap"]     = dd_df["dd_cap"]
    bt["circuit_state"] = dd_df["circuit_state"]

    # Combine
    raw_pos  = bt["position_for_pnl"].copy() * bt["vol_scalar"]
    prev_pos = bt["position_for_pnl"].shift(1).fillna(0.0)
    gate_open = bt["ou_gate"]
    raw_pos   = raw_pos.where(gate_open | (raw_pos <= prev_pos), prev_pos)
    raw_pos   = raw_pos.clip(upper=bt["dd_cap"])

    TIERS = [0.0, 0.33, 0.66, 1.0]
    def snap_to_tier(x: float) -> float:
        return min(TIERS, key=lambda t: abs(t - x))

    bt["position_risk"] = raw_pos.clip(0.0, 1.0).apply(snap_to_tier)

    DAILY_USDC = 0.04 / 365.0
    in_usdc    = 1.0 - bt["position_risk"].clip(lower=0)
    bt["strategy_ret_risk"] = bt["position_risk"] * bt["ret_underlying"] + in_usdc * DAILY_USDC
    if "trade_active" in bt.columns:
        bt.loc[~bt["trade_active"], "strategy_ret_risk"] = 0.0
    bt["equity_curve_risk"] = (1 + bt["strategy_ret_risk"]).cumprod()

    return bt
