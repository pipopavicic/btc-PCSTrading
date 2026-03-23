from __future__ import annotations

"""
Rolling OU engine for BTC vs production-cost PCR trading.

Core idea
---------
1. Fundamental cost is a lagged version of smoothed production cost.
2. PCR = price / fundamental_cost measures how rich or cheap BTC is relative
   to that anchor.
3. log(PCR) is fit on a rolling window to estimate OU parameters:
   mean (mu), equilibrium volatility (sigma_eq), and mean-reversion speed.
4. Daily z-scores relative to the rolling OU equilibrium generate tiered,
   long-only BTC exposure between 0 and 1.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RollingOUConfig:
    lookback_cost_col: str = "production_cost_7d"
    cost_lag_days:     int   = 7
    min_fit_window:    int   = 252
    entry_z_1: float = 0.5
    entry_z_2: float = 1.0
    entry_z_3: float = 1.5
    exit_z_1:  float = 0.5
    exit_z_2:  float = 1.0
    exit_z_3:  float = 1.5
    step_size: float = 1.0 / 3.0
    long_only: bool  = True
    apy_usdc:  float = 0.04
    trading_start: str | None = "2022-01-01"


@dataclass
class OUParameters:
    theta: float
    mu:    float
    sigma: float
    half_life_days: float

    @property
    def sigma_eq(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.theta) if self.theta > 0 else self.sigma

    def entry_threshold(self, k: float) -> float:
        return self.mu - k * self.sigma_eq

    def exit_threshold(self, k: float) -> float:
        return self.mu + k * self.sigma_eq


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _clean_utc_index(s: pd.Series) -> pd.Series:
    s = s.copy().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex")
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s[~s.index.duplicated(keep="last")]
    return s


# ---------------------------------------------------------------------------
# OU fitting
# ---------------------------------------------------------------------------

def fit_ou_parameters(pcr_series: pd.Series) -> OUParameters:
    """Fit OU parameters to log(PCR) using the AR(1) OLS mapping."""
    log_pcr = np.log(pcr_series.dropna().clip(lower=1e-9))
    log_pcr = log_pcr[np.isfinite(log_pcr)]
    if len(log_pcr) < 20:
        raise ValueError("Not enough observations to fit OU parameters")

    s_tm1 = log_pcr.values[:-1]
    s_t   = log_pcr.values[1:]
    n     = len(s_tm1)

    denom = n * np.dot(s_tm1, s_tm1) - s_tm1.sum() ** 2
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate series for OU fit")

    beta  = (n * np.dot(s_tm1, s_t) - s_tm1.sum() * s_t.sum()) / denom
    alpha = s_t.mean() - beta * s_tm1.mean()
    beta  = np.clip(beta, 1e-6, 1 - 1e-6)

    theta = float(-np.log(beta))
    mu    = float(alpha / (1 - beta))

    resid     = s_t - (alpha + beta * s_tm1)
    sigma_res = float(resid.std(ddof=2))
    sigma     = float(sigma_res * np.sqrt(-2 * np.log(beta) / (1 - beta ** 2)))
    half_life = float(np.log(2) / theta)

    return OUParameters(theta=theta, mu=mu, sigma=sigma, half_life_days=half_life)


def fit_ou_rolling_window(pcr_series: pd.Series, window: int = 252) -> pd.DataFrame:
    pcr_series = _clean_utc_index(pcr_series).dropna()
    rows = []
    for i in range(window, len(pcr_series)):
        hist = pcr_series.iloc[i - window:i]
        try:
            params = fit_ou_parameters(hist)
        except Exception:
            continue
        sigma_eq = max(float(params.sigma_eq), 0.03)
        rows.append({
            "timestamp": pcr_series.index[i],
            "theta":     params.theta,
            "mu":        params.mu,
            "sigma":     params.sigma,
            "sigma_eq":  sigma_eq,
            "half_life": params.half_life_days,
        })
    if not rows:
        raise ValueError("No rolling OU fits produced; increase window or history length")
    return pd.DataFrame(rows).set_index("timestamp")


def fit_ou_expanding(pcr_series: pd.Series, min_window: int = 252) -> pd.DataFrame:
    """Expanding-window OU fit — strictly no look-ahead."""
    pcr_series = _clean_utc_index(pcr_series).dropna()
    rows = []
    for i in range(min_window, len(pcr_series)):
        hist = pcr_series.iloc[:i]
        try:
            params = fit_ou_parameters(hist)
        except Exception:
            continue
        rows.append({
            "timestamp":    pcr_series.index[i],
            "theta":        params.theta,
            "mu":           params.mu,
            "sigma":        params.sigma,
            "sigma_eq":     params.sigma_eq,
            "half_life":    params.half_life_days,
            "entry_pcr_05": np.exp(params.entry_threshold(0.5)),
            "entry_pcr_10": np.exp(params.entry_threshold(1.0)),
            "entry_pcr_15": np.exp(params.entry_threshold(1.5)),
            "exit_pcr_05":  np.exp(params.exit_threshold(0.5)),
            "exit_pcr_10":  np.exp(params.exit_threshold(1.0)),
            "exit_pcr_15":  np.exp(params.exit_threshold(1.5)),
        })
    if not rows:
        raise ValueError("No rolling OU fits produced; increase history length")
    return pd.DataFrame(rows).set_index("timestamp")


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _tiered_target_from_z(z: float, cfg: RollingOUConfig) -> float:
    s = cfg.step_size
    if z <= -cfg.entry_z_3:
        return round(3 * s, 6)
    if z <= -cfg.entry_z_2:
        return round(2 * s, 6)
    if z <= -cfg.entry_z_1:
        return round(1 * s, 6)
    if z >= cfg.exit_z_3:
        return 0.0
    if z >= cfg.exit_z_2:
        return round(1 * s, 6)
    if z >= cfg.exit_z_1:
        return round(2 * s, 6)
    return np.nan


def _move_one_step(current: float, target: float, step: float) -> float:
    if np.isnan(target):
        return current
    if target > current:
        return min(current + step, target, 1.0)
    if target < current:
        return max(current - step, target, 0.0)
    return current


def generate_rolling_ou_signals(
    price: pd.Series,
    cost:  pd.Series,
    cfg:   RollingOUConfig,
) -> pd.DataFrame:
    """
    Generate BTC/USDC allocation signals from a rolling OU fit of PCR.

    Returns a DataFrame with aligned price/cost/fundamental, PCR,
    rolling OU parameters, z-score, target weight, position, and signal flags.
    """
    price = _clean_utc_index(price.rename("price_usd"))
    cost  = _clean_utc_index(cost.rename("cost_usd"))

    df = pd.concat([price, cost], axis=1).dropna().sort_index()
    df["fundamental"] = df["cost_usd"].shift(cfg.cost_lag_days)
    df = df.dropna(subset=["fundamental"]).copy()

    df["pcr"]     = df["price_usd"] / df["fundamental"]
    df["log_pcr"] = np.log(df["pcr"].clip(lower=1e-9))

    rolling_ou = fit_ou_rolling_window(df["pcr"], window=cfg.min_fit_window)
    df = df.join(rolling_ou, how="left")
    df = df.dropna(subset=["mu", "sigma_eq"]).copy()

    df["z_score_raw"] = (df["log_pcr"] - df["mu"]) / df["sigma_eq"]
    df["z_score"]     = df["z_score_raw"].clip(-5, 5)

    df["target_position"] = df["z_score"].apply(
        lambda z: _tiered_target_from_z(float(z), cfg)
    )

    position = []
    current  = 0.0
    for ts, row in df.iterrows():
        if cfg.trading_start is not None and ts < pd.Timestamp(cfg.trading_start, tz="UTC"):
            current = 0.0
        else:
            current = _move_one_step(current, float(row["target_position"]), cfg.step_size)

        current = float(np.clip(current, 0.0, 1.0))
        if cfg.long_only:
            current = max(current, 0.0)
        current = round(current / cfg.step_size) * cfg.step_size
        current = float(np.clip(current, 0.0, 1.0))
        current = round(current, 6)
        position.append(current)

    df["position"]     = position
    df["position"]     = df["position"].round(3)
    df["entry_signal"] = df["position"].diff().fillna(df["position"]) > 0
    df["exit_signal"]  = df["position"].diff().fillna(0.0) < 0
    return df
