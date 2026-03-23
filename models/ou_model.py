"""
Ornstein-Uhlenbeck model for PCR spread.
Calibrates mean-reversion parameters from historical data and derives
statistically optimal entry/exit thresholds.

Reference: Gottschalk (2022), QFE — cointegration of BTC price and cost.
arXiv companion: Hayes (2018) 1805.07610
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OUParameters:
    """Calibrated Ornstein-Uhlenbeck parameters."""
    theta: float   # mean reversion speed (per day)
    mu:    float   # long-run mean of log(PCR)
    sigma: float   # diffusion (volatility)
    half_life_days: float   # ln(2)/theta — time to halve a deviation

    def entry_threshold(self, k: float = 1.5) -> float:
        """Log-PCR below which we BUY (k std devs below mean)."""
        return self.mu - k * self.sigma_eq

    def exit_threshold(self, k: float = 1.0) -> float:
        """Log-PCR above which we SELL (k std devs above mean)."""
        return self.mu + k * self.sigma_eq

    @property
    def sigma_eq(self) -> float:
        """Equilibrium std dev of the OU process: σ / sqrt(2θ)."""
        return self.sigma / np.sqrt(2 * self.theta) if self.theta > 0 else self.sigma


def fit_ou_parameters(pcr_series: pd.Series) -> OUParameters:
    """
    Fit OU parameters to log-PCR series using Maximum Likelihood Estimation.

    Uses the discrete-time AR(1) representation:
        S_t = alpha + beta * S_{t-1} + epsilon_t

    Maps to OU params:
        theta = -ln(beta) / dt
        mu    = alpha / (1 - beta)
        sigma = std(epsilon) * sqrt(-2 ln(beta) / (dt*(1-beta^2)))
    """
    log_pcr = np.log(pcr_series.dropna())
    log_pcr = log_pcr[np.isfinite(log_pcr)]

    S   = log_pcr.values[:-1]   # S_{t-1}
    S1  = log_pcr.values[1:]    # S_t
    dt  = 1.0                   # daily data → dt = 1 day

    # OLS regression: S_t = alpha + beta * S_{t-1}
    n     = len(S)
    beta  = (n * np.dot(S, S1) - S.sum() * S1.sum()) / \
            (n * np.dot(S, S) - S.sum()**2)
    alpha = (S1.mean() - beta * S.mean())

    # Guard: beta must be in (0, 1) for mean reversion
    beta  = np.clip(beta, 1e-6, 1 - 1e-6)

    theta = -np.log(beta) / dt
    mu    = alpha / (1 - beta)

    residuals = S1 - (alpha + beta * S)
    sigma_res = residuals.std(ddof=2)
    sigma     = sigma_res * np.sqrt(-2 * np.log(beta) / (dt * (1 - beta**2)))
    half_life = np.log(2) / theta

    params = OUParameters(
        theta=round(theta, 6),
        mu=round(mu, 6),
        sigma=round(sigma, 6),
        half_life_days=round(half_life, 1),
    )

    logger.info(
        "OU fit: θ=%.4f  μ=%.4f  σ=%.4f  half-life=%.1f days",
        params.theta, params.mu, params.sigma, params.half_life_days,
    )
    return params


def generate_ou_signals(
    pcr_series: pd.Series,
    params: OUParameters,
    entry_k: float = 1.5,
    exit_k:  float = 1.0,
) -> pd.DataFrame:
    """
    Generate entry/exit signals using OU-calibrated thresholds.

    Args:
        pcr_series: Raw PCR series (not log).
        params: Fitted OUParameters.
        entry_k: Std deviations below mean to trigger BUY.
        exit_k:  Std deviations above mean to trigger SELL.

    Returns:
        DataFrame with columns: log_pcr, z_score, entry_signal, exit_signal
    """
    log_pcr = np.log(pcr_series.clip(lower=1e-6))

    # Z-score: how many equilibrium std devs from the long-run mean
    z_score = (log_pcr - params.mu) / params.sigma_eq

    entry = z_score < -entry_k    # deeply below mean → BUY
    exit_  = z_score >  exit_k    # above mean → SELL

    df = pd.DataFrame({
        "log_pcr":      log_pcr,
        "z_score":      z_score,
        "entry_signal": entry,
        "exit_signal":  exit_,
    }, index=pcr_series.index)

    logger.info(
        "OU signals: entry_k=%.1f (PCR < %.3f)  exit_k=%.1f (PCR > %.3f)  "
        "Entries: %d  Exits: %d",
        entry_k, np.exp(params.entry_threshold(entry_k)),
        exit_k,  np.exp(params.exit_threshold(exit_k)),
        entry.sum(), exit_.sum(),
    )
    return df


def cointegration_test(price: pd.Series, cost: pd.Series) -> dict:
    """
    Johansen cointegration test between BTC price and production cost.
    Formally tests the statistical assumption underlying the strategy.

    Reference: Gottschalk (2022) doi:10.3934/QFE.2022030
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        import warnings
        warnings.filterwarnings("ignore")

        df = pd.DataFrame({"price": np.log(price), "cost": np.log(cost)}).dropna()
        result = coint_johansen(df, det_order=0, k_ar_diff=1)

        # Trace statistic vs 95% critical value
        trace_stat  = result.lr1[0]
        crit_95     = result.cvt[0, 1]
        cointegrated = trace_stat > crit_95

        logger.info(
            "Johansen test: trace=%.2f  crit95=%.2f  cointegrated=%s",
            trace_stat, crit_95, cointegrated,
        )
        return {
            "cointegrated":  cointegrated,
            "trace_statistic": round(trace_stat, 3),
            "critical_value_95": round(crit_95, 3),
            "interpretation": (
                "✅ Price and cost are cointegrated — strategy has statistical foundation"
                if cointegrated else
                "⚠️  Cointegration not confirmed at 95% — verify data range"
            ),
        }
    except ImportError:
        logger.warning("statsmodels not installed. Run: pip install statsmodels")
        return {"cointegrated": None, "error": "statsmodels required"}
    
def fit_ou_rolling(
    pcr_series: pd.Series,
    min_window: int = 252,          # minimum 1 year of data before first fit
) -> pd.DataFrame:
    """
    Fit OU parameters on an expanding window — no look-ahead bias.
    At each date t, only uses data available up to and including t.

    Returns DataFrame with columns:
        theta, mu, sigma, sigma_eq, half_life,
        entry_threshold_15, exit_threshold_10
    indexed by date.
    """
    records = []
    dates   = pcr_series.index

    for i in range(min_window, len(pcr_series)):
        slice_  = pcr_series.iloc[:i]
        try:
            params = fit_ou_parameters(slice_)
            records.append({
                "date":               dates[i],
                "theta":              params.theta,
                "mu":                 params.mu,
                "sigma":              params.sigma,
                "sigma_eq":           params.sigma_eq,
                "half_life":          params.half_life_days,
                "entry_pcr_15":       np.exp(params.entry_threshold(1.5)),
                "exit_pcr_10":        np.exp(params.exit_threshold(1.0)),
            })
        except Exception:
            continue

    df = pd.DataFrame(records).set_index("date")
    logger.info(
        "Rolling OU calibration complete. "
        "First date: %s  Last date: %s  Rows: %d",
        df.index[0].date(), df.index[-1].date(), len(df),
    )
    return df