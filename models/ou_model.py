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
    theta: float  # mean reversion speed (per day)
    mu: float     # long-run mean of log(PCR)
    sigma: float  # diffusion (volatility)
    half_life_days: float  # ln(2)/theta — time to halve a deviation

    def entry_threshold(self, k: float = 1.5) -> float:
        """Log-PCR below which we BUY (k std devs below mean)."""
        return self.mu - k * self.sigma_eq

    def exit_threshold(self, k: float = 1.0) -> float:
        """Log-PCR above which we SELL (k std devs above mean)."""
        return self.mu + k * self.sigma_eq

    @property
    def sigma_eq(self) -> float:
        """Equilibrium std dev of the OU process: sigma / sqrt(2*theta)."""
        return self.sigma / np.sqrt(2 * self.theta) if self.theta > 0 else self.sigma


def fit_ou_parameters(pcr_series: pd.Series) -> OUParameters:
    """
    Fit OU parameters to log-PCR series using OLS AR(1) mapping.
    If S_t = alpha + beta*S_{t-1} + eps_t, then:
        theta = -ln(beta)
        mu    = alpha / (1 - beta)
        sigma = sigma_eps * sqrt(-2*ln(beta) / (1 - beta^2))
    """
    log_pcr = np.log(pcr_series.dropna().clip(lower=1e-9))
    log_pcr = log_pcr[np.isfinite(log_pcr)]
    if len(log_pcr) < 20:
        raise ValueError("Not enough observations to fit OU parameters")

    s_tm1 = log_pcr.values[:-1]
    s_t   = log_pcr.values[1:]
    n = len(s_tm1)

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


def cointegration_test(price: pd.Series, cost: pd.Series) -> dict:
    """
    Johansen cointegration test between BTC price and production cost.
    Returns a dict with trace stat, critical values, and a boolean flag.
    Requires statsmodels.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        raise ImportError("statsmodels is required for cointegration_test")

    data = pd.concat([price, cost], axis=1).dropna()
    result = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    trace_stat   = float(result.lr1[0])
    crit_90      = float(result.cvt[0, 0])
    crit_95      = float(result.cvt[0, 1])
    crit_99      = float(result.cvt[0, 2])
    cointegrated = trace_stat > crit_95

    return {
        "trace_stat":   trace_stat,
        "crit_90":      crit_90,
        "crit_95":      crit_95,
        "crit_99":      crit_99,
        "cointegrated": cointegrated,
    }
