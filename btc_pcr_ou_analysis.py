"""
BTC price vs production cost: cointegration and OU analysis.

Runs Johansen cointegration and OU fits across multiple time windows
for three cost variants (raw 0d, 7d smooth, 50d smooth).

Run:
    python btc_pcr_ou_analysis.py

Outputs:
    data/derived/inspect/btc_pcr_ou_summary_windows.csv
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from data.fetch_onchain import fetch_hashrate
from data.fetch_price import fetch_price_data
from models.production_cost_v2 import compute_production_cost_dynamic_v2
from models.ou_model import fit_ou_parameters, cointegration_test

START_DATE: Optional[str] = "2022-01-01"
END_DATE:   Optional[str] = None

WINDOWS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "full_sample":     (None, None),
    "year_2022":       ("2022-01-01", "2022-12-31"),
    "year_2023":       ("2023-01-01", "2023-12-31"),
    "post_China_ban":  ("2021-06-01", None),
    "post_2022_regime":("2022-01-01", None),
}

OUTPUT_INSPECT_DIR = Path("data/derived/inspect")
OUTPUT_INSPECT_DIR.mkdir(parents=True, exist_ok=True)


def load_price_and_cost() -> pd.DataFrame:
    hr   = fetch_hashrate()
    cost = compute_production_cost_dynamic_v2(hr)[["production_cost", "production_cost_7d"]].copy()
    if cost.index.tz is None:
        cost.index = cost.index.tz_localize("UTC")
    else:
        cost.index = cost.index.tz_convert("UTC")
    cost = cost[~cost.index.duplicated(keep="last")]

    price_df = fetch_price_data()[["close"]].rename(columns={"close": "price_usd"})
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize("UTC")
    else:
        price_df.index = price_df.index.tz_convert("UTC")
    price_df = price_df[~price_df.index.duplicated(keep="last")]

    df = pd.concat([price_df, cost], axis=1).sort_index().dropna()
    if START_DATE:
        df = df[df.index >= pd.to_datetime(START_DATE).tz_localize("UTC")]
    logger.info("Loaded %d rows from %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def build_cost_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_0d"]  = out["production_cost"]
    out["cost_7d"]  = out["production_cost_7d"]
    out["cost_50d"] = out["production_cost"].rolling(50, min_periods=10).mean()
    for col in ["cost_0d", "cost_7d", "cost_50d"]:
        out[f"pcr_{col}"] = out["price_usd"] / out[col]
    return out


def analyze_windows(df_all: pd.DataFrame) -> pd.DataFrame:
    records: List[dict] = []

    for win_name, (start, end) in WINDOWS.items():
        df_win = df_all.copy()
        if start:
            df_win = df_win[df_win.index >= pd.to_datetime(start).tz_localize("UTC")]
        if end:
            df_win = df_win[df_win.index <= pd.to_datetime(end).tz_localize("UTC")]
        if len(df_win) < 60:
            continue

        for cost_col in ["cost_0d", "cost_7d", "cost_50d"]:
            try:
                coint_res = cointegration_test(df_win["price_usd"], df_win[cost_col])
                ou_params = fit_ou_parameters(df_win[f"pcr_{cost_col}"])
                records.append({
                    "window":       win_name,
                    "cost_variant": cost_col,
                    "n_obs":        len(df_win),
                    **coint_res,
                    "ou_theta":     ou_params.theta,
                    "ou_mu":        ou_params.mu,
                    "ou_sigma_eq":  ou_params.sigma_eq,
                    "ou_halflife":  ou_params.half_life_days,
                })
            except Exception as exc:
                logger.warning("Window %s cost %s failed: %s", win_name, cost_col, exc)

    return pd.DataFrame(records)


def main() -> None:
    df_all = load_price_and_cost()
    df_all = build_cost_variants(df_all)
    summary = analyze_windows(df_all)
    out_path = OUTPUT_INSPECT_DIR / "btc_pcr_ou_summary_windows.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
