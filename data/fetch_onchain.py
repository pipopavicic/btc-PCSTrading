"""
Fetch Bitcoin on-chain metrics (hashrate, difficulty) from blockchain.info.

Data is cached locally as CSV files inside data/raw/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent / "raw"
HASHRATE_CSV  = RAW_DIR / "btc_hashrate.csv"
DIFFICULTY_CSV = RAW_DIR / "btc_difficulty.csv"

BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info/charts"
REQUEST_TIMEOUT      = 30


def fetch_hashrate(
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch BTC network hashrate (EH/s) from blockchain.info with local CSV cache."""
    cache_path = cache_path or HASHRATE_CSV
    return _fetch_chart("hash-rate", cache_path, "hashrate", force_refresh)


def fetch_difficulty(
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch BTC mining difficulty from blockchain.info with local CSV cache."""
    cache_path = cache_path or DIFFICULTY_CSV
    return _fetch_chart("difficulty", cache_path, "difficulty", force_refresh)


def _fetch_chart(chart: str, cache_path: Path, column_name: str, force_refresh: bool) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and cache_path.exists():
        logger.info("Loading %s from cache: %s", chart, cache_path)
        return _load_csv(cache_path)

    url    = f"{BLOCKCHAIN_INFO_BASE}/{chart}"
    params = {"format": "json", "timespan": "all"}
    logger.info("Fetching %s from %s", chart, url)

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", chart, exc)
        if cache_path.exists():
            logger.warning("Returning stale cached data for %s.", chart)
            return _load_csv(cache_path)
        raise

    data = response.json()
    df   = _parse_blockchain_info(data, column_name)

    if column_name == "hashrate":
        # blockchain.info units changed from GH/s to TH/s in late 2023
        df[column_name] = df[column_name] / 1e6  # → EH/s

    df.to_csv(cache_path)
    logger.info("Saved %d rows to %s", len(df), cache_path)
    return df


def _parse_blockchain_info(data: dict, column_name: str) -> pd.DataFrame:
    values  = data.get("values", [])
    records = [{"timestamp": v["x"], column_name: v["y"]} for v in values]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("timestamp").sort_index()


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    hr = fetch_hashrate()
    print("Hashrate (EH/s):")
    print(hr.tail())
    diff = fetch_difficulty()
    print("\nDifficulty:")
    print(diff.tail())
