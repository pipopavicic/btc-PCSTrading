"""
Fetch BTC/USDT OHLCV price data from Binance via ccxt and cache locally as CSV.

Supports incremental updates without duplicating data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).parent / "raw"
PRICE_CSV = RAW_DIR / "btc_usdt_ohlcv.csv"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
BINANCE_LIMIT = 1000  # max candles per ccxt request


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_price_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    limit: int = BINANCE_LIMIT,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch BTC/USDT OHLCV data from Binance, with local CSV caching.

    On first run, downloads up to `limit` candles.  On subsequent runs,
    only new candles are fetched and appended to the existing file.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        timeframe: Candle interval, e.g. ``"1d"``.
        limit: Maximum number of candles to fetch from the exchange.
        cache_path: Override the default CSV cache location.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
        indexed by timestamp (UTC, timezone-aware).
    """
    cache_path = cache_path or PRICE_CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = _load_cached(cache_path)
    since_ms: Optional[int] = None

    if existing is not None and not existing.empty:
        last_ts = existing.index[-1]
        since_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info("Cached data found.  Fetching new candles since %s", last_ts)
    else:
        logger.info("No cache found.  Fetching up to %d candles.", limit)

    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    except ccxt.NetworkError as exc:
        logger.error("Network error while fetching price data: %s", exc)
        if existing is not None:
            logger.warning("Returning cached data.")
            return existing
        raise
    except ccxt.ExchangeError as exc:
        logger.error("Exchange error: %s", exc)
        raise

    new_df = _ohlcv_to_dataframe(raw)

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df]).drop_duplicates().sort_index()
    else:
        combined = new_df

    combined.to_csv(cache_path)
    logger.info("Saved %d rows to %s", len(combined), cache_path)
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    """Convert a raw ccxt OHLCV list to a pandas DataFrame.

    Args:
        raw: List of ``[timestamp_ms, open, high, low, close, volume]`` rows.

    Returns:
        DataFrame indexed by UTC-aware datetime, columns:
        [open, high, low, close, volume].
    """
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def _load_cached(path: Path) -> Optional[pd.DataFrame]:
    """Load cached OHLCV CSV if it exists.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not read cache file %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = fetch_price_data()
    print(df.tail())
