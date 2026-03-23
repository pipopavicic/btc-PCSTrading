"""
Fetch BTC/USDT OHLCV price data from Binance via ccxt and cache locally as CSV.
Supports incremental updates without duplicating data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR     = Path(__file__).parent / "raw"
PRICE_CSV   = RAW_DIR / "btc_usdt_ohlcv.csv"
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1d"
BINANCE_LIMIT = 1000


def fetch_price_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    limit: int = BINANCE_LIMIT,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch BTC/USDT daily OHLCV from Binance, with local CSV caching."""
    cache_path = cache_path or PRICE_CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing  = _load_cached(cache_path)
    since_ms: Optional[int] = None

    if existing is not None and not existing.empty:
        last_ts  = existing.index[-1]
        since_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info("Cached data found. Fetching new candles since %s", last_ts)
    else:
        logger.info("No cache found. Fetching from 2018-01-01.")

    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        since    = exchange.parse8601("2018-01-01T00:00:00Z")
        all_ohlcv = []
        while True:
            batch = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since, limit=limit)
            if not batch:
                break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + 86_400_000
            if len(batch) < 1000:
                break
    except ccxt.NetworkError as exc:
        logger.error("Network error: %s", exc)
        if existing is not None:
            return existing
        raise
    except ccxt.ExchangeError as exc:
        logger.error("Exchange error: %s", exc)
        raise

    new_df = _ohlcv_to_dataframe(all_ohlcv)

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined[combined.index.notna()].drop_duplicates().sort_index()
    else:
        combined = new_df

    combined.to_csv(cache_path)
    logger.info("Saved %d rows to %s", len(combined), cache_path)
    return combined


def _ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def _load_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as exc:
        logger.warning("Could not read cache file %s: %s", path, exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = fetch_price_data()
    print(df.tail())
