"""
visualize.py — PCR chart + equity curves for multiple start periods.
Run after run_backtest.py has populated data/raw/ with CSVs.
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.production_cost import compute_production_cost
from data.fetch_onchain import fetch_hashrate
from signals.signal_engine import compute_signals

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────
price_df = pd.read_csv(RAW / "btc_usdt_ohlcv.csv", index_col=0, parse_dates=True)
price_df.index = pd.to_datetime(price_df.index, utc=True)

hr_df = fetch_hashrate()
cost_df = compute_production_cost(hr_df)
signals_df = compute_signals(price_df, cost_df)

# Merge everything on date
df = price_df[["close"]].join(cost_df[["production_cost_smooth"]], how="inner")
df = df.join(signals_df[["pcr", "rsi", "signal_zone", "entry_signal", "exit_signal"]], how="inner")
df.columns = ["price", "cost", "pcr", "rsi", "signal_zone", "entry_signal", "exit_signal"]
df = df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Full PCR History with signal zones
# ══════════════════════════════════════════════════════════════════════════════
def plot_pcr_full(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 2, 1.2]})
    fig.suptitle("BTC Production Cost Strategy — Full History", fontsize=15, fontweight="bold")

    # ── Panel 1: Price vs Cost ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df.index, df["price"], color="#F7931A", linewidth=1.5, label="BTC Price (USDT)")
    ax1.plot(df.index, df["cost"],  color="#2196F3", linewidth=1.5, linestyle="--", label="Production Cost (smoothed 30d)")
    ax1.fill_between(df.index, df["price"], df["cost"],
                     where=df["price"] < df["cost"], alpha=0.15, color="green", label="Price < Cost (BUY zone)")
    ax1.fill_between(df.index, df["price"], df["cost"],
                     where=df["price"] > df["cost"] * 1.4, alpha=0.1, color="red", label="Price > 1.4× Cost (SELL zone)")

    # Mark buy/sell signals
    buys  = df[df["entry_signal"] == True]
    sells = df[df["exit_signal"]  == True]
    ax1.scatter(buys.index,  buys["price"],  marker="^", color="green", s=80, zorder=5, label="BUY signal")
    ax1.scatter(sells.index, sells["price"], marker="v", color="red",   s=80, zorder=5, label="SELL signal")

    ax1.set_ylabel("USD / BTC", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(alpha=0.3)

    # ── Panel 2: PCR with horizontal zone bands ─────────────────────────────
    ax2 = axes[1]
    ax2.plot(df.index, df["pcr"], color="#9C27B0", linewidth=1.4, label="PCR (Price / Cost)")
    ax2.axhline(0.9,  color="green",  linestyle=":",  linewidth=1.2, label="Strong BUY (0.9)")
    ax2.axhline(1.1,  color="gray",   linestyle=":",  linewidth=1.0, label="Fair Value (1.1)")
    ax2.axhline(1.4,  color="orange", linestyle="--", linewidth=1.2, label="SELL (1.4)")
    ax2.axhline(1.8,  color="red",    linestyle="--", linewidth=1.2, label="Strong SELL (1.8)")
    # Colored background zones on PCR panel
    ax2.axhspan(0,    0.9, alpha=0.07, color="green",  label="STRONG_BUY zone")
    ax2.axhspan(0.9,  1.1, alpha=0.05, color="lime",   label="WEAK_BUY zone")
    ax2.axhspan(1.1,  1.4, alpha=0.03, color="gray",   label="HOLD zone")
    ax2.axhspan(1.4,  1.8, alpha=0.07, color="orange", label="SELL zone")
    ax2.axhspan(1.8,  df["pcr"].max() + 0.5,
                       alpha=0.07, color="red",    label="STRONG_SELL zone")

    ax2.fill_between(df.index, 0, df["pcr"],
                     where=df["pcr"] < 0.9,  alpha=0.15, color="green")
    ax2.fill_between(df.index, 0, df["pcr"],
                     where=df["pcr"] > 1.4,  alpha=0.10, color="red")
    ax2.set_ylabel("PCR", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: RSI ────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(df.index, df["rsi"], color="#FF9800", linewidth=1.2, label="RSI (14)")
    ax3.axhline(45, color="green", linestyle=":", linewidth=1.0, label="Buy filter (45)")
    ax3.axhline(55, color="red",   linestyle=":", linewidth=1.0, label="Sell filter (55)")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", save_path)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Multi-period PCR panels (2020, 2022, 2024 starts)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pcr_multiperiod(df: pd.DataFrame, save_path: Path) -> None:
    periods = {
        "From 2020 (Full Cycle)":   "2020-01-01",
        "From 2022 (Bear Market)":  "2022-01-01",
        "From 2024 (Post-Halving)": "2024-01-01",
    }

    fig, axes = plt.subplots(len(periods), 1, figsize=(16, 14))
    fig.suptitle("PCR by Time Period — BTC Production Cost Strategy", fontsize=14, fontweight="bold")

    for ax, (title, start) in zip(axes, periods.items()):
        subset = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if subset.empty:
            ax.set_title(f"{title} — no data available", fontsize=10)
            continue

        # Price and cost
        ax2 = ax.twinx()
        ax2.plot(subset.index, subset["price"], color="#F7931A", linewidth=1.0,
                 alpha=0.4, label="BTC Price")
        ax2.plot(subset.index, subset["cost"],  color="#2196F3", linewidth=1.0,
                 alpha=0.4, linestyle="--", label="Prod. Cost")
        ax2.set_ylabel("USD", fontsize=8, color="gray")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

        # PCR on primary axis
        ax.plot(subset.index, subset["pcr"], color="#9C27B0", linewidth=1.5, label="PCR", zorder=3)
        ax.axhline(0.9,  color="green",  linestyle=":", linewidth=1.0)
        ax.axhline(1.4,  color="orange", linestyle="--", linewidth=1.0)
        ax.axhline(1.8,  color="red",    linestyle="--", linewidth=1.0)
        ax.fill_between(subset.index, 0, subset["pcr"],
                        where=subset["pcr"] < 0.9,  alpha=0.2, color="green")
        ax.fill_between(subset.index, 0, subset["pcr"],
                        where=subset["pcr"] > 1.4,  alpha=0.15, color="red")

        # Signal markers
        buys  = subset[subset["entry_signal"] == True]
        sells = subset[subset["exit_signal"]  == True]
        ax.scatter(buys.index,  buys["pcr"],  marker="^", color="green", s=60, zorder=5)
        ax.scatter(sells.index, sells["pcr"], marker="v", color="red",   s=60, zorder=5)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_ylabel("PCR", fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", save_path)
    plt.close()


# ── Run both charts ────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_pcr_full(df,           OUT / "pcr_full_history.png")
    plot_pcr_multiperiod(df,    OUT / "pcr_multiperiod.png")
    logger.info("All charts generated in %s", OUT)
