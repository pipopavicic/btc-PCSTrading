# btc-PCSTrading

A quantitative BTC/USDC allocation strategy grounded in the **Ornstein–Uhlenbeck mean-reversion** of Bitcoin's *Price-to-Cost Ratio* (PCR).

The core hypothesis: BTC price co-integrates with its marginal production cost. When the PCR diverges significantly from its rolling OU equilibrium, the strategy increases BTC exposure and earns USDC carry while under-allocated.

PYTHON VERSION - Python 3.10.11
---

## Strategy overview

| Component | Detail |
|---|---|
| **Fundamental anchor** | Marginal production cost — Hayes (2018) formula with Gottschalk (2022) time-varying efficiency & electricity inputs |
| **Signal** | Rolling OU z-score of log(PCR); tiered long-only exposure at \|z\| = 0.75 / 1.25 / 1.75 σ |
| **Position sizing** | Three tiers: 1/3, 2/3, 3/3 BTC; remainder earns 4% APY USDC carry |
| **Risk overlay** | Vol scalar (target 80% ann. vol) · OU half-life gate (>120d blocks new buys) · Drawdown circuit breaker (−15/−20/−30% thresholds) |
| **Backtest window** | 2022-01-01 → present (252-day OU warm-up from 2021) |

---

## Repo structure

```
.
├── backtest_7d.py            # Backtest: 7-day smoothed cost + 7-day lag
├── backtest_50d.py           # Backtest: 50-day smoothed cost + 50-day lag + risk overlay
├── parameter_sweep.py        # Grid search over cost variant / lag / window / thresholds
├── btc_pcr_ou_analysis.py    # Cointegration + OU analysis across time windows
│
├── models/
│   ├── ou_model.py           # OU parameter fitting (AR1 OLS) + Johansen cointegration
│   ├── engine_rolling_ou.py  # Rolling OU engine: signal generation & tiered position logic
│   ├── production_cost_v2.py # BTC production cost model (Hayes 2018 + Gottschalk 2022)
│   └── risk.py               # Three-layer risk overlay (vol / OU gate / circuit breaker)
│
└── data/
    ├── fetch_price.py        # BTC/USDT OHLCV from Binance via ccxt (cached CSV)
    └── fetch_onchain.py      # Network hashrate from blockchain.info (cached CSV)
```

---

## Quick start

```bash
pip install -r requirements.txt

# Run the primary backtest (50d smoothing + risk overlay)
python backtest_50d.py

# Run the 7-day variant
python backtest_7d.py

# Grid search
python parameter_sweep.py

# Cointegration analysis
python btc_pcr_ou_analysis.py
```

All scripts auto-download and cache data on first run (internet required). Subsequent runs load from `data/raw/`.

---

## References

- Hayes, A. (2018). *Bitcoin price and its marginal cost of production.* arXiv:1805.07610  
- Gottschalk, K. (2022). *Digital currency price formation: A production cost perspective.* QFE 6(4): 669–695. DOI:10.3934/QFE.2022030
