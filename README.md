# BTC Production Cost Strategy

A professional, modular Python quant-research repository that implements a
Bitcoin trading strategy based on the **mining production cost** as a dynamic
fundamental price floor.

---

## Strategy Thesis

Bitcoin miners must sell BTC at or above their break-even cost to remain
solvent.  When market price falls toward (or below) the production cost, miner
capitulation increases selling pressure in the short term but simultaneously
reduces future supply—ultimately establishing a fundamental price floor.

This repository operationalises that hypothesis by:

1. Estimating the dynamic USD/BTC production cost from live network hashrate
   data.
2. Constructing the **Price-to-Cost Ratio (PCR)** — a signal that identifies
   extreme undervaluation and overvaluation relative to mining economics.
3. Confirming signals with an RSI filter to avoid entering during continued
   down-trends.

---

## Production Cost Methodology

The production cost is estimated using the formula originally proposed by
Hayes (2018):

```
Cost ($/BTC) = (fleet_efficiency [J/TH]
                × energy_rate    [$/kWh]
                × hashrate       [TH/s])
               / (block_reward [BTC] × blocks_per_hour × 1000)
```

### Calibrated Constants

| Parameter          | Value      | Notes                              |
|--------------------|------------|------------------------------------|
| `fleet_efficiency` | 25 J/TH    | Average ASIC efficiency (2024)     |
| `energy_rate`      | 0.06 $/kWh | Global average electricity rate    |
| `block_reward`     | 3.125 BTC  | Post-April 2024 halving            |
| `blocks_per_hour`  | 6          | ~10-minute block time              |

A **30-day rolling mean** is applied to smooth short-term hashrate volatility.

Network hashrate is fetched dynamically from the Blockchain.info public API
(no API key required).

---

## Signal Logic

### Price-to-Cost Ratio (PCR)

```
PCR = market_price / production_cost_smooth
```

| PCR Range         | Signal Zone   |
|-------------------|---------------|
| < 0.9             | STRONG BUY    |
| 0.9 – 1.1         | WEAK BUY      |
| 1.1 – 2.0         | HOLD          |
| 2.0 – 3.5         | SELL          |
| ≥ 3.5             | STRONG SELL   |

### RSI Confirmation Filter (RSI-14)

| Condition        | Required for             |
|------------------|--------------------------|
| RSI < 45         | Entry (Buy) confirmation |
| RSI > 55         | Exit (Sell) confirmation |

### Backtest Entry / Exit Rules

```python
entries = (pcr < 0.9) & (rsi < 45)
exits   = (pcr > 2.0) & (rsi > 55)
```

---

## Project Structure

```
btc-PCSTrading/
│
├── data/
│   ├── fetch_price.py      # BTC/USDT OHLCV via ccxt (cached CSV)
│   ├── fetch_onchain.py    # Hashrate & difficulty from blockchain.info
│   └── raw/                # Auto-created CSV cache directory
│
├── models/
│   └── production_cost.py  # Production cost estimation + smoothing
│
├── signals/
│   └── signal_engine.py    # PCR signal zones + RSI confirmation
│
├── backtest/
│   └── run_backtest.py     # vectorbt end-to-end backtest
│
├── risk/
│   └── risk_manager.py     # Three-layer risk management
│
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory data analysis
│   ├── 02_Backtest.ipynb   # Interactive backtest walkthrough
│   └── 03_RiskAnalysis.ipynb # Drawdown, VaR, stress tests
│
├── README.md
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/pipopavicic/btc-PCSTrading.git
cd btc-PCSTrading

# 2. Create and activate a virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backtest
python backtest/run_backtest.py
```

Price data and on-chain metrics are fetched automatically on first run and
cached in `data/raw/`.  Subsequent runs only fetch new data.

---

## Backtest Results

*(Results are generated automatically when you run `python backtest/run_backtest.py`)*

| Metric          | Value  |
|-----------------|--------|
| Sharpe Ratio    | TBD    |
| Max Drawdown    | TBD    |
| Calmar Ratio    | TBD    |
| Win Rate        | TBD    |
| Total Return    | TBD    |

An equity-curve PNG is saved to `backtest/equity_curve.png`.

---

## Risk Management

### Trade Level
- Maximum position size: **10% of portfolio value** per trade.

### Position Level
- ATR-based trailing stop:  
  `stop_price = entry_price − 2 × ATR(14)`

### Portfolio Level
- **Minimum 25% USDC cash buffer** maintained at all times.
- **Rolling 30-day 99% VaR** (parametric, normal distribution):  
  `VaR = position_size × σ_30d × 2.33`
- **Circuit breaker**: If the portfolio drops **≥ 20%** from its rolling
  21-day (≈ monthly) peak, all new entries are halted until the next month.

---

## Notebooks

| Notebook               | Content                                          |
|------------------------|--------------------------------------------------|
| `01_EDA.ipynb`         | BTC price vs production cost, PCR visualisation  |
| `02_Backtest.ipynb`    | Full vectorbt backtest, strategy vs Buy-and-Hold |
| `03_RiskAnalysis.ipynb`| Drawdown analysis, VaR, 70% crash stress test    |

---

## Limitations & Future Improvements

| Limitation                              | Suggested Improvement                      |
|-----------------------------------------|--------------------------------------------|
| Single fleet efficiency constant        | Model fleet composition dynamically        |
| Parametric (normal) VaR                 | Replace with historical or Monte-Carlo VaR |
| Long-only strategy                      | Add short-side for overvaluation regimes   |
| No on-chain sentiment data              | Integrate SOPR, MVRV, NVT ratio            |
| Fixed block reward                      | Auto-detect halving epochs                 |
| Single exchange (Binance)               | Aggregate multi-exchange order flow        |

---

## References

- Hayes, A. (2018). *Bitcoin Price and its Marginal Cost of Production: Support
  for a Fundamental Value*. arXiv:1805.07610.
  <https://arxiv.org/abs/1805.07610>
- Blockchain.info Charts API: <https://www.blockchain.com/explorer/charts>
- vectorbt documentation: <https://vectorbt.dev/>
- pandas-ta documentation: <https://github.com/twopirllc/pandas-ta>

---

## License

MIT License — see [LICENSE](LICENSE) for details.