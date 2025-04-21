# Python Project Quickstart Guide

**Prerequisites**
- Python 3.7+
- pandas, numpy, openpyxl (for Excel I/O)
- Project files in the same directory:
  - `portfolio_construction.py`
  - `backtester.py`
  - `correlation_analyzer.py`
  - `currency_exposure.py`
  - `performance_attribution.py`
  - `run_main_analysis.py`

Install dependencies:

```bash
pip install pandas numpy openpyxl
```

---

## 1. Portfolio Construction (`portfolio_construction.py`)
Construct an equal‑volatility portfolio:

```python
from portfolio_construction import EqualVolatilityPortfolio
import pandas as pd

# Load your substrategy total return indices (TRI):
tri = pd.read_csv('substrat_tri.csv', index_col=0, parse_dates=True)

# Initialize:
pc = EqualVolatilityPortfolio(
    total_return_index=tri,
    lookback_years=2,
    skip_recent_month=False,
    rebalance_freq='M'
)

# Build daily weights:
weights = pc.construct_portfolio()
print(weights.head())
```

- **Args**:
  - `total_return_index`: DataFrame of TRI
  - `lookback_years`: int
  - `skip_recent_month`: bool
  - `rebalance_freq`: 'M'/'Q'/'A'/'W'
- **Output**: DataFrame of daily weights

---

## 2. Backtester (`backtester.py`)
Compute returns, indexes, and metrics:

```python
from backtester import Backtester

bt = Backtester(strategy_weights=weights, substrategy_total_return_index=tri)

# Daily returns:
daily_ret = bt.strategy_daily_returns

# Monthly returns:
monthly_ret = bt.strategy_monthly_returns

# Performance summary (daily):
perf = bt.calculate_performance_summary(
    periods={'Full Sample': None, 'Last 1Y': '1Y'},
    frequency='daily'
)
print(perf)

# Strategy index:
idx = bt.get_strategy_index(initial_value=100, frequency='daily')
```

---

## 3. Correlation Analysis (`correlation_analyzer.py`)
Rolling and fixed‑lookback correlations:

```python
from correlation_analyzer import CorrelationAnalyzer

ca = CorrelationAnalyzer(
    df_index=tri,
    rolling_period_years=1.0,
    fixed_lookback_config=[1, 3, 5]
)
results = ca.analyze()
rolling_corr = results['rolling']
fixed_corrs = results['fixed']
```

---

## 4. Currency Exposure (`currency_exposure.py`)
Aggregate substrategy currency weights:

```python
from currency_exposure import CurrencyExposureCalculator

cec = CurrencyExposureCalculator(
    top_level_weights=weights,
    substrat_weight_files_map={
        'StratA': 'substrat_A_weights.xlsx',
        'StratB': 'substrat_B_weights.xlsx'
    }
)
exposures = cec.calculate_exposures()
print(exposures.head())
```

---

## 5. Performance Attribution (`performance_attribution.py`)
Break down P&L by substrategy and currency:

```python
from performance_attribution import PerformanceAttribution

pa = PerformanceAttribution(
    top_level_weights=weights,
    substrat_tri=tri,
    substrat_attrib_files_map={
        'StratA': 'subA_attrib.xlsx',
        'StratB': 'subB_attrib.xlsx'
    },
    currency_classification_map={
        'AUD': 'DM', 'BRL': 'EM', ...
    }
)
attrib = pa.analyze_attribution(
    periods={'Full Sample': None, 'Last 1Y': '1Y'}
)
print(attrib['Substrategy'])
```

---

## 6. Main Script (`run_main_analysis.py`)
Edit the top configuration (file paths, params), then run:

```bash
python run_main_analysis.py
```

This executes portfolio construction, backtesting, correlation analysis, performance attribution, currency exposure, and writes `strategy_analysis_report.xlsx`.

---

*Excluded modules*:  
- `portfolio_construction_testing.py` (unit tests)  
- `simulation.py` (simulation & data generation)
