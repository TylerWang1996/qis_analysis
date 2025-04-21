## Quickstart Guide: Portfolio Analysis Toolkit

This guide provides a walkthrough on how to use the provided Python modules for quantitative portfolio analysis:

1.  **`portfolio_construction`**: Generate portfolio weights (e.g., Equal Volatility).
2.  **`backtester`**: Calculate strategy performance and metrics.
3.  **`performance_attribution`**: Analyze performance contributions by substrategy and currency.

### Prerequisites

* **Libraries:** Ensure you have `pandas` and `numpy` installed. You'll also need `openpyxl` if your attribution data is in `.xlsx` format.
    ```bash
    pip install pandas numpy openpyxl
    ```
* **Project Modules:** Make sure the Python files (`portfolio_construction.py`, `backtester.py`, `performance_attribution.py`, etc.) are in your Python path or the same working directory.
* **Data:**
    * **Substrategy TRI:** A pandas DataFrame (`substrat_tri`) with a `DatetimeIndex` (sorted ascending, business days preferred) and columns representing the Total Return Index for each substrategy. No NaNs allowed (or handle them before passing).
    * **Attribution Files (for `PerformanceAttribution`):** Separate Excel files for each substrategy. Each file must contain a sheet named `'attrib'` where the index is the date (sparse, only on rebalance days) and columns are currencies (e.g., 'AUD', 'EUR', 'USD', 'BRL'). Values should be the dollar P&L attribution for that currency on that specific date. Days not included are assumed to have zero attribution.
    * **Currency Classification Map (for `PerformanceAttribution`):** A Python dictionary mapping currency codes (uppercase strings, e.g., 'AUD') to classification strings (e.g., 'DM', 'EM'). Must cover all currencies present in the attribution files (except potentially 'USD').

### Step 1: Portfolio Construction (`EqualVolatilityPortfolio`)

This module calculates portfolio weights based on certain rules. We'll use the `EqualVolatilityPortfolio` class as an example.

**1.1 Import and Instantiate:**

```python
import pandas as pd
import numpy as np

# Assume substrat_tri is your loaded DataFrame of substrategy TRIs
# Example Generation:
dates_example = pd.date_range('2018-01-01', '2023-12-31', freq='B')
substrat_tri_example = pd.DataFrame(
    100 + np.random.randn(len(dates_example), 3).cumsum(axis=0) * 0.5,
    index=dates_example, columns=['FX_Carry', 'FX_Value', 'FX_Mom']
).ffill().fillna(100) # Example data

# Replace with your actual data loading
substrat_tri = substrat_tri_example

from portfolio_construction import EqualVolatilityPortfolio

# --- Tunable Parameters ---
lookback = 2          # Years for volatility calculation (e.g., 1, 2, 3)
skip_month = False    # Skip most recent month in vol calc? (True/False)
rebal_freq = 'M'      # Rebalance frequency: 'M', 'Q', 'W', 'A'
days_in_year = 252    # Assumed trading days for annualization

# Instantiate the portfolio builder
try:
    portfolio_builder = EqualVolatilityPortfolio(
        total_return_index=substrat_tri,
        lookback_years=lookback,
        skip_recent_month=skip_month,
        rebalance_freq=rebal_freq,
        trading_days_per_year=days_in_year
    )
    print("Portfolio constructor initialized.")
except ValueError as e:
    print(f"Error initializing portfolio construction: {e}")
    # Handle error appropriately
    exit()
