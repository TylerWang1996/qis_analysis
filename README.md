Quickstart Guide: Portfolio Analysis ToolkitThis guide provides a walkthrough on how to use the provided Python modules for quantitative portfolio analysis:portfolio_construction: Generate portfolio weights (e.g., Equal Volatility).backtester: Calculate strategy performance and metrics.performance_attribution: Analyze performance contributions by substrategy and currency.PrerequisitesLibraries: Ensure you have pandas and numpy installed. You'll also need openpyxl if your attribution data is in .xlsx format.pip install pandas numpy openpyxl
Project Modules: Make sure the Python files (portfolio_construction.py, backtester.py, performance_attribution.py, etc.) are in your Python path or the same working directory.Data:Substrategy TRI: A pandas DataFrame (substrat_tri) with a DatetimeIndex (sorted ascending, business days preferred) and columns representing the Total Return Index for each substrategy. No NaNs allowed (or handle them before passing).Attribution Files (for PerformanceAttribution): Separate Excel files for each substrategy. Each file must contain a sheet named 'attrib' where the index is the date (sparse, only on rebalance days) and columns are currencies (e.g., 'AUD', 'EUR', 'USD', 'BRL'). Values should be the dollar P&L attribution for that currency on that specific date (or for the period ending on that date, as interpreted by the module). Days not included are assumed to have zero attribution.Currency Classification Map (for PerformanceAttribution): A Python dictionary mapping currency codes (uppercase strings, e.g., 'AUD') to classification strings (e.g., 'DM', 'EM'). Must cover all currencies present in the attribution files (except potentially 'USD').Step 1: Portfolio Construction (EqualVolatilityPortfolio)This module calculates portfolio weights based on certain rules. We'll use the EqualVolatilityPortfolio class as an example.1.1 Import and Instantiate:import pandas as pd
# Assume substrat_tri is your loaded DataFrame of substrategy TRIs
# Example:
# dates = pd.date_range('2018-01-01', '2023-12-31', freq='B')
# substrat_tri = pd.DataFrame(np.random.randn(len(dates), 3).cumsum() * 0.01 + 100,
#                             index=dates, columns=['FX_Carry', 'FX_Value', 'FX_Mom'])

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
except ValueError as e:
    print(f"Error initializing portfolio construction: {e}")
    # Handle error appropriately
    exit()

1.2 Tuning Parameters:lookback_years: Controls how far back the volatility calculation looks. Shorter lookbacks react faster but can be noisier.skip_recent_month: If True, the most recent month's data is excluded from the volatility calculation. This can reduce sensitivity to very recent spikes but introduces a lag.rebalance_freq: Determines how often target weights are recalculated ('M' = Month-end, 'Q' = Quarter-end, 'W' = Week-end (Fri), 'A' = Year-end). More frequent rebalancing keeps weights closer to target but may increase turnover.trading_days_per_year: Used for annualizing volatility. Typically 252 for daily data.1.3 Construct Weights:# Calculate the daily EOD weights
try:
    strategy_weights = portfolio_builder.construct_portfolio()
    print("Strategy weights calculated:")
    print(strategy_weights.head())
    print(strategy_weights.tail())
except ValueError as e:
    print(f"Error constructing portfolio weights: {e}")
    # Handle error appropriately
    exit()
The strategy_weights DataFrame now holds the daily target weights for each substrategy.Step 2: Backtesting (Backtester)Use the Backtester class to analyze the performance of the strategy defined by the weights generated in Step 1.2.1 Import and Instantiate:from backtester import Backtester

# Use the weights and TRI from Step 1
try:
    backtester = Backtester(
        strategy_weights=strategy_weights,
        substrategy_total_return_index=substrat_tri
        # Optional: trading_days_per_year, months_per_year
    )
    print("\nBacktester initialized.")
except ValueError as e:
    print(f"Error initializing backtester: {e}")
    exit()
2.2 Get Strategy Returns & Index:# Get daily or monthly returns
daily_rets = backtester.get_strategy_returns(frequency='daily')
monthly_rets = backtester.get_strategy_returns(frequency='monthly')

# Get strategy total return index (starts at 100 by default)
strategy_index = backtester.get_strategy_index(initial_value=100, frequency='daily')
print("\nStrategy Index (Daily, Tail):")
print(strategy_index.tail())
2.3 Calculate Performance Summary:The calculate_performance_summary method calculates various metrics over specified periods. You can analyze based on daily or monthly returns.# --- Daily Performance ---
print("\nCalculating Daily Performance Summary...")

# Default periods: Full Sample, Last 1Y, 3Y, 5Y
daily_summary_default = backtester.calculate_performance_summary(frequency='daily')
print("\nDaily Summary (Default Periods):")
print(daily_summary_default.round(4)) # Display rounded

# Custom periods
custom_periods = {
    'Since 2020': ('2020-01-01', None), # Start date to end
    'COVID Crisis': ('2020-02-01', '2020-04-30'), # Specific range
    'Last 6M': '6M' # Relative (Months)
}
daily_summary_custom = backtester.calculate_performance_summary(periods=custom_periods, frequency='daily')
print("\nDaily Summary (Custom Periods):")
print(daily_summary_custom.round(4))

# --- Monthly Performance ---
print("\nCalculating Monthly Performance Summary...")

# Using the convenience method for monthly default periods
monthly_summary_default = backtester.calculate_monthly_performance_summary()
print("\nMonthly Summary (Default Periods):")
print(monthly_summary_default.round(4))

# Monthly custom periods
monthly_summary_custom = backtester.calculate_monthly_performance_summary(periods=custom_periods)
print("\nMonthly Summary (Custom Periods):")
print(monthly_summary_custom.round(4))

Interpreting Metrics: The summary DataFrames contain metrics like Annualized Return, Volatility, Max Drawdown, Sortino Ratio, Calmar Ratio, etc., calculated for each specified period based on the chosen frequency (daily or monthly). Note that metrics like Volatility and Max Drawdown will typically differ between daily and monthly calculations.Step 3: Performance Attribution (PerformanceAttribution)Analyze where the performance came from using the PerformanceAttribution class.3.1 Prepare Inputs:top_level_weights: The strategy_weights DataFrame from Step 1.substrat_tri: The original substrategy TRI DataFrame.substrat_attrib_files_map: Crucially, you need a dictionary mapping substrategy tickers (matching columns in top_level_weights) to the file paths of their attribution Excel files.# Example (replace with your actual paths!)
attrib_files = {
    'FX_Carry': '/path/to/your/data/fx_carry_attrib.xlsx',
    'FX_Value': '/path/to/your/data/fx_value_attrib.xlsx',
    'FX_Mom':   '/path/to/your/data/fx_mom_attrib.xlsx'
}
# Ensure each Excel file has a sheet named 'attrib'
# with Date index and Currency columns containing dollar attribution.
currency_classification_map: Define which currencies are DM or EM.# Example Classification (customize as needed!)
ccy_map = {
    # DM Examples
    'AUD': 'DM', 'CAD': 'DM', 'CHF': 'DM', 'EUR': 'DM', 'GBP': 'DM',
    'JPY': 'DM', 'NOK': 'DM', 'NZD': 'DM', 'SEK': 'DM', 'USD': 'USD', # USD is special
    # EM Examples
    'BRL': 'EM', 'CLP': 'EM', 'CNY': 'EM', 'COP': 'EM', 'CZK': 'EM',
    'HUF': 'EM', 'IDR': 'EM', 'ILS': 'EM', 'INR': 'EM', 'KRW': 'EM',
    'MXN': 'EM', 'MYR': 'EM', 'PHP': 'EM', 'PLN': 'EM', 'RUB': 'EM',
    'SGD': 'EM', 'THB': 'EM', 'TRY': 'EM', 'TWD': 'EM', 'ZAR': 'EM'
    # Add ALL currencies present in your attribution files!
}
3.2 Import and Instantiate:from performance_attribution import PerformanceAttribution

try:
    pa_analyzer = PerformanceAttribution(
        top_level_weights=strategy_weights, # From Step 1
        substrat_tri=substrat_tri,          # Original TRI
        substrat_attrib_files_map=attrib_files, # Your file map
        currency_classification_map=ccy_map     # Your currency map
    )
    print("\nPerformanceAttribution initialized.")
except (ValueError, FileNotFoundError, TypeError) as e:
     print(f"Error initializing PerformanceAttribution: {e}")
     exit()
3.3 Analyze Attribution:The analyze_attribution method calculates the summed dollar contributions over specified periods.# Analyze using default periods
try:
    attribution_results = pa_analyzer.analyze_attribution()

    # Access results
    sub_attribution = attribution_results['Substrategy']
    currency_attribution = attribution_results['Currency']
    dm_em_attribution = attribution_results['DM_EM']

    print("\n--- Attribution Summary (Default Periods) ---")
    print("\nSubstrategy Contribution:")
    print(sub_attribution.round(2)) # Display rounded dollar amounts
    print("\nCurrency Contribution:")
    print(currency_attribution.round(2))
    print("\nDM/EM Contribution:")
    print(dm_em_attribution.round(2))

    # Analyze using custom periods
    print("\n--- Attribution Summary (Custom Periods) ---")
    custom_periods_pa = {
        'Full Sample': None,
        'Last 1Y': '1Y',
        'YTD': (f'{pd.Timestamp.now().year}-01-01', None) # Example YTD
    }
    attribution_results_custom = pa_analyzer.analyze_attribution(periods=custom_periods_pa)
    print("\nSubstrategy Contribution (Custom):")
    print(attribution_results_custom['Substrategy'].round(2))
    print("\nCurrency Contribution (Custom):")
    print(attribution_results_custom['Currency'].round(2))
    print("\nDM/EM Contribution (Custom):")
    print(attribution_results_custom['DM_EM'].round(2))


except Exception as e:
     print(f"Error during attribution analysis: {e}")

Interpreting Attribution: The results show the total summed dollar contribution of each substrategy, currency, or group (DM/EM) to the overall strategy's performance during each specified period. This helps identify the drivers of P&L.Putting It Together (Conceptual Workflow)import pandas as pd
from portfolio_construction import EqualVolatilityPortfolio
from backtester import Backtester
from performance_attribution import PerformanceAttribution
# Assume other necessary imports and data loading for substrat_tri

# --- 1. Portfolio Construction ---
# Load your substrat_tri DataFrame here
# ...

portfolio_builder = EqualVolatilityPortfolio(
    total_return_index=substrat_tri,
    lookback_years=2,
    rebalance_freq='M'
)
strategy_weights = portfolio_builder.construct_portfolio()

# --- 2. Backtesting ---
backtester = Backtester(
    strategy_weights=strategy_weights,
    substrategy_total_return_index=substrat_tri
)
# Get daily summary
daily_summary = backtester.calculate_performance_summary(frequency='daily')
# Get monthly summary
monthly_summary = backtester.calculate_monthly_performance_summary()

print("\n--- Backtest Summaries ---")
print("Daily:\n", daily_summary.round(4))
print("\nMonthly:\n", monthly_summary.round(4))

# --- 3. Performance Attribution ---
# Define your file map and currency map
attrib_files = { # Replace with actual paths!
    ticker: f'/path/to/data/{ticker}_attrib.xlsx'
    for ticker in strategy_weights.columns
}
ccy_map = { # Replace/extend with your actual map!
    'AUD': 'DM', 'CAD': 'DM', 'EUR': 'DM', 'GBP': 'DM', 'JPY': 'DM',
    'USD': 'USD', 'BRL': 'EM', 'MXN': 'EM', 'ZAR': 'EM'
