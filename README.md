Quickstart Guide: Portfolio Analysis ToolkitThis guide provides a walkthrough on how to use the provided Python modules for quantitative portfolio analysis:portfolio_construction: Generate portfolio weights (e.g., Equal Volatility).backtester: Calculate strategy performance and metrics.performance_attribution: Analyze performance contributions by substrategy and currency.PrerequisitesLibraries: Ensure you have pandas and numpy installed. You'll also need openpyxl if your attribution data is in .xlsx format.pip install pandas numpy openpyxl
Project Modules: Make sure the Python files (portfolio_construction.py, backtester.py, performance_attribution.py, etc.) are in your Python path or the same working directory.Data:Substrategy TRI: A pandas DataFrame (substrat_tri) with a DatetimeIndex (sorted ascending, business days preferred) and columns representing the Total Return Index for each substrategy. No NaNs allowed (or handle them before passing).Attribution Files (for PerformanceAttribution): Separate Excel files for each substrategy. Each file must contain a sheet named 'attrib' where the index is the date (sparse, only on rebalance days) and columns are currencies (e.g., 'AUD', 'EUR', 'USD', 'BRL'). Values should be the dollar P&L attribution for that currency on that specific date. Days not included are assumed to have zero attribution.Currency Classification Map (for PerformanceAttribution): A Python dictionary mapping currency codes (uppercase strings, e.g., 'AUD') to classification strings (e.g., 'DM', 'EM'). Must cover all currencies present in the attribution files (except potentially 'USD').Step 1: Portfolio Construction (EqualVolatilityPortfolio)This module calculates portfolio weights based on certain rules. We'll use the EqualVolatilityPortfolio class as an example.1.1 Import and Instantiate:import pandas as pd
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
1.2 Tuning Parameters:lookback_years: Controls how far back the volatility calculation looks. Shorter lookbacks react faster but can be noisier.skip_recent_month: If True, the most recent month's data is excluded from the volatility calculation. This can reduce sensitivity to very recent spikes but introduces a lag.rebalance_freq: Determines how often target weights are recalculated ('M' = Month-end, 'Q' = Quarter-end, 'W' = Week-end (Fri), 'A' = Year-end). More frequent rebalancing keeps weights closer to target but may increase turnover.trading_days_per_year: Used for annualizing volatility. Typically 252 for daily data.1.3 Construct Weights:# Calculate the daily EOD weights
try:
    strategy_weights = portfolio_builder.construct_portfolio()
    print("\nStrategy weights calculated:")
    print("Shape:", strategy_weights.shape)
    print("Head:\n", strategy_weights.head())
    print("\nTail:\n", strategy_weights.tail())
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
print(f"\nCalculated {len(daily_rets)} daily returns and {len(monthly_rets)} monthly returns.")

# Get strategy total return index (starts near 100 by default)
strategy_index = backtester.get_strategy_index(initial_value=100, frequency='daily')
print("\nStrategy Index (Daily, Tail):")
print(strategy_index.tail())
2.3 Calculate Performance Summary:The calculate_performance_summary method calculates various metrics over specified periods. You can analyze based on daily or monthly returns.# --- Daily Performance ---
print("\nCalculating Daily Performance Summary...")

# Default periods: Full Sample, Last 1Y, 3Y, 5Y
daily_summary_default = backtester.calculate_performance_summary(frequency='daily')
print("\nDaily Summary (Default Periods):")
# Example formatting for display
print(daily_summary_default.applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.number)) and abs(x)<5 else (f"{x:.2f}" if isinstance(x, (float, np.number)) else x)))


# Custom periods
custom_periods = {
    'Since 2020': ('2020-01-01', None), # Start date to end
    'COVID Crisis': ('2020-02-01', '2020-04-30'), # Specific range
    'Last 6M': '6M' # Relative (Months)
}
daily_summary_custom = backtester.calculate_performance_summary(periods=custom_periods, frequency='daily')
print("\nDaily Summary (Custom Periods):")
print(daily_summary_custom.applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.number)) and abs(x)<5 else (f"{x:.2f}" if isinstance(x, (float, np.number)) else x)))


# --- Monthly Performance ---
print("\nCalculating Monthly Performance Summary...")

# Using the convenience method for monthly default periods
monthly_summary_default = backtester.calculate_monthly_performance_summary()
print("\nMonthly Summary (Default Periods):")
print(monthly_summary_default.applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.number)) and abs(x)<5 else (f"{x:.2f}" if isinstance(x, (float, np.number)) else x)))


# Monthly custom periods
monthly_summary_custom = backtester.calculate_monthly_performance_summary(periods=custom_periods)
print("\nMonthly Summary (Custom Periods):")
print(monthly_summary_custom.applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.number)) and abs(x)<5 else (f"{x:.2f}" if isinstance(x, (float, np.number)) else x)))

Interpreting Metrics: The summary DataFrames contain metrics like Annualized Return, Volatility, Max Drawdown, Sortino Ratio, Calmar Ratio, etc., calculated for each specified period based on the chosen frequency (daily or monthly). Note that metrics like Volatility and Max Drawdown will typically differ between daily and monthly calculations.Step 3: Performance Attribution (PerformanceAttribution)Analyze where the performance came from using the PerformanceAttribution class.3.1 Prepare Inputs:top_level_weights: The strategy_weights DataFrame from Step 1.substrat_tri: The original substrategy TRI DataFrame.substrat_attrib_files_map: Crucially, you need a dictionary mapping substrategy tickers (matching columns in top_level_weights) to the file paths of their attribution Excel files.# Example (replace with your actual paths!)
# Ensure the tickers 'FX_Carry', 'FX_Value', 'FX_Mom' match your weight columns
attrib_files = {
    'FX_Carry': '/path/to/your/data/fx_carry_attrib.xlsx',
    'FX_Value': '/path/to/your/data/fx_value_attrib.xlsx',
    'FX_Mom':   '/path/to/your/data/fx_mom_attrib.xlsx'
    # Add entries for ALL strategies in strategy_weights
}
# Ensure each Excel file has a sheet named 'attrib'
# with Date index and Currency columns containing dollar attribution.
# Days not present in the file are assumed to have zero attribution.
currency_classification_map: Define which currencies are DM or EM.# Example Classification (customize and ensure it covers ALL currencies in attrib files!)
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

# Make sure attrib_files and ccy_map are defined as above
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
     # Handle error appropriately (e.g., check file paths, map content)
     exit()
3.3 Analyze Attribution:The analyze_attribution method calculates the summed dollar contributions over specified periods.# Analyze using default periods
try:
    attribution_results = pa_analyzer.analyze_attribution()

    # Access results (dictionaries containing DataFrames)
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
import numpy as np # Needed for example data generation
from portfolio_construction import EqualVolatilityPortfolio
from backtester import Backtester
from performance_attribution import PerformanceAttribution

# --- 0. Load/Prepare Data ---
# Replace this with your actual data loading
print("Loading/Generating Substrategy TRI Data...")
dates_example = pd.date_range('2018-01-01', '2023-12-31', freq='B')
substrat_tri = pd.DataFrame(
    100 + np.random.randn(len(dates_example), 3).cumsum(axis=0) * 0.5,
    index=dates_example, columns=['FX_Carry', 'FX_Value', 'FX_Mom']
).ffill().fillna(100)
print("Substrategy TRI Data Ready.")

# --- 1. Portfolio Construction ---
print("\nRunning Portfolio Construction...")
portfolio_builder = EqualVolatilityPortfolio(
    total_return_index=substrat_tri,
    lookback_years=2,
    rebalance_freq='M'
)
strategy_weights = portfolio_builder.construct_portfolio()
print("Strategy Weights Generated.")

# --- 2. Backtesting ---
print("\nRunning Backtester...")
backtester = Backtester(
    strategy_weights=strategy_weights,
    substrategy_total_return_index=substrat_tri
)
daily_summary = backtester.calculate_performance_summary(frequency='daily')
monthly_summary = backtester.calculate_monthly_performance_summary()
print("Backtesting Complete.")
# print("\n--- Backtest Summaries ---") # Optional: Display results here
# print("Daily:\n", daily_summary.round(4))
# print("\nMonthly:\n", monthly_summary.round(4))

# --- 3. Performance Attribution ---
print("\nRunning Performance Attribution...")
# Define your file map and currency map
# !!! IMPORTANT: Replace with actual paths and complete currency map !!!
attrib_files = {
    ticker: f'./dummy_{ticker}_attrib.xlsx' # Placeholder - requires actual files
    for ticker in strategy_weights.columns
}
ccy_map = {
    'AUD': 'DM', 'CAD': 'DM', 'CHF': 'DM', 'EUR': 'DM', 'GBP': 'DM',
    'JPY': 'DM', 'NOK': 'DM', 'NZD': 'DM', 'SEK': 'DM', 'USD': 'USD',
    'BRL': 'EM', 'CLP': 'EM', 'CNY': 'EM', 'COP': 'EM', 'CZK': 'EM',
    'HUF': 'EM', 'IDR': 'EM', 'ILS': 'EM', 'INR': 'EM', 'KRW': 'EM',
    'MXN': 'EM', 'MYR': 'EM', 'PHP': 'EM', 'PLN': 'EM', 'RUB': 'EM',
    'SGD': 'EM', 'THB': 'EM', 'TRY': 'EM', 'TWD': 'EM', 'ZAR': 'EM'
    # Add any other currencies present in your files!
}

# Create dummy attribution files for the example to run without error
# In reality, you would have your actual files at the specified paths
print("Creating dummy attribution files for example...")
for ticker, path in attrib_files.items():
    try:
        # Create a simple dummy file if it doesn't exist
        if not os.path.exists(path):
             dummy_dates = pd.date_range(strategy_weights.index[0], strategy_weights.index[-1], freq='MS') # Monthly dummy data
             dummy_attrib = pd.DataFrame({
                 'AUD': np.random.randn(len(dummy_dates)) * 10,
                 'EUR': np.random.randn(len(dummy_dates)) * 5,
                 'BRL': np.random.randn(len(dummy_dates)) * 2,
                 'USD': np.random.randn(len(dummy_dates)) * -15,
             }, index=dummy_dates)
             # Ensure directory exists (optional)
             # os.makedirs(os.path.dirname(path), exist_ok=True)
             with pd.ExcelWriter(path) as writer:
                  dummy_attrib.to_excel(writer, sheet_name='attrib')
             print(f"Created dummy file: {path}")
    except Exception as file_err:
        print(f"Warning: Could not create dummy file {path}: {file_err}")
# --- End of Dummy File Creation ---


try:
    pa_analyzer = PerformanceAttribution(
        top_level_weights=strategy_weights,
        substrat_tri=substrat_tri,
        substrat_attrib_files_map=attrib_files,
        currency_classification_map=ccy_map
    )
    attribution_results = pa_analyzer.analyze_attribution()
    print("Attribution Analysis Complete.")

    # Display results (optional)
    # print("\n--- Attribution Summaries ---")
    # print("\nSubstrategy:\n", attribution_results['Substrategy'].round(2))
    # print("\nCurrency:\n", attribution_results['Currency'].round(2))
    # print("\nDM/EM:\n", attribution_results['DM_EM'].round(2))

except (FileNotFoundError, ValueError) as e:
     print(f"\nCould not run attribution: {e}")
     print("Please ensure attribution files exist at the specified paths and contain the 'attrib' sheet.")
     print("Also ensure the currency map covers all currencies in the files.")
except Exception as e:
     print(f"\nAn unexpected error occurred during attribution: {e}")

print("\n--- Workflow Finished ---")
ConclusionThis guide provides the basic steps to use the portfolio construction, backtesting, and performance attribution modules. You can adjust the parameters in EqualVolatilityPortfolio, the analysis periods in Backtester and PerformanceAttribution, and the currency classification map to suit your specific needs. Remember to ensure your input data (TRI, attribution files) is clean and correctly formatted.
