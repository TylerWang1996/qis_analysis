import pandas as pd
import numpy as np
import sys # Import sys to use sys.exit()

# Assume the classes are available (either imported from the files or defined in the same script)
# For this example, let's assume the classes are defined in these files:
from portfolio_construction import EqualVolatilityPortfolio
from backtester import Backtester

# --- 1. Load Your Data ---
# Replace the section below with your code to load the total return index data.
# The data should be in a pandas DataFrame named 'total_return_index'.

# Requirements for the 'total_return_index' DataFrame:
#   - Index: Must be a pandas DatetimeIndex, sorted in ascending order (oldest date first).
#            Represents the dates for which the index values are recorded (e.g., daily, business days).
#   - Columns: Each column should represent a different asset or substrategy,
#              with the column names being the tickers or identifiers for those assets.
#   - Values: The values in the DataFrame should be the total return index figures
#             (i.e., reflecting price appreciation and reinvested dividends/income).
#             The index should typically start at a base value (e.g., 100 or 1000) on the first date.
#   - No NaNs: The DataFrame should not contain any NaN (Not a Number) values.
#              Ensure data is cleaned or appropriately filled before passing it to the constructor.
#   - Sufficient History: Must contain enough historical data to cover the specified
#                        'lookback_years' for the initial calculation date. For example,
#                        if lookback_years=2 and the first calculation needs data up to
#                        2020-12-31, the DataFrame must contain data going back to at least
#                        2019-01-01 (or earlier, depending on exact date alignment).

# --- Placeholder: Assign your loaded DataFrame here ---
# Initialize as None before the try block
total_return_index = None

# Example of how you might load data from a CSV and validate it:
try:
    # --- Replace this line with your actual data loading ---
    # Example:
    # total_return_index = pd.read_csv(
    #     'your_total_return_data.csv',
    #     index_col='Date',        # Assuming 'Date' column contains dates
    #     parse_dates=True         # Automatically parse the index column as dates
    # )

    # --- For demonstration, create a small valid DataFrame ---
    # !!! REMOVE OR REPLACE THIS BLOCK WITH YOUR ACTUAL DATA LOADING !!!
    print("--- Creating placeholder data for demonstration ---")
    _dates_demo = pd.date_range(start='2019-01-01', end='2023-12-31', freq='B')
    _assets_demo = ['AssetX', 'AssetY']
    _data_demo = np.random.randn(len(_dates_demo), len(_assets_demo)).cumsum(axis=0) + 100
    total_return_index = pd.DataFrame(_data_demo, index=_dates_demo, columns=_assets_demo)
    print("--- Placeholder data created ---")
    # !!! END OF PLACEHOLDER BLOCK !!!


    # --- Data Validation (Now Active) ---
    if total_return_index is None or not isinstance(total_return_index, pd.DataFrame):
         raise ValueError("Data loading failed or did not produce a DataFrame.")

    # Ensure the index is sorted (do this *before* checking monotonicity)
    total_return_index.sort_index(inplace=True)

    if not isinstance(total_return_index.index, pd.DatetimeIndex):
        raise TypeError("Index is not a DatetimeIndex after loading. Check date parsing.")
    if not total_return_index.index.is_monotonic_increasing:
        # This check should ideally not fail if sort_index() was called, but good practice to keep.
        raise ValueError("Index is not sorted ascending after sorting attempt.")
    if total_return_index.isnull().values.any():
        # Handle NaNs if necessary, e.g., forward fill, backward fill, or raise error
        print("Error: Input data contains NaNs. Please clean data before running.")
        # Example: total_return_index.fillna(method='ffill', inplace=True)
        # If cleaning is not desired, raise an error or exit:
        raise ValueError("Input data contains NaNs. Please clean data.")
    if total_return_index.empty:
        raise ValueError("Loaded data DataFrame is empty.")

    print("--- Data Validation Passed ---")
    print("--- Loaded Total Return Index (first 5 rows) ---")
    print(total_return_index.head())
    print("\n")

except FileNotFoundError:
     print("Error: Data file not found. Please provide the correct path.")
     sys.exit(1) # Exit script with an error code
except (TypeError, ValueError) as e:
     # Catch specific validation errors raised above
     print(f"Data validation error: {e}")
     sys.exit(1) # Exit script with an error code
except Exception as e:
     # Catch any other unexpected errors during loading/validation
     print(f"An unexpected error occurred during data loading or validation: {e}")
     sys.exit(1) # Exit script with an error code


# --- Proceed only if data loading and validation were successful ---

# --- 2. Construct Portfolio Weights ---
# Use the EqualVolatilityPortfolio class from portfolio_construction.py

# Initialize the portfolio construction object
# Available options (Defaults shown):
#   lookback_years (int): Lookback for volatility calculation. Default: 2
#   skip_recent_month (bool): Exclude the most recent month from volatility window. Default: False
#   rebalance_freq (str): Rebalancing frequency ('M', 'Q', 'A', 'W'). Default: 'M'
#   trading_days_per_year (int): For annualization. Default: 252

try:
    portfolio_constructor = EqualVolatilityPortfolio(
        total_return_index=total_return_index,
        lookback_years=2,              # Use a 2-year lookback for volatility
        skip_recent_month=False,       # Include the most recent month
        rebalance_freq='Q',            # Rebalance quarterly ('Q')
        trading_days_per_year=252
    )

    # Construct the daily portfolio weights
    # This method calculates target weights at rebalance dates and handles drift between rebalances
    strategy_weights = portfolio_constructor.construct_portfolio()

    print("--- Calculated Strategy Weights (first 5 rows) ---")
    print(strategy_weights.head())
    print("\n--- Calculated Strategy Weights (last 5 rows) ---")
    print(strategy_weights.tail())
    print("\n")

    # --- 3. Backtest the Strategy ---
    # Use the Backtester class from backtester.py

    # Initialize the backtester
    # Requires the calculated strategy weights and the original total return index
    backtester = Backtester(
        strategy_weights=strategy_weights,
        substrategy_total_return_index=total_return_index, # This is the TRI of the assets used
        trading_days_per_year=252,
        months_per_year=12
    )

    # --- Optional: Get Strategy Returns and Index ---
    # You can get daily or monthly returns
    daily_strat_returns = backtester.get_strategy_returns(frequency='daily')
    monthly_strat_returns = backtester.get_strategy_returns(frequency='monthly')
    print("--- Strategy Daily Returns (first 5) ---")
    print(daily_strat_returns.head())
    print("\n--- Strategy Monthly Returns (first 5) ---")
    print(monthly_strat_returns.head())
    print("\n")

    # You can get the strategy's total return index
    strategy_index_daily = backtester.get_strategy_index(frequency='daily')
    print("--- Strategy Index (Daily, first 5) ---")
    print(strategy_index_daily.head())
    print("\n")


    # --- Calculate Performance Summary ---
    # Define periods for analysis (optional, defaults are provided)
    # Examples:
    #   None: Use default periods ('Full Sample', 'Last 1Y', 'Last 3Y', 'Last 5Y')
    #   {'My Period': ('2020-01-01', '2022-12-31')}: Custom date range
    #   {'Last 6M': '6M'}: Relative period
    custom_periods = {
        'Full Sample': None,
        'Last 3Y': '3Y',
        'Since 2021': ('2021-01-01', None) # Start date to end
    }

    # Calculate performance summary using daily returns
    performance_summary_daily = backtester.calculate_performance_summary(
        periods=custom_periods,
        frequency='daily'
    )
    print("--- Performance Summary (Based on Daily Returns) ---")
    print(performance_summary_daily.round(4)) # Round for display
    print("\n")

    # Calculate performance summary using monthly returns
    # Uses the convenience method calculate_monthly_performance_summary
    performance_summary_monthly = backtester.calculate_monthly_performance_summary(
        periods=custom_periods
    )
    print("--- Performance Summary (Based on Monthly Returns) ---")
    print(performance_summary_monthly.round(4)) # Round for display
    print("\n")

except ValueError as e:
    # Catch errors specifically from EqualVolatilityPortfolio or Backtester initialization/methods
    print(f"An error occurred during portfolio construction or backtesting: {e}")
    sys.exit(1) # Exit script with an error code
except Exception as e:
    # Catch any other unexpected errors during construction/backtesting
    print(f"An unexpected error occurred: {e}")
    sys.exit(1) # Exit script with an error code

print("--- Script finished successfully ---")
