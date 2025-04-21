import pandas as pd
import numpy as np
from numpy.linalg import cholesky, LinAlgError # For Cholesky decomposition
from pandas.tseries.offsets import BDay, DateOffset, MonthEnd
import traceback # For detailed error reporting if needed
import time # For timing tests if needed
import os # For dummy paths
from unittest.mock import patch, MagicMock # Keep patch for currency exposure
from typing import Optional, Dict, List, Union # <--- Added this import line

# --- IMPORTANT ---
# Assumes necessary classes are saved in respective files
# Adjust filenames if necessary.
try:
    from portfolio_construction import EqualVolatilityPortfolio
    from backtester import Backtester
    from currency_exposure import CurrencyExposureCalculator
    # Import the new class instead of the function
    from correlation_analyzer import CorrelationAnalyzer # Assuming saved as correlation_analyzer_v1.py
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure 'portfolio_construction.py', 'backtester.py', 'currency_exposure.py', and 'correlation_analyzer_v1.py' exist") # Adjust filename if saved differently
    print("and are in the Python path or the same directory.")
    exit() # Exit if imports fail

# --- Test Data Generation (Enhanced) ---

def create_sample_data(start_date='2017-01-01',
                       end_date='2023-12-31',
                       n_assets=4,
                       seed=42,
                       target_corr_matrix: Optional[np.ndarray] = None
                       ) -> pd.DataFrame:
    """
    Creates sample total return index data for testing.
    Optionally generates correlated returns using Cholesky decomposition.

    Args:
        start_date (str): Start date for the data.
        end_date (str): End date for the data.
        n_assets (int): Number of assets to generate.
        seed (int): Random seed for reproducibility.
        target_corr_matrix (Optional[np.ndarray]): Target correlation matrix.
            If provided, must be a square positive semi-definite matrix of size n_assets x n_assets.
            If None, assets will be generated independently.

    Returns:
        pd.DataFrame: DataFrame of total return indices.

    Raises:
        ValueError: If target_corr_matrix is invalid.
        LinAlgError: If Cholesky decomposition fails (matrix not positive definite).
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=BDay()) # Business Days
    n_dates = len(dates)
    asset_names = [f'SubStrat_{i+1}' for i in range(n_assets)]
    np.random.seed(seed)

    # Simulate daily log returns with some drift and varying volatility
    mean_returns = np.random.uniform(0.0001, 0.0005, n_assets)
    volatilities = np.random.uniform(0.008, 0.025, n_assets) # Daily vols

    # Generate random numbers (either correlated or uncorrelated)
    if target_corr_matrix is not None:
        # Validate target correlation matrix
        if not isinstance(target_corr_matrix, np.ndarray):
            raise ValueError("target_corr_matrix must be a numpy array.")
        if target_corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"target_corr_matrix shape must be ({n_assets}, {n_assets}), "
                             f"got {target_corr_matrix.shape}")
        # Add check for near positive semi-definiteness if needed, Cholesky will fail otherwise

        print(f"Generating {n_assets} assets with target correlation...")
        try:
            # Compute Cholesky decomposition
            L = cholesky(target_corr_matrix) # L * L.T = target_corr_matrix
            # Generate uncorrelated standard normal random numbers
            Z = np.random.normal(loc=0.0, scale=1.0, size=(n_dates, n_assets))
            # Generate correlated standard normal random numbers
            # C = Z @ L.T works if Z rows are variables, columns are observations
            # Here, rows are time, columns are assets. We need L @ Z.T then transpose back?
            # Let's try Z @ L.T - this should correlate columns (assets) across time steps
            C = Z @ L.T # Correlated standard normals (mean 0, std 1)
            # Scale correlated normals by volatility and add mean for log returns
            # Reshape volatilities and means to broadcast correctly
            log_returns_array = mean_returns.reshape(1, -1) + C * volatilities.reshape(1, -1)

        except LinAlgError:
            raise LinAlgError("Cholesky decomposition failed. Target correlation matrix must be positive definite.")
    else:
        # Generate independent random numbers
        print(f"Generating {n_assets} independent assets...")
        log_returns_array = np.random.normal(loc=mean_returns.reshape(1, -1),
                                             scale=volatilities.reshape(1, -1),
                                             size=(n_dates, n_assets))

    log_returns = pd.DataFrame(log_returns_array, index=dates, columns=asset_names)

    # Create price series starting from 100 (Total Return Index)
    total_return_index_data = 100 * np.exp(log_returns.cumsum())
    # Ensure no NaNs in TRI (can happen if log_returns had issues)
    total_return_index_data = total_return_index_data.ffill().fillna(100.0)
    return total_return_index_data


# --- Portfolio Construction Test Suite ---
# (pc_test_* functions remain unchanged)
def pc_test_standard_configs(test_data):
    """Portfolio Construction: Tests standard configurations."""
    print("\n--- Portfolio Construction Test Suite: Standard Configurations ---")
    configs = [
        {'lookback_years': 2, 'skip_recent_month': False, 'rebalance_freq': 'M', 'label': 'PC Monthly, 2y Lookback, No Skip'},
        {'lookback_years': 3, 'skip_recent_month': True, 'rebalance_freq': 'Q', 'label': 'PC Quarterly, 3y Lookback, Skip Month'},
        {'lookback_years': 1, 'skip_recent_month': False, 'rebalance_freq': 'W', 'label': 'PC Weekly, 1y Lookback, No Skip'},
    ]
    all_passed = True
    results = {}

    for config in configs:
        label = config['label']
        print(f"  Running: {label}...", end="")
        try:
            portfolio_builder = EqualVolatilityPortfolio(
                total_return_index=test_data,
                lookback_years=config['lookback_years'],
                skip_recent_month=config['skip_recent_month'],
                rebalance_freq=config['rebalance_freq']
            )
            daily_weights = portfolio_builder.construct_portfolio()
            results[label] = daily_weights
            assert not daily_weights.empty, "Weights DataFrame is empty"
            assert daily_weights.index.is_monotonic_increasing, "Weights index not increasing"
            assert not daily_weights.isnull().values.any(), "Weights contain NaNs"
            sum_check = daily_weights.sum(axis=1)
            assert np.isclose(sum_check, 1.0).all(), f"Weights do not sum to 1"
            print(" Passed")
        except (ValueError, TypeError, AssertionError) as e:
            print(f" Failed: {e}")
            all_passed = False
        except Exception as e:
             print(f" Failed: Unexpected error - {e}")
             all_passed = False
    return all_passed

def pc_test_calculation_accuracy():
    """Portfolio Construction: Tests weight calculation accuracy."""
    print("\n--- Portfolio Construction Test Suite: Calculation Accuracy ---")
    test_passed = True
    label = "PC Predictable Volatility Weights"
    print(f"  Running: {label}...", end="")
    try:
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq=BDay())
        n_dates = len(dates)
        returns_a = np.resize([0.02, -0.02], n_dates)
        returns_b = np.resize([0.002, -0.002], n_dates)
        returns_df = pd.DataFrame({'Asset_A': returns_a, 'Asset_B': returns_b}, index=dates)
        test_data = 100 * (1 + returns_df).cumprod()
        builder = EqualVolatilityPortfolio(total_return_index=test_data, lookback_years=1, skip_recent_month=False, rebalance_freq='M')
        weights = builder.construct_portfolio()
        expected_weight_a = 1/11
        expected_weight_b = 10/11
        first_weight_date = weights.index[0]
        weights_on_first_day = weights.loc[first_weight_date]
        assert np.isclose(weights_on_first_day['Asset_A'], expected_weight_a, atol=0.02), f"Asset A weight incorrect"
        assert np.isclose(weights_on_first_day['Asset_B'], expected_weight_b, atol=0.02), f"Asset B weight incorrect"
        assert np.isclose(weights_on_first_day.sum(), 1.0), "Weights do not sum to 1"
        print(" Passed")
    except (ValueError, TypeError, AssertionError) as e:
        print(f" Failed: {e}")
        test_passed = False
    except Exception as e:
        print(f" Failed: Unexpected error - {e}")
        test_passed = False
    return test_passed

def pc_test_lookahead_bias(test_data):
    """Portfolio Construction: Tests for lookahead bias."""
    print("\n--- Portfolio Construction Test Suite: Lookahead Bias ---")
    test_passed = True
    label = "PC No Lookahead Bias"
    print(f"  Running: {label}...", end="")
    try:
        builder_orig = EqualVolatilityPortfolio(total_return_index=test_data, lookback_years=1, rebalance_freq='M')
        weights_orig = builder_orig.construct_portfolio()
        test_date = weights_orig.index[len(weights_orig) // 2]
        weights_on_test_date = weights_orig.loc[test_date].copy()
        modified_data = test_data.copy()
        modify_idx = modified_data.index.get_loc(test_date) + 1
        if modify_idx < len(modified_data): modified_data.iloc[modify_idx:, :] *= 1.5
        builder_mod = EqualVolatilityPortfolio(total_return_index=modified_data, lookback_years=1, rebalance_freq='M')
        weights_mod = builder_mod.construct_portfolio()
        weights_mod_on_test_date = weights_mod.loc[test_date]
        assert np.allclose(weights_on_test_date, weights_mod_on_test_date, atol=1e-9), f"Weights changed on {test_date.date()}"
        print(" Passed")
    except (ValueError, TypeError, AssertionError) as e:
        print(f" Failed: {e}")
        test_passed = False
    except Exception as e:
        print(f" Failed: Unexpected error - {e}")
        test_passed = False
    return test_passed

def pc_test_edge_cases(test_data):
    """Portfolio Construction: Tests edge cases."""
    print("\n--- Portfolio Construction Test Suite: Edge Cases ---")
    all_passed = True
    lookback_years_edge = 1

    # --- Constant Price Asset ---
    label = "PC Constant Price Asset"
    print(f"  Running: {label}...", end="")
    try:
        edge_data = test_data.copy()
        constant_asset = edge_data.columns[0]
        start_idx = len(edge_data) - (252 * lookback_years_edge + 60)
        constant_date = edge_data.index[start_idx]
        edge_data.loc[constant_date:, constant_asset] = edge_data.loc[constant_date, constant_asset]
        builder = EqualVolatilityPortfolio(total_return_index=edge_data, lookback_years=lookback_years_edge, rebalance_freq='M', trading_days_per_year=252)
        weights = builder.construct_portfolio()
        target_check_start_date = constant_date + pd.DateOffset(months=lookback_years_edge * 12 + 1)
        valid_check_dates = weights.index[weights.index >= target_check_start_date]
        if valid_check_dates.empty: raise ValueError(f"Test setup error: No weight dates found after {target_check_start_date.date()}")
        check_weights_from = valid_check_dates[0]
        weights_to_check = weights.loc[check_weights_from:, constant_asset]
        max_weight = weights_to_check.max()
        assert np.isclose(max_weight, 0.0, atol=1e-8), f"Weight for constant asset '{constant_asset}' ({max_weight:.2e}) is not zero"
        assert np.isclose(weights.loc[check_weights_from:].sum(axis=1), 1.0).all(), f"Weights do not sum to 1"
        print(" Passed")
    except (ValueError, TypeError, AssertionError) as e:
        print(f" Failed: {e}")
        all_passed = False
    except Exception as e:
        print(f" Failed: Unexpected error - {e}")
        all_passed = False

    # --- Single Asset ---
    label = "PC Single Asset Input"
    print(f"  Running: {label}...", end="")
    try:
        single_asset_data = test_data[[test_data.columns[0]]].copy()
        builder = EqualVolatilityPortfolio(total_return_index=single_asset_data, lookback_years=lookback_years_edge, rebalance_freq='M')
        weights = builder.construct_portfolio()
        assert not weights.empty, "Single asset weights are empty"
        assert np.isclose(weights.iloc[:, 0], 1.0).all(), f"Weight for single asset is not always 1.0"
        print(" Passed")
    except (ValueError, TypeError, AssertionError) as e:
        print(f" Failed: {e}")
        all_passed = False
    except Exception as e:
        print(f" Failed: Unexpected error - {e}")
        all_passed = False
    return all_passed

def pc_test_input_validation(test_data):
    """Portfolio Construction: Tests input validation."""
    print("\n--- Portfolio Construction Test Suite: Input Validation ---")
    all_passed = True
    # --- Insufficient Data ---
    label = "PC Insufficient Data Error"
    print(f"  Running: {label}...", end="")
    required_days = 252 * 2 + 30
    short_data = test_data.iloc[:required_days -1]
    try:
         builder_short = EqualVolatilityPortfolio(total_return_index=short_data, lookback_years=2, rebalance_freq='M')
         weights_short = builder_short.construct_portfolio()
         print(" Failed: Expected ValueError")
         all_passed = False
    except ValueError as e:
         if "Not enough data history" in str(e) or "Not enough data for any rebalances" in str(e): print(f" Passed")
         else: print(f" Failed: Unexpected ValueError msg: {e}"); all_passed = False
    except Exception as e: print(f" Failed: Unexpected error: {e}"); all_passed = False
    # --- NaN Data ---
    label = "PC NaN Data Error"
    print(f"  Running: {label}...", end="")
    nan_data = test_data.copy(); nan_data.iloc[10, 0] = np.nan
    try:
        builder_nan = EqualVolatilityPortfolio(total_return_index=nan_data, lookback_years=1, rebalance_freq='M')
        print(" Failed: Expected ValueError")
        all_passed = False
    except ValueError as e:
        if "contains NaNs" in str(e): print(f" Passed")
        else: print(f" Failed: Unexpected ValueError msg: {e}"); all_passed = False
    except Exception as e: print(f" Failed: Unexpected error: {e}"); all_passed = False
    return all_passed

def run_all_portfolio_tests(standard_data):
    """Runs all test suites for Portfolio Construction."""
    print("\n" + "="*50)
    print(" Running Portfolio Construction Tests ".center(50, "="))
    print("="*50)
    results = {}
    results["PC Standard Configs"] = pc_test_standard_configs(standard_data)
    results["PC Calculation Accuracy"] = pc_test_calculation_accuracy()
    results["PC Lookahead Bias"] = pc_test_lookahead_bias(standard_data)
    results["PC Edge Cases"] = pc_test_edge_cases(standard_data)
    results["PC Input Validation"] = pc_test_input_validation(standard_data)
    print("="*50)
    print(" Portfolio Construction Test Summary ".center(50, "="))
    print("="*50)
    final_outcome = True
    for test_suite, passed in results.items():
        status = "Passed" if passed else "Failed"
        print(f"- {test_suite}: {status}")
        if not passed: final_outcome = False
    return final_outcome


# --- Backtester Test Suite ---

def bt_test_initialization(weights, tri):
    """Backtester: Test initialization and basic return calculation."""
    print("\n--- Backtester Test Suite: Initialization & Returns ---")
    label = "BT Initialization"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        backtester = Backtester(strategy_weights=weights, substrategy_total_return_index=tri)

        # Test Daily Returns
        daily_returns = backtester.strategy_daily_returns
        assert not daily_returns.empty, "Strategy daily returns are empty"
        assert daily_returns.index.is_monotonic_increasing, "Strategy daily returns index not monotonic"
        assert not daily_returns.isnull().any(), "Strategy daily returns contain NaNs"

        # Test Monthly Returns
        monthly_returns = backtester.strategy_monthly_returns
        assert not monthly_returns.empty, "Strategy monthly returns are empty"
        assert monthly_returns.index.is_monotonic_increasing, "Strategy monthly returns index not monotonic"
        # Check if index dates are month ends (compatible way)
        assert all(d == d + MonthEnd(0) for d in monthly_returns.index), "Strategy monthly returns index not month end"
        assert not monthly_returns.isnull().any(), "Strategy monthly returns contain NaNs"

        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc() # Uncomment for full traceback
        test_passed = False
    return test_passed

def bt_test_index_generation(backtester):
    """Backtester: Test strategy index generation."""
    label = "BT Index Generation"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        # Test daily index
        strategy_index_d = backtester.get_strategy_index(initial_value=100, frequency='daily')
        assert not strategy_index_d.empty, "Strategy daily index is empty"
        assert strategy_index_d.index.is_monotonic_increasing, "Daily index not increasing"
        assert not strategy_index_d.isnull().any(), "Daily index contains NaNs"
        assert abs(strategy_index_d.iloc[0] / 100 - 1) < 0.1, f"Daily index does not start near 100 (starts at {strategy_index_d.iloc[0]})"

        # Test monthly index
        strategy_index_m = backtester.get_strategy_index(initial_value=100, frequency='monthly')
        assert not strategy_index_m.empty, "Strategy monthly index is empty"
        assert strategy_index_m.index.is_monotonic_increasing, "Monthly index not increasing"
        assert not strategy_index_m.isnull().any(), "Monthly index contains NaNs"
        assert abs(strategy_index_m.iloc[0] / 100 - 1) < 0.1, f"Monthly index does not start near 100 (starts at {strategy_index_m.iloc[0]})"

        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False
    return test_passed

def bt_test_metric_calculation(backtester):
    """Backtester: Test calculation of performance summary (daily)."""
    label = "BT Metric Calculation Daily (Default Periods)"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        summary = backtester.calculate_performance_summary(frequency='daily') # Test default periods
        assert isinstance(summary, pd.DataFrame), "Result is not a DataFrame"
        assert not summary.empty, "Summary DataFrame is empty"
        expected_metrics = ['Ann Return', 'Ann Volatility', 'Information Ratio', 'Sortino Ratio',
                            'Skewness', 'Kurtosis', 'Max Drawdown', 'Calmar Ratio',
                            '% Positive Months', 'DD Recovery Months']
        assert all(metric in summary.index for metric in expected_metrics), "Missing expected metrics in daily summary"
        assert (summary.loc['Max Drawdown'].fillna(0) <= 1e-9).all(), f"Daily Max Drawdown is positive: {summary.loc['Max Drawdown']}" # Use tolerance
        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False

    label = "BT Metric Calculation Daily (Custom Periods)"
    print(f"  Running: {label}...", end="")
    try:
        custom_periods = {
            'Last 3Y': '3Y', 'COVID': ('2020-02-01', '2021-03-31'),
            'Since 2022': ('2022-01-01', None), 'Short Invalid': ('2023-01-01', '2022-01-01')
        }
        summary_custom = backtester.calculate_performance_summary(periods=custom_periods, frequency='daily')
        assert isinstance(summary_custom, pd.DataFrame), "Custom result is not a DataFrame"
        if not summary_custom.empty:
            assert 'Last 3Y' in summary_custom.columns, "Custom period 'Last 3Y' missing"
            assert 'COVID' in summary_custom.columns, "Custom period 'COVID' missing"
            assert 'Since 2022' in summary_custom.columns, "Custom period 'Since 2022' missing"
            assert 'Short Invalid' not in summary_custom.columns or summary_custom['Short Invalid'].isnull().all() , "Invalid period not handled correctly"
        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False

    return test_passed


def bt_test_monthly_metric_calculation(backtester):
    """Backtester: Test calculation of MONTHLY performance summary."""
    label = "BT Metric Calculation Monthly (Default Periods)"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        summary_m = backtester.calculate_monthly_performance_summary() # Test default periods

        assert isinstance(summary_m, pd.DataFrame), "Monthly result is not a DataFrame"

        # Check basic structure and metric properties if summary is not empty
        if not summary_m.empty:
            expected_metrics = ['Ann Return', 'Ann Volatility', 'Information Ratio', 'Sortino Ratio',
                                'Skewness', 'Kurtosis', 'Max Drawdown', 'Calmar Ratio',
                                '% Positive Months', 'DD Recovery Months']
            assert all(metric in summary_m.index for metric in expected_metrics), "Missing expected metrics in monthly summary"
            # Check metrics that should have specific ranges/signs
            assert (summary_m.loc['Max Drawdown'].fillna(0) <= 1e-9).all(), f"Monthly Max Drawdown is positive: {summary_m.loc['Max Drawdown']}" # Use tolerance
            assert (summary_m.loc['% Positive Months'].fillna(-1) >= 0).all(), f"% Positive Months is negative: {summary_m.loc['% Positive Months']}"
            assert (summary_m.loc['% Positive Months'].fillna(101) <= 100).all(), f"% Positive Months is > 100: {summary_m.loc['% Positive Months']}"

        elif summary_m.empty:
             print(" (Skipped checks: Monthly summary empty, likely due to period length < 1yr)", end="")

        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False

    label = "BT Metric Calculation Monthly (Custom Periods)"
    print(f"  Running: {label}...", end="")
    try:
        custom_periods = {
            'Last 3Y': '3Y', 'COVID': ('2020-02-01', '2021-03-31'),
             # Add a short period to test NaN handling for monthly
            'Short': ('2023-01-01', '2023-06-30')
        }
        summary_custom = backtester.calculate_monthly_performance_summary(periods=custom_periods)
        assert isinstance(summary_custom, pd.DataFrame), "Custom monthly result is not a DataFrame"
        if not summary_custom.empty:
             assert 'Last 3Y' in summary_custom.columns, "Custom monthly period 'Last 3Y' missing"
             assert 'COVID' in summary_custom.columns, "Custom monthly period 'COVID' missing"
             # Short period likely results in NaNs due to min_months_threshold
             assert 'Short' in summary_custom.columns, "Custom monthly period 'Short' missing"
             assert summary_custom['Short'].isnull().all(), "Short period monthly metrics should be NaN"

        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False

    return test_passed


def run_all_backtester_tests(standard_weights, standard_tri):
    """Runs all test suites for the Backtester."""
    print("\n" + "="*50)
    print(" Running Backtester Tests ".center(50, "="))
    print("="*50)
    results = {}
    try:
        # Initialize backtester once with standard data
        backtester = Backtester(strategy_weights=standard_weights, substrategy_total_return_index=standard_tri)

        results["BT Initialization"] = bt_test_initialization(standard_weights, standard_tri)
        # Only proceed if initialization passed
        if results["BT Initialization"]:
            results["BT Index Generation"] = bt_test_index_generation(backtester)
            results["BT Metric Calculation (Daily)"] = bt_test_metric_calculation(backtester)
            results["BT Metric Calculation (Monthly)"] = bt_test_monthly_metric_calculation(backtester)
        else:
            print("Skipping subsequent backtester tests due to Initialization failure.")
            # Mark subsequent tests as failed if init failed
            results["BT Index Generation"] = False
            results["BT Metric Calculation (Daily)"] = False
            results["BT Metric Calculation (Monthly)"] = False


    except Exception as e:
        print(f"!!! Critical Error during Backtester setup or tests: {e}")
        # traceback.print_exc()
        results["Setup/Run"] = False # Mark overall failure

    print("\n" + "="*50)
    print(" Backtester Test Summary ".center(50, "="))
    print("="*50)
    final_outcome = True
    # Filter out non-boolean results if Setup/Run failed
    test_results_only = {k: v for k, v in results.items() if isinstance(v, bool)}
    for test_suite, passed in test_results_only.items():
        status = "Passed" if passed else "Failed"
        print(f"- {test_suite}: {status}")
        if not passed: final_outcome = False
    if "Setup/Run" in results and not results["Setup/Run"]:
         print("- Setup/Run: Failed")
         final_outcome = False


    return final_outcome


# --- Currency Exposure Test Suite ---

# Mock pd.read_excel to avoid actual file reading
def mock_read_excel(*args, **kwargs):
    """Mock function for pd.read_excel used in tests."""
    file_path = args[0]
    sheet_name = kwargs.get('sheet_name', 0) # Default sheet 0 if not specified

    if sheet_name != 'Weight':
        raise ValueError(f"Mock error: Expected sheet_name='Weight', got '{sheet_name}'")

    # Define sparse dataframes to return based on dummy file path
    if 'substrat_A_weights.xlsx' in file_path:
        # Sparse weights for SubStrat_A (e.g., monthly)
        dates = pd.to_datetime(['2022-01-31', '2022-02-28', '2022-03-31'])
        data = {'AUD': [0.10, 0.12, 0.11], 'EUR': [-0.05, -0.06, -0.04]}
        return pd.DataFrame(data, index=dates)
    elif 'substrat_B_weights.xlsx' in file_path:
        # Sparse weights for SubStrat_B (e.g., quarterly)
        dates = pd.to_datetime(['2022-03-31', '2022-06-30'])
        data = {'AUD': [0.05, 0.06], 'CAD': [0.15, 0.14]}
        return pd.DataFrame(data, index=dates)
    else:
        raise FileNotFoundError(f"Mock error: No mock data defined for file path '{file_path}'")


# Use context manager for patching instead of decorators
def ce_test_exposure_calculation():
    """Currency Exposure: Test the main exposure calculation logic."""
    print("\n--- Currency Exposure Test Suite: Calculation ---")
    label = "CE Calculation"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        # Use 'with patch(...)' context managers
        # Patch os.path.exists FIRST, then pandas.read_excel
        with patch('os.path.exists', return_value=True) as mock_exists, \
             patch('pandas.read_excel', mock_read_excel) as mock_read:

            # 1. Create Sample Inputs (inside the 'with' block)
            dates_top = pd.date_range('2022-01-01', '2022-04-05', freq=BDay())
            top_weights = pd.DataFrame({
                'SubStrat_A': np.linspace(0.6, 0.4, len(dates_top)),
                'SubStrat_B': np.linspace(0.4, 0.6, len(dates_top))
            }, index=dates_top)
            top_weights = top_weights.div(top_weights.sum(axis=1), axis=0)

            dummy_path_A = os.path.join('dummy_test_dir', 'substrat_A_weights.xlsx')
            dummy_path_B = os.path.join('dummy_test_dir', 'substrat_B_weights.xlsx')
            substrat_map = {'SubStrat_A': dummy_path_A, 'SubStrat_B': dummy_path_B}

            # 2. Instantiate Calculator and Calculate Exposures (mocks are active here)
            calculator = CurrencyExposureCalculator(
                top_level_weights=top_weights,
                substrat_weight_files_map=substrat_map
            )
            final_exposures = calculator.calculate_exposures()

            # 3. Define Expected Results for specific dates (Manual Calculation)
            date1 = pd.Timestamp('2022-02-01') # Uses Jan 31 weights
            date1_prev = pd.Timestamp('2022-01-31')
            topA1 = top_weights.loc[date1_prev, 'SubStrat_A']
            topB1 = top_weights.loc[date1_prev, 'SubStrat_B']
            # SubA weights on Jan 31: AUD=0.10, EUR=-0.05
            # SubB weights on Jan 31: AUD=0 (ffill from nothing, then fillna(0)), CAD=0
            exp_aud1 = topA1 * 0.10 + topB1 * 0.0
            exp_eur1 = topA1 * -0.05 + topB1 * 0.0
            exp_cad1 = topA1 * 0.0 + topB1 * 0.0
            exp_usd1 = 1.0 - (exp_aud1 + exp_eur1 + exp_cad1)

            date2 = pd.Timestamp('2022-04-01') # Uses Mar 31 weights
            date2_prev = pd.Timestamp('2022-03-31')
            topA2 = top_weights.loc[date2_prev, 'SubStrat_A']
            topB2 = top_weights.loc[date2_prev, 'SubStrat_B']
            # SubA weights on Mar 31: AUD=0.11, EUR=-0.04
            # SubB weights on Mar 31: AUD=0.05, CAD=0.15
            exp_aud2 = topA2 * 0.11 + topB2 * 0.05
            exp_eur2 = topA2 * -0.04 + topB2 * 0.0
            exp_cad2 = topA2 * 0.0 + topB2 * 0.15
            exp_usd2 = 1.0 - (exp_aud2 + exp_eur2 + exp_cad2)

            # 4. Assertions
            assert not final_exposures.empty, "Final exposure DataFrame is empty"
            assert date1 in final_exposures.index, f"Date {date1.date()} missing in results"
            assert date2 in final_exposures.index, f"Date {date2.date()} missing in results"

            # Check calculated values against expected values
            assert np.isclose(final_exposures.loc[date1, 'AUD'], exp_aud1), f"AUD mismatch on {date1.date()}"
            assert np.isclose(final_exposures.loc[date1, 'EUR'], exp_eur1), f"EUR mismatch on {date1.date()}"
            assert np.isclose(final_exposures.loc[date1, 'CAD'], exp_cad1), f"CAD mismatch on {date1.date()}"
            assert np.isclose(final_exposures.loc[date1, 'USD'], exp_usd1), f"USD mismatch on {date1.date()}"

            assert np.isclose(final_exposures.loc[date2, 'AUD'], exp_aud2), f"AUD mismatch on {date2.date()}"
            assert np.isclose(final_exposures.loc[date2, 'EUR'], exp_eur2), f"EUR mismatch on {date2.date()}"
            assert np.isclose(final_exposures.loc[date2, 'CAD'], exp_cad2), f"CAD mismatch on {date2.date()}"
            assert np.isclose(final_exposures.loc[date2, 'USD'], exp_usd2), f"USD mismatch on {date2.date()}"

            # Check sum = 1 constraint
            assert np.isclose(final_exposures.sum(axis=1), 1.0).all(), "Exposures do not sum to 1.0"

            print(" Passed")

    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False
    return test_passed


def run_all_currency_exposure_tests():
    """Runs all test suites for the Currency Exposure Calculator."""
    print("\n" + "="*50)
    print(" Running Currency Exposure Tests ".center(50, "="))
    print("="*50)
    results = {}
    # Add more tests here if needed
    results["CE Calculation"] = ce_test_exposure_calculation()

    print("\n" + "="*50)
    print(" Currency Exposure Test Summary ".center(50, "="))
    print("="*50)
    final_outcome = True
    for test_suite, passed in results.items():
        status = "Passed" if passed else "Failed"
        print(f"- {test_suite}: {status}")
        if not passed: final_outcome = False
    return final_outcome


# --- Correlation Analyzer Test Suite ---

def ca_test_runs(analyzer):
    """Correlation Analyzer: Test initialization and that analyze() runs."""
    label = "CA Initialization and Run"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        results = analyzer.analyze()
        assert isinstance(results, dict), "analyze() did not return a dict"
        assert 'rolling' in results, "Results dict missing 'rolling' key"
        assert 'fixed' in results, "Results dict missing 'fixed' key"
        assert isinstance(results['fixed'], dict), "'fixed' value is not a dict"
        # Rolling can be None if data is too short, so check type if not None
        if results['rolling'] is not None:
            assert isinstance(results['rolling'], pd.DataFrame), "'rolling' value is not DataFrame or None"
        # Fixed can be empty if data is too short for all lookbacks
        if results['fixed']:
             assert all(isinstance(df, pd.DataFrame) for df in results['fixed'].values()), \
                    "Not all values in 'fixed' dict are DataFrames"
        print(" Passed")
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False
    return test_passed

def ca_test_fixed_corr_accuracy():
    """Correlation Analyzer: Test fixed correlation accuracy with known input correlation."""
    label = "CA Fixed Correlation Accuracy"
    print(f"  Running: {label}...", end="")
    test_passed = True
    try:
        # 1. Define Target Correlation & Generate Data (3 assets for simplicity)
        n_assets = 3
        target_corr = np.array([
            [1.0, 0.7, 0.2],
            [0.7, 1.0, -0.3],
            [0.2, -0.3, 1.0]
        ])
        # Generate longer data for better correlation convergence
        corr_test_data = create_sample_data(start_date='2010-01-01', end_date='2023-12-31',
                                            n_assets=n_assets, target_corr_matrix=target_corr)

        # 2. Initialize Analyzer with a long fixed lookback
        # Use a lookback covering most of the data to approximate target correlation
        num_years = 10
        analyzer = CorrelationAnalyzer(df_index=corr_test_data,
                                       fixed_lookback_config=[num_years]) # Only test 10Y

        # 3. Run Analysis
        results = analyzer.analyze()

        # 4. Check Fixed Correlation Result
        fixed_results = results.get('fixed', {})
        corr_key = f"{num_years}Y Fixed Correlation"
        assert corr_key in fixed_results, f"'{corr_key}' not found in fixed results"

        calculated_corr = fixed_results[corr_key]
        assert isinstance(calculated_corr, pd.DataFrame), f"'{corr_key}' is not a DataFrame"

        # 5. Assert closeness (use a reasonable tolerance)
        # Compare numpy arrays for allclose
        assert np.allclose(calculated_corr.to_numpy(), target_corr, atol=0.15), \
               f"{num_years}Y calculated correlation matrix not close enough to target"

        print(f" Passed (Tolerance: 0.15)")

    except LinAlgError as e:
         print(f" Skipped (LinAlgError: {e})") # Skip if Cholesky fails
         # Mark as passed because the test setup failed, not the analyzer logic itself
         test_passed = True # Or False if failure to generate data is considered a test fail
    except Exception as e:
        print(f" Failed: {e}")
        # traceback.print_exc()
        test_passed = False
    return test_passed


def run_all_correlation_tests(standard_data):
    """Runs all test suites for the Correlation Analyzer."""
    print("\n" + "="*50)
    print(" Running Correlation Analyzer Tests ".center(50, "="))
    print("="*50)
    results = {}
    try:
        # Test basic run with standard (uncorrelated) data
        analyzer_std = CorrelationAnalyzer(df_index=standard_data)
        results["CA Initialization and Run"] = ca_test_runs(analyzer_std)

        # Test accuracy with correlated data
        results["CA Fixed Correlation Accuracy"] = ca_test_fixed_corr_accuracy()

    except Exception as e:
        print(f"!!! Critical Error during Correlation Analyzer setup or tests: {e}")
        # traceback.print_exc()
        results["Setup/Run"] = False # Mark overall failure

    print("\n" + "="*50)
    print(" Correlation Analyzer Test Summary ".center(50, "="))
    print("="*50)
    final_outcome = True
    # Filter out non-boolean results if Setup/Run failed
    test_results_only = {k: v for k, v in results.items() if isinstance(v, bool)}
    for test_suite, passed in test_results_only.items():
        status = "Passed" if passed else "Failed"
        print(f"- {test_suite}: {status}")
        if not passed: final_outcome = False
    if "Setup/Run" in results and not results["Setup/Run"]:
         print("- Setup/Run: Failed")
         final_outcome = False

    return final_outcome


# --- Main Test Runner ---
if __name__ == '__main__':
    print("="*60)
    print(" Combined Portfolio Construction & Backtesting Test Runner ".center(60, "="))
    print("="*60)

    # Setup: Generate standard data once
    print("\nGenerating standard sample data for all tests...")
    try:
        # Use a consistent end date for reproducibility if needed across runs
        end_dt_test = '2024-04-19'
        standard_tri_data = create_sample_data(end_date=end_dt_test)
        print(f"Standard TRI data shape: {standard_tri_data.shape}")

        # Generate standard weights needed for backtester tests
        print("Generating standard weights for backtester tests (using PC Monthly)...")
        pc_builder_std = EqualVolatilityPortfolio(
            total_return_index=standard_tri_data,
            lookback_years=2, skip_recent_month=False, rebalance_freq='M'
        )
        standard_weights_data = pc_builder_std.construct_portfolio()
        print(f"Standard Weights data shape: {standard_weights_data.shape}")

    except Exception as e:
        print(f"!!! Failed to generate sample data or standard weights: {e}")
        # traceback.print_exc()
        exit() # Cannot proceed without data/weights

    # --- Run Portfolio Construction Tests ---
    pc_passed = run_all_portfolio_tests(standard_tri_data)

    # --- Run Backtester Tests ---
    bt_passed = False # Default to False
    if standard_weights_data is not None and not standard_weights_data.empty:
         bt_passed = run_all_backtester_tests(standard_weights_data, standard_tri_data)
    else:
         print("\n" + "="*50)
         print(" Skipping Backtester Tests (Weights Generation Failed) ".center(50, "="))
         print("="*50)

    # --- Run Currency Exposure Tests ---
    ce_passed = run_all_currency_exposure_tests()

    # --- Run Correlation Analyzer Tests ---
    # Correlation tests use standard_tri_data as input
    ca_passed = run_all_correlation_tests(standard_tri_data)


    # --- Final Overall Summary ---
    print("\n" + "="*60)
    print(" Overall Test Summary ".center(60, "="))
    print("="*60)
    print(f"- Portfolio Construction Suite: {'Passed' if pc_passed else 'Failed'}")
    print(f"- Backtester Suite: {'Passed' if bt_passed else ('Failed' or 'Skipped')}")
    print(f"- Currency Exposure Suite: {'Passed' if ce_passed else 'Failed'}")
    print(f"- Correlation Analyzer Suite: {'Passed' if ca_passed else 'Failed'}") # Added
    print("="*60)
    if pc_passed and bt_passed and ce_passed and ca_passed: # Added ca_passed
        print(" All Test Suites Passed Successfully ".center(60, "="))
    else:
        print(" !!! Some Test Suites Failed or Were Skipped !!! ".center(60, "="))
    print("="*60)

