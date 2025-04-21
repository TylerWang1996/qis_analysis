import pandas as pd
import numpy as np
import os
import shutil # For cleaning up temporary directory
import traceback # Added for debugging prints in run_simulation_and_analysis
from typing import Dict, List, Tuple, Optional, Union

# --- Import Project Modules ---
# Assume these are in the same directory or Python path
try:
    from portfolio_construction import EqualVolatilityPortfolio
    from backtester import Backtester
    from correlation_analyzer import CorrelationAnalyzer
    from currency_exposure import CurrencyExposureCalculator
    from performance_attribution import PerformanceAttribution
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all required module files (.py) for the analysis")
    print("(portfolio_construction, backtester, correlation_analyzer_v1, "
          "currency_exposure, performance_attribution) are accessible.")
    exit()

# ==============================================================================
# --- SIMULATION CONFIGURATION ---
# ==============================================================================
# General Simulation Parameters
SIM_START_DATE = '2018-01-01'
SIM_END_DATE = '2023-12-31'
N_ASSETS = 3 # Number of substrategies to simulate
ASSET_PREFIX = 'FX_Sub' # Prefix for simulated substrategy names
CURRENCIES = ['AUD', 'EUR', 'CAD', 'BRL', 'MXN', 'USD'] # Currencies for exposure/attribution
SEED = 42 # For reproducible random data

# File Paths for Simulated Data (will be created in a temporary directory)
TEMP_DIR = './_temp_simulated_data'
SIM_TRI_FILENAME = 'sim_substrat_tri.csv'
SIM_WEIGHT_FILENAME_TPL = 'sim_{}_weights.xlsx' # Template for weight files
SIM_ATTRIB_FILENAME_TPL = 'sim_{}_attrib.xlsx' # Template for attribution files
FINAL_REPORT_FILENAME = 'simulated_strategy_analysis_report.xlsx'

# Analysis Parameters (can be adjusted)
PC_PARAMS = {'lookback_years': 2, 'skip_recent_month': False, 'rebalance_freq': 'M'}
CORR_PARAMS = {'rolling_period_years': 1.0, 'fixed_lookback_config': [1, 3, 5]}
# Added 10Y to periods to match default fixed corr lookbacks
ANALYSIS_PERIODS = {'Full Sample': None, 'Last 1Y': '1Y', 'Last 3Y': '3Y', 'Last 5Y': '5Y', 'Last 10Y': '10Y'}
CCY_MAP = {'AUD': 'DM', 'EUR': 'DM', 'CAD': 'DM', 'USD': 'USD', 'BRL': 'EM', 'MXN': 'EM'}

# ==============================================================================
# --- Data Generation Functions ---
# ==============================================================================

def create_simulated_tri(start_date, end_date, asset_names, seed) -> pd.DataFrame:
    """Generates a simulated TRI DataFrame."""
    print("Generating simulated TRI data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_dates = len(dates)
    n_assets = len(asset_names)
    np.random.seed(seed)
    mean_returns = np.random.uniform(0.0001, 0.0005, n_assets)
    volatilities = np.random.uniform(0.008, 0.025, n_assets)
    log_returns_array = np.random.normal(loc=mean_returns.reshape(1, -1),
                                         scale=volatilities.reshape(1, -1),
                                         size=(n_dates, n_assets))
    log_returns = pd.DataFrame(log_returns_array, index=dates, columns=asset_names)
    tri_data = 100 * np.exp(log_returns.cumsum())
    return tri_data.ffill().fillna(100.0)

def create_sparse_excel(file_path: str, sheet_name: str, dates: pd.DatetimeIndex, columns: List[str], seed: int, is_attrib: bool):
    """
    Creates an Excel file with sparse data (simulating rebalance days).
    For weights (is_attrib=False), ensures data exists on the first day.
    For attribution (is_attrib=True), uses only month ends.
    """
    np.random.seed(seed)

    if is_attrib:
        # Simulate attribution data only on month ends for sparsity
        sparse_dates = dates[dates.is_month_end]
        if sparse_dates.empty: sparse_dates = dates[[-1]] # Ensure at least one date
        # Simulate dollar attribution (can be positive/negative)
        data = np.random.randn(len(sparse_dates), len(columns)) * 5 # Example scaling
        sparse_df = pd.DataFrame(data, index=sparse_dates, columns=columns)
    else: # Generating Weights file
        # Simulate weights on month ends AND the very first day
        month_end_dates = dates[dates.is_month_end]
        first_date = dates[[0]] # Get the first date as a DatetimeIndex
        # Combine first date with month-end dates, remove duplicates, sort
        sparse_dates = first_date.union(month_end_dates)
        if sparse_dates.empty: sparse_dates = dates[[-1]] # Fallback

        # Simulate weights (typically positive, maybe summing near 1 but not required)
        cols_to_generate = [c for c in columns if c != 'USD'] # Exclude USD column
        data = np.random.rand(len(sparse_dates), len(cols_to_generate)) * 0.5
        sparse_df = pd.DataFrame(data, index=sparse_dates, columns=cols_to_generate)

    # Write the sparse DataFrame to Excel
    try:
        # Ensure directory exists before writing
        dir_name = os.path.dirname(file_path)
        if dir_name: # Check if path includes a directory
             os.makedirs(dir_name, exist_ok=True)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            sparse_df.to_excel(writer, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error writing simulated Excel file {file_path}: {e}")
        raise # Re-raise error if file writing fails

# ==============================================================================
# --- Main Simulation and Analysis ---
# ==============================================================================

def run_simulation_and_analysis():
    """Generates simulated data, runs analysis modules, saves report."""

    # --- 1. Setup Environment ---
    print(f"Attempting to create temporary directory: {TEMP_DIR}")
    try:
        if os.path.exists(TEMP_DIR):
            print(f"Removing existing temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR) # Clean up previous run if necessary
        os.makedirs(TEMP_DIR)
        print(f"Temporary directory created: {TEMP_DIR}")
    except Exception as e:
        print(f"ERROR: Failed to create or clean temporary directory '{TEMP_DIR}': {e}")
        return

    sim_tri_path = os.path.join(TEMP_DIR, SIM_TRI_FILENAME)
    asset_tickers = [f"{ASSET_PREFIX}_{i+1}" for i in range(N_ASSETS)]

    # --- 2. Generate and Save Simulated Data ---
    try:
        # Generate TRI
        substrat_tri = create_simulated_tri(SIM_START_DATE, SIM_END_DATE, asset_tickers, SEED)
        substrat_tri.to_csv(sim_tri_path)
        print(f"Simulated TRI saved to {sim_tri_path}")

        # Generate sparse Weight and Attrib files
        sim_weight_map = {}
        sim_attrib_map = {}
        currencies_no_usd = [c for c in CURRENCIES if c.upper() != 'USD'] # Exclude USD for weight generation

        for i, ticker in enumerate(asset_tickers):
            # Weights File (will now include first day)
            weight_file = os.path.join(TEMP_DIR, SIM_WEIGHT_FILENAME_TPL.format(ticker))
            create_sparse_excel(weight_file, 'Weight', substrat_tri.index, currencies_no_usd, SEED + i + 1, is_attrib=False)
            sim_weight_map[ticker] = weight_file
            print(f"Simulated Weight file created: {weight_file}")

            # Attribution File (still only month-ends)
            attrib_file = os.path.join(TEMP_DIR, SIM_ATTRIB_FILENAME_TPL.format(ticker))
            create_sparse_excel(attrib_file, 'attrib', substrat_tri.index, CURRENCIES, SEED + i + 101, is_attrib=True)
            sim_attrib_map[ticker] = attrib_file
            print(f"Simulated Attribution file created: {attrib_file}")

    except Exception as e:
        print(f"Error during simulated data generation: {e}")
        traceback.print_exc()
        print("Aborting analysis.")
        # Cleanup might be needed here too if setup failed partially
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        return # Abort if data generation fails

    # --- 3. Run Analysis using Simulated Data ---
    results = {} # Dictionary to store results from each module
    try:
        print("\n--- Running Portfolio Construction ---")
        pc_builder = EqualVolatilityPortfolio(
            total_return_index=substrat_tri,
            lookback_years=int(PC_PARAMS['lookback_years']),
            skip_recent_month=bool(PC_PARAMS['skip_recent_month']),
            rebalance_freq=str(PC_PARAMS['rebalance_freq'])
        )
        results['strategy_weights'] = pc_builder.construct_portfolio()
        print("Portfolio Construction complete.")

        print("\n--- Running Backtester ---")
        backtester = Backtester(
            strategy_weights=results['strategy_weights'],
            substrategy_total_return_index=substrat_tri
        )
        results['daily_summary'] = backtester.calculate_performance_summary(
            periods=ANALYSIS_PERIODS, frequency='daily'
        )
        results['strategy_index_ts'] = backtester.get_strategy_index(frequency='daily')
        print("Backtesting complete.")

        print("\n--- Running Correlation Analysis ---")
        # Ensure fixed lookbacks match requested periods where possible
        fixed_lookbacks = [int(p[:-1]) for p in ANALYSIS_PERIODS.values() if isinstance(p, str) and p.endswith('Y')]
        fixed_lookbacks = sorted(list(set(fixed_lookbacks))) # Unique sorted years
        corr_analyzer = CorrelationAnalyzer(
            df_index=substrat_tri, # Use substrat TRI for correlation
            rolling_period_years=float(CORR_PARAMS['rolling_period_years']),
            fixed_lookback_config=fixed_lookbacks # Use years from ANALYSIS_PERIODS
        )
        corr_results = corr_analyzer.analyze()
        results['rolling_corr_ts'] = corr_results.get('rolling')
        results['fixed_corr_dict'] = corr_results.get('fixed')
        print("Correlation analysis complete.")

        print("\n--- Running Performance Attribution ---")
        pa_analyzer = PerformanceAttribution(
            top_level_weights=results['strategy_weights'],
            substrat_tri=substrat_tri,
            substrat_attrib_files_map=sim_attrib_map, # Use simulated attrib map
            currency_classification_map=CCY_MAP
        )
        attr_results = pa_analyzer.analyze_attribution(periods=ANALYSIS_PERIODS)
        # Combine Currency and DM/EM attribution for output sheet
        curr_df = attr_results.get('Currency')
        dmem_df = attr_results.get('DM_EM')
        if curr_df is not None and dmem_df is not None:
             combined_attr = pd.concat([curr_df.drop('Total', errors='ignore'), dmem_df.drop('Total', errors='ignore')], axis=0) # Concat rows, drop totals first
             combined_attr.loc['Total',:] = combined_attr.sum(axis=0) # Recalculate total if needed
        elif curr_df is not None: combined_attr = curr_df
        elif dmem_df is not None: combined_attr = dmem_df
        else: combined_attr = None
        results['sub_attr'] = attr_results.get('Substrategy')
        results['combined_curr_dm_em_attr'] = combined_attr
        print("Performance attribution complete.")

        print("\n--- Running Currency Exposure Calculation ---")
        exposure_calculator = CurrencyExposureCalculator(
            top_level_weights=results['strategy_weights'],
            substrat_weight_files_map=sim_weight_map # Use simulated weight map
        )
        results['currency_exposure_ts'] = exposure_calculator.calculate_exposures()
        print("Currency exposure calculation complete.")

        # --- 4. Write Combined Report ---
        print(f"\n--- Writing combined report to {FINAL_REPORT_FILENAME} ---")
        with pd.ExcelWriter(FINAL_REPORT_FILENAME, engine='openpyxl') as writer:
            start_row = 0
            # Sheet 1: Backtest Summary
            if results.get('daily_summary') is not None:
                pd.DataFrame(["Backtest Performance Summary (Daily)"]).to_excel(writer, sheet_name='Backtest Summary', startrow=start_row, index=False, header=False)
                start_row += 2
                results['daily_summary'].to_excel(writer, sheet_name='Backtest Summary', startrow=start_row)
                print("- Writing Backtest Summary sheet...")
            else: print("- Skipping Backtest Summary.")

            # Sheet 2: Backtest TRI
            if results.get('strategy_index_ts') is not None:
                results['strategy_index_ts'].to_excel(writer, sheet_name='Backtest TRI', index=True, header=['Strategy Index'])
                print("- Writing Backtest TRI sheet...")
            else: print("- Skipping Backtest TRI.")

            # Sheet 3: Correlation Summary (Fixed)
            start_row_corr = 0
            fixed_corr_dict = results.get('fixed_corr_dict')
            if fixed_corr_dict:
                print("- Writing Correlation Summary sheet...")
                for name, matrix in fixed_corr_dict.items():
                    pd.DataFrame([name]).to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr, index=False, header=False)
                    start_row_corr += 1
                    matrix.to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr)
                    start_row_corr += matrix.shape[0] + 2
            else: print("- Skipping Correlation Summary.")

            # Sheet 4: Rolling Correlation
            if results.get('rolling_corr_ts') is not None:
                results['rolling_corr_ts'].to_excel(writer, sheet_name='Rolling Correlation', index=True)
                print("- Writing Rolling Correlation sheet...")
            else: print("- Skipping Rolling Correlation.")

            # Sheet 5: Attribution Summary
            start_row_attr = 0
            print("- Writing Attribution Summary sheet...")
            sub_attr = results.get('sub_attr')
            combined_attr = results.get('combined_curr_dm_em_attr')
            if sub_attr is not None:
                 pd.DataFrame(["Substrategy Attribution (Summed $)"]).to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr, index=False, header=False)
                 start_row_attr += 2
                 sub_attr.to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr)
                 start_row_attr += sub_attr.shape[0] + 3
            else: print("  - Substrategy attribution data missing.")

            if combined_attr is not None:
                 pd.DataFrame(["Currency & DM/EM Attribution (Summed $)"]).to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr, index=False, header=False)
                 start_row_attr += 2
                 combined_attr.to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr)
            else: print("  - Currency/DM_EM attribution data missing.")

            # Sheet 6: Currency Exposure
            if results.get('currency_exposure_ts') is not None:
                results['currency_exposure_ts'].to_excel(writer, sheet_name='Currency Exposure', index=True)
                print("- Writing Currency Exposure sheet...")
            else: print("- Skipping Currency Exposure.")

        print(f"--- Analysis complete. Report saved to {FINAL_REPORT_FILENAME} ---")

    except ImportError:
         print("\nError: 'openpyxl' engine required for Excel export. Please install it (`pip install openpyxl`).")
    except Exception as e:
        print(f"\nAn error occurred during the analysis workflow: {e}")
        traceback.print_exc()
    finally:
        # --- 5. Cleanup ---
        print(f"\nCleaning up temporary directory: {TEMP_DIR}")
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
                print("Temporary directory removed.")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory '{TEMP_DIR}': {e}")
        else:
            print("Temporary directory not found (already cleaned up or never created).")


# --- Run the simulation and analysis ---
if __name__ == "__main__":
    run_simulation_and_analysis()
