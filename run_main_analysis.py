import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union # Import necessary types

# --- Import Project Modules ---
try:
    from portfolio_construction import EqualVolatilityPortfolio
    from backtester import Backtester
    from correlation_analyzer import CorrelationAnalyzer # Assuming OOP version
    from currency_exposure import CurrencyExposureCalculator
    from performance_attribution import PerformanceAttribution
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all required module files (.py) are in the same directory or Python path.")
    exit()

# ==============================================================================
# --- USER INPUTS: PLEASE MODIFY THE VALUES BELOW ---
# ==============================================================================

# 1. File Paths
# Path to your substrategy Total Return Index file (CSV or Excel)
# Example: '/path/to/your/data/substrategy_tri.csv' or '/path/to/your/data/substrategy_tri.xlsx'
SUBSTRAT_TRI_PATH = '/path/to/your/substrategy_tri_data.csv' # *** MODIFY THIS ***

# Dictionary mapping substrategy tickers (MUST match TRI columns) to their CURRENCY WEIGHT Excel files
# Example: {'FX_Carry': '/path/to/fx_carry_weights.xlsx', ...}
# Each Excel file must have a sheet named 'Weight' with Date index and Currency columns
CURRENCY_WEIGHT_FILES_MAP = { # *** MODIFY THIS ***
    'SubStrat_1': '/path/to/your/data/sub1_weights.xlsx',
    'SubStrat_2': '/path/to/your/data/sub2_weights.xlsx',
    # Add all substrategy tickers used in TRI and their weight file paths
}

# Dictionary mapping substrategy tickers to their CURRENCY ATTRIBUTION Excel files
# Example: {'FX_Carry': '/path/to/fx_carry_attrib.xlsx', ...}
# Each Excel file must have a sheet named 'attrib' with Date index and Currency columns (dollar P&L)
# Data should be sparse (only on rebalance days); missing days assumed zero attribution.
CURRENCY_ATTRIB_FILES_MAP = { # *** MODIFY THIS ***
    'SubStrat_1': '/path/to/your/data/sub1_attrib.xlsx',
    'SubStrat_2': '/path/to/your/data/sub2_attrib.xlsx',
    # Add all substrategy tickers used in TRI and their attribution file paths
}

# Path for the output Excel report file
OUTPUT_EXCEL_PATH = './strategy_analysis_report.xlsx' # *** MODIFY THIS ***

# 2. Portfolio Construction Parameters (Equal Volatility Example)
PORTFOLIO_CONSTRUCTION_PARAMS: Dict[str, Union[int, bool, str]] = {
    'lookback_years': 2,
    'skip_recent_month': False,
    'rebalance_freq': 'M' # 'M', 'Q', 'W', 'A'
}

# 3. Correlation Analysis Parameters
CORRELATION_PARAMS: Dict[str, Union[float, List[int]]] = {
    'rolling_period_years': 1.0, # For rolling correlation calculation
    'fixed_lookback_config': [1, 3, 5, 10] # Years for fixed correlation matrices
}

# 4. Performance Attribution Parameters
# Dictionary mapping currency codes (uppercase) to 'DM' or 'EM'
# Must cover ALL currencies found in your attribution files (except USD)
CURRENCY_CLASSIFICATION_MAP: Dict[str, str] = { # *** MODIFY THIS ***
    # DM Examples
    'AUD': 'DM', 'CAD': 'DM', 'CHF': 'DM', 'EUR': 'DM', 'GBP': 'DM',
    'JPY': 'DM', 'NOK': 'DM', 'NZD': 'DM', 'SEK': 'DM', 'USD': 'USD', # USD is special
    # EM Examples
    'BRL': 'EM', 'CLP': 'EM', 'CNY': 'EM', 'COP': 'EM', 'CZK': 'EM',
    'HUF': 'EM', 'IDR': 'EM', 'ILS': 'EM', 'INR': 'EM', 'KRW': 'EM',
    'MXN': 'EM', 'MYR': 'EM', 'PHP': 'EM', 'PLN': 'EM', 'RUB': 'EM',
    'SGD': 'EM', 'THB': 'EM', 'TRY': 'EM', 'TWD': 'EM', 'ZAR': 'EM'
}

# 5. Analysis Periods (Used for Backtester and Attribution)
# Define periods for summary tables. Format: {Name: Definition}
# Definition: None (Full Sample), 'XY'/'XM' (Relative), (start_str, end_str) (Specific)
ANALYSIS_PERIODS: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = {
    'Full Sample': None,
    'Last 1Y': '1Y',
    'Last 3Y': '3Y',
    'Last 5Y': '5Y',
    'Last 10Y': '10Y'
    # Add specific periods like: 'COVID': ('2020-02-01', '2021-03-31')
}

# ==============================================================================
# --- Main Analysis Workflow ---
# ==============================================================================

def load_tri_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads TRI data from CSV or Excel, basic validation."""
    print(f"Loading Substrategy TRI data from: {file_path}")
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, index_col=0, parse_dates=True)
        else:
            raise ValueError("Unsupported file type. Please use CSV or Excel.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index could not be parsed as DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            print("Warning: TRI index not sorted. Sorting...")
            df = df.sort_index()
        # Basic check for numeric data
        if not all(dtype.kind in 'ifc' for dtype in df.dtypes): # i=int, f=float, c=complex
             print("Warning: Non-numeric columns found in TRI data. Check input.")
        print(f"Loaded TRI data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: TRI file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading TRI data: {e}")
        return None

def run_analysis():
    """Executes the full analysis workflow."""
    print("Starting analysis workflow...")

    # --- Load Data ---
    substrat_tri = load_tri_data(SUBSTRAT_TRI_PATH)
    if substrat_tri is None:
        print("Aborting due to error loading TRI data.")
        return

    # --- Step 1: Portfolio Construction ---
    print("\n--- Running Portfolio Construction ---")
    strategy_weights = None
    try:
        portfolio_builder = EqualVolatilityPortfolio(
            total_return_index=substrat_tri,
            lookback_years=int(PORTFOLIO_CONSTRUCTION_PARAMS['lookback_years']),
            skip_recent_month=bool(PORTFOLIO_CONSTRUCTION_PARAMS['skip_recent_month']),
            rebalance_freq=str(PORTFOLIO_CONSTRUCTION_PARAMS['rebalance_freq'])
        )
        strategy_weights = portfolio_builder.construct_portfolio()
        print(f"Strategy weights generated. Shape: {strategy_weights.shape}")
    except Exception as e:
        print(f"Error during portfolio construction: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Decide whether to continue without weights or abort
        print("Aborting analysis.")
        return

    # --- Step 2: Backtesting ---
    print("\n--- Running Backtester ---")
    daily_summary = None
    strategy_index_ts = None
    try:
        backtester = Backtester(
            strategy_weights=strategy_weights,
            substrategy_total_return_index=substrat_tri
        )
        daily_summary = backtester.calculate_performance_summary(
            periods=ANALYSIS_PERIODS, frequency='daily'
        )
        strategy_index_ts = backtester.get_strategy_index(frequency='daily')
        print("Backtesting calculations complete.")
    except Exception as e:
        print(f"Error during backtesting: {e}")
        traceback.print_exc()
        # Continue analysis, results will be missing

    # --- Step 3: Correlation Analysis ---
    print("\n--- Running Correlation Analysis ---")
    rolling_corr_ts = None
    fixed_corr_dict = None
    try:
        # Ensure fixed lookbacks match requested periods where possible
        corr_analyzer = CorrelationAnalyzer(
            df_index=substrat_tri, # Analyze correlations of underlying substrats
            rolling_period_years=float(CORRELATION_PARAMS['rolling_period_years']),
            fixed_lookback_config=list(CORRELATION_PARAMS['fixed_lookback_config'])
        )
        corr_results = corr_analyzer.analyze()
        rolling_corr_ts = corr_results.get('rolling')
        fixed_corr_dict = corr_results.get('fixed')
        print("Correlation analysis complete.")
    except Exception as e:
        print(f"Error during correlation analysis: {e}")
        traceback.print_exc()
        # Continue analysis

    # --- Step 4: Performance Attribution ---
    print("\n--- Running Performance Attribution ---")
    sub_attr = None
    curr_attr = None
    dm_em_attr = None
    try:
        # Ensure required inputs exist
        if set(strategy_weights.columns) != set(CURRENCY_ATTRIB_FILES_MAP.keys()):
             raise ValueError("Mismatch between strategy weight columns and attribution file map keys.")

        pa_analyzer = PerformanceAttribution(
            top_level_weights=strategy_weights,
            substrat_tri=substrat_tri,
            substrat_attrib_files_map=CURRENCY_ATTRIB_FILES_MAP,
            currency_classification_map=CURRENCY_CLASSIFICATION_MAP
        )
        attr_results = pa_analyzer.analyze_attribution(periods=ANALYSIS_PERIODS)
        sub_attr = attr_results.get('Substrategy')
        curr_attr = attr_results.get('Currency')
        dm_em_attr = attr_results.get('DM_EM')
        print("Performance attribution complete.")
    except Exception as e:
        print(f"Error during performance attribution: {e}")
        print("Please check file paths, sheet names ('attrib'), and currency map.")
        traceback.print_exc()
        # Continue analysis

    # --- Step 5: Currency Exposure ---
    print("\n--- Running Currency Exposure Calculation ---")
    currency_exposure_ts = None
    try:
         # Ensure required inputs exist
        if set(strategy_weights.columns) != set(CURRENCY_WEIGHT_FILES_MAP.keys()):
             raise ValueError("Mismatch between strategy weight columns and currency weight file map keys.")

        exposure_calculator = CurrencyExposureCalculator(
            top_level_weights=strategy_weights,
            substrat_weight_files_map=CURRENCY_WEIGHT_FILES_MAP
        )
        currency_exposure_ts = exposure_calculator.calculate_exposures()
        print("Currency exposure calculation complete.")
    except Exception as e:
        print(f"Error during currency exposure calculation: {e}")
        print("Please check file paths and sheet names ('Weight').")
        traceback.print_exc()
        # Continue analysis

    # --- Step 6: Write to Excel ---
    print(f"\n--- Writing results to {OUTPUT_EXCEL_PATH} ---")
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
            start_row = 0

            # Sheet 1: Backtest Summary
            if daily_summary is not None:
                pd.DataFrame(["Backtest Performance Summary (Daily)"]).to_excel(writer, sheet_name='Backtest Summary', startrow=start_row, index=False, header=False)
                start_row += 2
                daily_summary.to_excel(writer, sheet_name='Backtest Summary', startrow=start_row)
                start_row += len(daily_summary) + 3
                print("- Writing Backtest Summary sheet...")
            else:
                print("- Skipping Backtest Summary (calculation failed).")

            # Sheet 2: Backtest TRI
            if strategy_index_ts is not None:
                strategy_index_ts.to_excel(writer, sheet_name='Backtest TRI', index=True, header=['Strategy Index'])
                print("- Writing Backtest TRI sheet...")
            else:
                print("- Skipping Backtest TRI (calculation failed).")

            # Sheet 3: Correlation Summary (Fixed)
            start_row_corr = 0
            if fixed_corr_dict:
                print("- Writing Correlation Summary sheet...")
                for name, matrix in fixed_corr_dict.items():
                    pd.DataFrame([name]).to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr, index=False, header=False)
                    start_row_corr += 1
                    matrix.to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr)
                    start_row_corr += matrix.shape[0] + 2 # Add space
            else:
                 print("- Skipping Correlation Summary (calculation failed or no results).")


            # Sheet 4: Rolling Correlation
            if rolling_corr_ts is not None:
                rolling_corr_ts.to_excel(writer, sheet_name='Rolling Correlation', index=True)
                print("- Writing Rolling Correlation sheet...")
            else:
                print("- Skipping Rolling Correlation (calculation failed or no results).")

            # Sheet 5: Attribution Summary
            start_row_attr = 0
            print("- Writing Attribution Summary sheet...")
            if sub_attr is not None:
                 pd.DataFrame(["Substrategy Attribution (Summed $)"]).to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr, index=False, header=False)
                 start_row_attr += 2
                 sub_attr.to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr)
                 start_row_attr += sub_attr.shape[0] + 3
            else:
                 print("  - Substrategy attribution data missing.")

            if curr_attr is not None:
                 pd.DataFrame(["Currency Attribution (Summed $)"]).to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr, index=False, header=False)
                 start_row_attr += 2
                 curr_attr.to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr)
                 start_row_attr += curr_attr.shape[0] + 3
            else:
                 print("  - Currency attribution data missing.")

            if dm_em_attr is not None:
                 pd.DataFrame(["DM vs EM Attribution (Summed $)"]).to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr, index=False, header=False)
                 start_row_attr += 2
                 dm_em_attr.to_excel(writer, sheet_name='Attribution Summary', startrow=start_row_attr)
                 # start_row_attr += dm_em_attr.shape[0] + 3 # No need if last table
            else:
                 print("  - DM/EM attribution data missing.")

            # Sheet 6: Currency Exposure
            if currency_exposure_ts is not None:
                currency_exposure_ts.to_excel(writer, sheet_name='Currency Exposure', index=True)
                print("- Writing Currency Exposure sheet...")
            else:
                print("- Skipping Currency Exposure (calculation failed).")

        print(f"--- Analysis complete. Results saved to {OUTPUT_EXCEL_PATH} ---")

    except ImportError:
         print("\nError: 'openpyxl' engine required for Excel export. Please install it (`pip install openpyxl`).")
    except Exception as e:
        print(f"\nError writing to Excel file {OUTPUT_EXCEL_PATH}: {e}")
        print("Please ensure the file path is valid, the file is not open elsewhere,")
        print("and you have write permissions to the directory.")


# --- Run the main analysis ---
if __name__ == "__main__":
    run_analysis()
