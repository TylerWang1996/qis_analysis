import pandas as pd
import numpy as np
import os
import traceback  # Import traceback for better error logging
from typing import Dict, List, Tuple, Optional, Union

# --- Import Project Modules ---
try:
    # Import Both Portfolio Construction Classes
    from portfolio_construction import EqualVolatilityPortfolio, VolatilityTargetPortfolio
    from backtester import Backtester
    from correlation_analyzer import CorrelationAnalyzer
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
SUBSTRAT_TRI_PATH = './simulated_tri_data.csv' # *** MODIFY THIS ***
CURRENCY_WEIGHT_FILES_MAP = {} # *** MODIFY THIS or leave empty if not used ***
CURRENCY_ATTRIB_FILES_MAP = {} # *** MODIFY THIS or leave empty if not used ***
OUTPUT_EXCEL_PATH = './strategy_analysis_report_vtp.xlsx' # *** MODIFY THIS ***

# 2. Portfolio Construction Parameters (Equal Volatility)
PORTFOLIO_CONSTRUCTION_PARAMS: Dict[str, Union[int, bool, str]] = {
    'lookback_years': 2,
    'skip_recent_month': False,
    'rebalance_freq': 'M'
}

# --- NEW: Volatility Targeting Parameters ---
# Set this to True to apply Volatility Targeting on top of Equal Vol
ENABLE_VOL_TARGETING: bool = True  # *** SET TO True TO ENABLE ***

VOL_TARGET_PARAMS: Dict[str, Union[float, int, bool, str]] = {
    'target_volatility': 0.10,       # e.g., 0.10 for 10% annualized vol
    'volatility_lookback_years': 1,  # Lookback for VTP's own vol calculation
    'skip_recent_month': False,      # Skip recent month for VTP's vol calculation
    'rebalance_freq': 'M',           # How often to adjust leverage ('D', 'W', 'M', 'Q', 'A')
    'max_leverage': 2.0,             # Max leverage (e.g., 200%)
    'min_leverage': 0.0              # Min leverage
}
# --- END NEW ---

# 3. Correlation Analysis Parameters
CORRELATION_PARAMS: Dict[str, Union[float, List[int]]] = {
    'rolling_period_years': 1.0,
    'fixed_lookback_config': [1, 3, 5, 10]
}

# 4. Performance Attribution Parameters
CURRENCY_CLASSIFICATION_MAP: Dict[str, str] = {
    'AUD': 'DM', 'CAD': 'DM', 'CHF': 'DM', 'EUR': 'DM', 'GBP': 'DM',
    'JPY': 'DM', 'NOK': 'DM', 'NZD': 'DM', 'SEK': 'DM', 'USD': 'USD',
    'BRL': 'EM', 'CLP': 'EM', 'CNY': 'EM', 'COP': 'EM', 'CZK': 'EM',
    'HUF': 'EM', 'IDR': 'EM', 'ILS': 'EM', 'INR': 'EM', 'KRW': 'EM',
    'MXN': 'EM', 'MYR': 'EM', 'PHP': 'EM', 'PLN': 'EM', 'RUB': 'EM',
    'SGD': 'EM', 'THB': 'EM', 'TRY': 'EM', 'TWD': 'EM', 'ZAR': 'EM'
}

# 5. Analysis Periods
ANALYSIS_PERIODS: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = {
    'Full Sample': None, 'Last 1Y': '1Y', 'Last 3Y': '3Y',
    'Last 5Y': '5Y', 'Last 10Y': '10Y'
}

# ==============================================================================
# --- Main Analysis Workflow ---
# ==============================================================================

def load_tri_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads TRI data from CSV or Excel, basic validation."""
    print(f"Loading Substrategy TRI data from: {file_path}")
    # Use a dummy if path not found for demo purposes
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Using dummy data.")
        dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='B')
        df = pd.DataFrame(np.random.randn(len(dates), 4) * 0.005 - 0.001,
                          index=dates, columns=['SubStrat_1', 'SubStrat_2', 'SubStrat_3', 'SubStrat_4'])
        df = 100 * (1 + df).cumprod()
        return df

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
        if not all(dtype.kind in 'ifc' for dtype in df.dtypes):
            print("Warning: Non-numeric columns found in TRI data. Check input.")
        print(f"Loaded TRI data shape: {df.shape}")
        return df.dropna() # Drop any NaNs before processing
    except FileNotFoundError:
        print(f"Error: TRI file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading TRI data: {e}")
        return None

def run_analysis():
    """Executes the full analysis workflow."""
    print("Starting analysis workflow...")

    substrat_tri = load_tri_data(SUBSTRAT_TRI_PATH)
    if substrat_tri is None:
        print("Aborting due to error loading TRI data.")
        return

    # --- Step 1: Portfolio Construction ---
    print("\n--- Running Portfolio Construction ---")
    strategy_weights = None
    try:
        # --- Always run Equal Vol first ---
        print("   - Building Equal Volatility base portfolio...")
        portfolio_builder = EqualVolatilityPortfolio(
            total_return_index=substrat_tri,
            lookback_years=int(PORTFOLIO_CONSTRUCTION_PARAMS['lookback_years']),
            skip_recent_month=bool(PORTFOLIO_CONSTRUCTION_PARAMS['skip_recent_month']),
            rebalance_freq=str(PORTFOLIO_CONSTRUCTION_PARAMS['rebalance_freq'])
        )
        base_strategy_weights = portfolio_builder.construct_portfolio()
        print(f"   - Base weights generated. Shape: {base_strategy_weights.shape}")

        # --- Conditionally run Vol Target ---
        if ENABLE_VOL_TARGETING:
            print("   - Applying Volatility Targeting...")
            vol_target_builder = VolatilityTargetPortfolio(
                equal_vol_weights=base_strategy_weights,
                underlying_asset_tri=substrat_tri,
                target_volatility=float(VOL_TARGET_PARAMS['target_volatility']),
                volatility_lookback_years=int(VOL_TARGET_PARAMS['volatility_lookback_years']),
                skip_recent_month=bool(VOL_TARGET_PARAMS['skip_recent_month']),
                rebalance_freq=str(VOL_TARGET_PARAMS['rebalance_freq']),
                max_leverage=float(VOL_TARGET_PARAMS['max_leverage']),
                min_leverage=float(VOL_TARGET_PARAMS['min_leverage'])
            )
            strategy_weights = vol_target_builder.construct_target_vol_weights()
            print(f"   - Volatility Targeted weights generated. Shape: {strategy_weights.shape}")
            # Drop initial NaNs from VTP (due to lookback) before proceeding
            strategy_weights = strategy_weights.dropna()
        else:
            print("   - Skipping Volatility Targeting (using base weights).")
            strategy_weights = base_strategy_weights

        if strategy_weights.empty:
             raise ValueError("Strategy weights are empty after construction step.")

    except Exception as e:
        print(f"Error during portfolio construction: {e}")
        traceback.print_exc()
        print("Aborting analysis.")
        return

    # --- Step 2: Backtesting ---
    print("\n--- Running Backtester ---")
    daily_summary = None
    strategy_index_ts = None
    try:
        # Ensure TRI aligns with weights (especially if VTP dropped NaNs)
        aligned_tri = substrat_tri.loc[strategy_weights.index]

        backtester = Backtester(
            strategy_weights=strategy_weights,
            substrategy_total_return_index=aligned_tri
        )
        daily_summary = backtester.calculate_performance_summary(
            periods=ANALYSIS_PERIODS, frequency='daily'
        )
        strategy_index_ts = backtester.get_strategy_index(frequency='daily')
        print("Backtesting calculations complete.")
    except Exception as e:
        print(f"Error during backtesting: {e}")
        traceback.print_exc()

    # --- Step 3: Correlation Analysis ---
    print("\n--- Running Correlation Analysis ---")
    rolling_corr_ts = None
    fixed_corr_dict = None
    try:
        corr_analyzer = CorrelationAnalyzer(
            df_index=substrat_tri,
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

    # --- Step 4 & 5: Attribution & Exposure (Skip if maps are empty) ---
    sub_attr, curr_attr, dm_em_attr = None, None, None
    if CURRENCY_ATTRIB_FILES_MAP:
        print("\n--- Running Performance Attribution ---")
        try:
            pa_analyzer = PerformanceAttribution(
                top_level_weights=strategy_weights, substrat_tri=substrat_tri,
                substrat_attrib_files_map=CURRENCY_ATTRIB_FILES_MAP,
                currency_classification_map=CURRENCY_CLASSIFICATION_MAP
            )
            attr_results = pa_analyzer.analyze_attribution(periods=ANALYSIS_PERIODS)
            sub_attr, curr_attr, dm_em_attr = attr_results.get('Substrategy'), attr_results.get('Currency'), attr_results.get('DM_EM')
            print("Performance attribution complete.")
        except Exception as e:
            print(f"Error during performance attribution: {e}")
            traceback.print_exc()
    else:
        print("\n--- Skipping Performance Attribution (no files provided) ---")

    currency_exposure_ts = None
    if CURRENCY_WEIGHT_FILES_MAP:
        print("\n--- Running Currency Exposure Calculation ---")
        try:
            exposure_calculator = CurrencyExposureCalculator(
                top_level_weights=strategy_weights,
                substrat_weight_files_map=CURRENCY_WEIGHT_FILES_MAP
            )
            currency_exposure_ts = exposure_calculator.calculate_exposures()
            print("Currency exposure calculation complete.")
        except Exception as e:
            print(f"Error during currency exposure calculation: {e}")
            traceback.print_exc()
    else:
        print("\n--- Skipping Currency Exposure (no files provided) ---")


    # --- Step 6: Write to Excel ---
    print(f"\n--- Writing results to {OUTPUT_EXCEL_PATH} ---")
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
            start_row = 0
            if daily_summary is not None:
                pd.DataFrame(["Backtest Performance Summary (Daily)"]).to_excel(writer, sheet_name='Backtest Summary', startrow=start_row, index=False, header=False)
                daily_summary.to_excel(writer, sheet_name='Backtest Summary', startrow=start_row+2)
                print("- Writing Backtest Summary sheet...")
            else: print("- Skipping Backtest Summary.")

            if strategy_index_ts is not None:
                strategy_index_ts.to_excel(writer, sheet_name='Backtest TRI', index=True, header=['Strategy Index'])
                print("- Writing Backtest TRI sheet...")
            else: print("- Skipping Backtest TRI.")
            
            # (Add writing logic for Correlation, Attribution, Exposure as before)
            # Example for Correlation:
            start_row_corr = 0
            if fixed_corr_dict:
                print("- Writing Correlation Summary sheet...")
                for name, matrix in fixed_corr_dict.items():
                    pd.DataFrame([name]).to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr, index=False, header=False)
                    matrix.to_excel(writer, sheet_name='Correlation Summary', startrow=start_row_corr+1)
                    start_row_corr += matrix.shape[0] + 3
            else: print("- Skipping Correlation Summary.")
            
            if rolling_corr_ts is not None:
                 rolling_corr_ts.to_excel(writer, sheet_name='Rolling Correlation', index=True)
                 print("- Writing Rolling Correlation sheet...")
            else: print("- Skipping Rolling Correlation.")
            
            # Add other sheets here...

        print(f"--- Analysis complete. Results saved to {OUTPUT_EXCEL_PATH} ---")

    except ImportError:
        print("\nError: 'openpyxl' engine required. Please install (`pip install openpyxl`).")
    except Exception as e:
        print(f"\nError writing to Excel file: {e}")

# --- Run the main analysis ---
if __name__ == "__main__":
    run_analysis()