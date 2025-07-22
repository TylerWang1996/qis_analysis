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

# --- NEW: Currency Grouping Definition ---
CURRENCY_GROUPS: Dict[str, List[str]] = {
    'USD': ['USD'],
    'CAD': ['CAD'],
    'JPY': ['JPY'],
    'European Majors': ['EUR', 'GBP', 'CHF'],
    'Scandinavians': ['SEK', 'NOK'],
    'Antipodeans': ['AUD', 'NZD'],
    'EM - EMEA': ['CZK', 'HUF', 'ILS', 'PLN', 'RON', 'RUB', 'TRY', 'ZAR'],
    'EM - LATAM': ['BRL', 'CLP', 'COP', 'MXN', 'PEN'],
    'EM - Asia': ['CNY', 'IDR', 'INR', 'KRW', 'MYR', 'PHP', 'SGD', 'THB', 'TWD']
}

# 5. Analysis Periods
ANALYSIS_PERIODS: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = {
    'Full Sample': None, 'Last 1Y': '1Y', 'Last 3Y': '3Y',
    'Last 5Y': '5Y', 'Last 10Y': '10Y'
}

# ==============================================================================
# --- NEW HELPER FUNCTIONS FOR AGGREGATION ---
# ==============================================================================

def aggregate_attribution_by_group(
    detailed_attribution_df: pd.DataFrame,
    currency_groups: Dict[str, List[str]],
    currency_classification_map: Dict[str, str]
    ) -> pd.DataFrame:
    """Aggregates a detailed currency attribution summary into custom groups."""
    if detailed_attribution_df.empty:
        return pd.DataFrame()

    print("   - Aggregating currency attribution into custom groups...")
    detailed_attribution_df_T = detailed_attribution_df.transpose()
    
    grouped_summary = {}
    all_grouped_ccys = set(c.upper() for grp in currency_groups.values() for c in grp)

    for group_name, currency_list in currency_groups.items():
        currency_list_upper = [c.upper() for c in currency_list]
        valid_ccys_in_group = [c for c in currency_list_upper if c in detailed_attribution_df_T.index]
        if valid_ccys_in_group:
            grouped_summary[group_name] = detailed_attribution_df_T.loc[valid_ccys_in_group].sum(axis=0)
        else:
            # *** FIX: Instead of scalar 0.0, create a Series with the correct index ***
            grouped_summary[group_name] = pd.Series(0.0, index=detailed_attribution_df_T.columns)

    other_em_ccys = []
    for ccy, classification in currency_classification_map.items():
        ccy_upper = ccy.upper()
        if (classification.upper() == 'EM' and 
            ccy_upper not in all_grouped_ccys and 
            ccy_upper in detailed_attribution_df_T.index):
            other_em_ccys.append(ccy_upper)
            
    if other_em_ccys:
        grouped_summary['EM - Other'] = detailed_attribution_df_T.loc[other_em_ccys].sum(axis=0)
    else:
        # *** FIX: Also apply the fix here for consistency ***
        grouped_summary['EM - Other'] = pd.Series(0.0, index=detailed_attribution_df_T.columns)
        
    final_df = pd.DataFrame(grouped_summary).transpose()
    if not final_df.empty:
        final_df.loc['Total', :] = final_df.sum(axis=0)
        
    return final_df


def aggregate_exposure_by_group(
    detailed_exposure_ts: pd.DataFrame,
    currency_groups: Dict[str, List[str]],
    currency_classification_map: Dict[str, str]
    ) -> pd.DataFrame:
    """Aggregates a detailed currency exposure time series into custom groups."""
    if detailed_exposure_ts.empty:
        return pd.DataFrame()

    print("   - Aggregating currency exposure into custom groups...")
    grouped_exposures_df = pd.DataFrame(index=detailed_exposure_ts.index)
    all_grouped_ccys = set(c.upper() for grp in currency_groups.values() for c in grp)

    for group_name, currency_list in currency_groups.items():
        currency_list_upper = [c.upper() for c in currency_list]
        valid_ccys_in_group = [c for c in currency_list_upper if c.upper() in detailed_exposure_ts.columns]
        if valid_ccys_in_group:
            grouped_exposures_df[group_name] = detailed_exposure_ts[valid_ccys_in_group].sum(axis=1)
        else:
            grouped_exposures_df[group_name] = 0.0

    other_em_ccys = []
    for ccy, classification in currency_classification_map.items():
        ccy_upper = ccy.upper()
        if (classification.upper() == 'EM' and 
            ccy_upper not in all_grouped_ccys and 
            ccy_upper in detailed_exposure_ts.columns):
            other_em_ccys.append(ccy_upper)
            
    if other_em_ccys:
        grouped_exposures_df['EM - Other'] = detailed_exposure_ts[other_em_ccys].sum(axis=1)
    else:
        grouped_exposures_df['EM - Other'] = 0.0
        
    return grouped_exposures_df


def calculate_netting_efficiency(
    final_exposures: pd.DataFrame,
    top_level_weights: pd.DataFrame,
    daily_substrat_weights_map: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
    """
    Calculates the netting efficiency by comparing the final gross exposure
    to the theoretical pre-netting gross exposure.

    Args:
        final_exposures (pd.DataFrame): The final net currency exposures of the top-level portfolio.
        top_level_weights (pd.DataFrame): The daily weights allocated to each substrategy.
        daily_substrat_weights_map (Dict[str, pd.DataFrame]): A map of substrat tickers
                                                              to their daily currency weights DataFrame.

    Returns:
        pd.DataFrame: A daily time series of pre-netting, post-netting, and ratio values.
    """
    print("   - Calculating Netting Efficiency Ratio...")
    if final_exposures.empty:
        return None

    # Numerator: Calculate the gross exposure of the final, netted portfolio
    post_netting_gross = final_exposures.abs().sum(axis=1)
    post_netting_gross.name = 'Post-Netting Gross Exposure'

    # Denominator: Calculate the theoretical gross exposure before any netting
    # Align indices for calculation
    common_index = final_exposures.index
    pre_netting_gross = pd.Series(0.0, index=common_index)
    shifted_top_weights = top_level_weights.shift(1) # Use t-1 weights

    for ticker, substrat_df in daily_substrat_weights_map.items():
        if ticker in shifted_top_weights.columns:
            # Gross exposure of the individual substrategy
            substrat_gross = substrat_df.abs().sum(axis=1)
            # Weight for this substrat
            top_weight = shifted_top_weights[ticker]
            # Align and calculate the weighted gross exposure contribution
            aligned_gross, aligned_weight = substrat_gross.align(top_weight, join='right', fill_value=0)
            pre_netting_gross = pre_netting_gross.add(aligned_gross * aligned_weight, fill_value=0)

    pre_netting_gross.name = 'Pre-Netting Gross Exposure'
    pre_netting_gross = pre_netting_gross.loc[common_index] # Ensure index matches

    # Calculate the ratio, handling division by zero
    ratio = (post_netting_gross / pre_netting_gross).fillna(0.0)
    ratio.name = 'Netting Efficiency Ratio'

    # Combine into a single DataFrame for output
    result_df = pd.concat([post_netting_gross, pre_netting_gross, ratio], axis=1)

    return result_df

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
        
        # <<< NEW: Calculate contribution summary >>>
        print("   - Calculating substrategy contribution summary...")
        substrat_contribution_summary = backtester.calculate_contribution_summary(
            periods=ANALYSIS_PERIODS
        )
        # <<< END NEW >>>

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
            # --- NEW: Call helper functions to aggregate the results ---
            if attr_results and 'Currency' in attr_results:
                # Use the detailed 'Currency' attribution summary as input
                detailed_currency_attribution = attr_results['Currency']
                grouped_attr_summary = aggregate_attribution_by_group(
                    detailed_currency_attribution, CURRENCY_GROUPS, CURRENCY_CLASSIFICATION_MAP
                )
            if currency_exposure_ts is not None:
                grouped_exposure_ts = aggregate_exposure_by_group(
                    currency_exposure_ts, CURRENCY_GROUPS, CURRENCY_CLASSIFICATION_MAP
                )
            if currency_exposure_ts is not None and hasattr(exposure_calculator, 'daily_substrat_weights_map'):
                netting_efficiency_df = calculate_netting_efficiency(
                final_exposures=currency_exposure_ts,
                top_level_weights=exposure_calculator.top_level_weights,
                daily_substrat_weights_map=exposure_calculator.daily_substrat_weights_map)
            print("Attribution & Exposure calculations complete.")
        except Exception as e:
            print(f"Error during attribution/exposure: {e}")
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
            
            # <<< NEW: Write Contribution Summary to a new sheet >>>
            if substrat_contribution_summary is not None:
                substrat_contribution_summary.to_excel(writer, sheet_name='Substrat Contribution', index=True)
                print("- Writing Substrat Contribution sheet...")
            else:
                print("- Skipping Substrat Contribution.")
            # <<< END NEW >>>
            
            # --- Writing Detailed Attribution and Exposure ---
            if attr_results:
                 for key, df in attr_results.items():
                      if not df.empty:
                          df.to_excel(writer, sheet_name=f"Detailed {key} Attribution")
                 print("- Writing detailed Attribution sheets...")
            if currency_exposure_ts is not None:
                 currency_exposure_ts.to_excel(writer, sheet_name='Detailed Ccy Exposure')
                 print("- Writing Detailed Ccy Exposure sheet...")
            
            # --- NEW: Write Grouped Summaries to new sheets ---
            if grouped_attr_summary is not None:
                grouped_attr_summary.to_excel(writer, sheet_name='Grouped Ccy Attribution')
                print("- Writing Grouped Ccy Attribution sheet...")
            if grouped_exposure_ts is not None:
                grouped_exposure_ts.to_excel(writer, sheet_name='Grouped Ccy Exposure')
                print("- Writing Grouped Ccy Exposure sheet...")
                
            if netting_efficiency_df is not None:
                netting_efficiency_df.to_excel(writer, sheet_name='Netting Efficiency')
                print("- Writing Netting Efficiency sheet...")
            else:
                print("- Skipping Netting Efficiency.")
            # --- END NEW ---
                        
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