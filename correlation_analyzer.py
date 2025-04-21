import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

# --- Constants ---
# Moved inside class or passed via init where appropriate

class CorrelationAnalyzer:
    """
    Analyzes rolling and fixed-lookback correlations for asset returns derived
    from total return indices.

    Attributes:
        df_index (pd.DataFrame): Original DataFrame of total return indices.
        df_returns (pd.DataFrame): Calculated DataFrame of returns.
        freq_multiplier (int): Inferred or default frequency multiplier (periods per year).
        rolling_period_years (float): Lookback period for rolling correlations.
        fixed_lookback_config (List[int]): Lookback periods for fixed correlations.
    """
    # Default values as class attributes
    DEFAULT_ROLLING_YEARS: float = 1.0
    DEFAULT_FIXED_LOOKBACK_YEARS: List[int] = [1, 3, 5, 10]
    DEFAULT_DAILY_FREQ_MULT: int = 252
    DEFAULT_MONTHLY_FREQ_MULT: int = 12

    def __init__(self,
                 df_index: pd.DataFrame,
                 rolling_period_years: float = DEFAULT_ROLLING_YEARS,
                 fixed_lookback_config: List[int] = DEFAULT_FIXED_LOOKBACK_YEARS):
        """
        Initializes the CorrelationAnalyzer.

        Args:
            df_index (pd.DataFrame): DataFrame with DatetimeIndex and asset total return indices.
            rolling_period_years (float): Lookback period in years for rolling correlations.
            fixed_lookback_config (List[int]): List of lookback periods (in years) for fixed correlations.

        Raises:
            ValueError: If df_index is invalid (empty, wrong index type, unsorted).
        """
        self.rolling_period_years = rolling_period_years
        # Use copy to ensure config list is not modified externally
        self.fixed_lookback_config = list(fixed_lookback_config)

        # Validate and store input index DataFrame
        self._validate_input_index(df_index)
        self.df_index = df_index.copy() # Store a copy

        # Calculate and store returns
        self.df_returns = self._calculate_returns()
        if self.df_returns.empty:
            # Raise error if returns are empty after calculation and validation
             raise ValueError("Return calculation resulted in an empty DataFrame. Cannot proceed.")

        # Infer and store frequency multiplier
        self.freq_multiplier = self._infer_frequency_multiplier()
        if self.freq_multiplier is None:
            print(f"Could not infer frequency. Using default: {self.DEFAULT_DAILY_FREQ_MULT} (Daily assumption).")
            self.freq_multiplier = self.DEFAULT_DAILY_FREQ_MULT

        print(f"CorrelationAnalyzer initialized. Found {len(self.df_returns)} return periods.")


    def _validate_input_index(self, df_index: pd.DataFrame):
        """Validates the input DataFrame index."""
        if not isinstance(df_index, pd.DataFrame) or df_index.empty:
            raise ValueError("Input df_index must be a non-empty pandas DataFrame.")
        try:
            # Ensure index is DatetimeIndex (attempt conversion if not)
            if not isinstance(df_index.index, pd.DatetimeIndex):
                print("Warning: df_index index is not DatetimeIndex. Attempting conversion.")
                df_index.index = pd.to_datetime(df_index.index)
            # Ensure index is sorted
            if not df_index.index.is_monotonic_increasing:
                print("Warning: df_index index is not sorted. Sorting index.")
                df_index.sort_index(inplace=True) # Sort in place for validation check
        except Exception as e:
            raise ValueError(f"Could not process DataFrame index: {e}")
        if df_index.shape[0] < 2:
             raise ValueError("Input DataFrame must have at least 2 rows to calculate returns.")


    def _calculate_returns(self) -> pd.DataFrame:
        """
        Converts the stored total return index DataFrame to percentage returns.
        Internal method using self.df_index.
        """
        df_processed = self.df_index.copy() # Work on a copy

        # Basic NaN handling - forward fill before calculating returns
        if df_processed.isnull().values.any():
            print("Warning: Input DataFrame contains NaNs. Forward-filling before calculating returns.")
            df_processed = df_processed.ffill()
            # Check if NaNs remain at the beginning after ffill
            if df_processed.iloc[0].isnull().any():
                print("Warning: NaNs remain at the beginning after ffill. Filling with 0.")
                df_processed = df_processed.fillna(0) # Or consider other strategies like dropping cols/rows

        # pct_change() operates on all numeric columns by default
        df_returns = df_processed.pct_change()

        # Drop the first row which will always be NaN after pct_change
        return df_returns.iloc[1:]


    def _infer_frequency_multiplier(self) -> Optional[int]:
        """
        Infers frequency from the returns DatetimeIndex.
        Internal method using self.df_returns.index.
        """
        index = self.df_returns.index
        if not isinstance(index, pd.DatetimeIndex):
            print("Warning: Returns index is not DatetimeIndex, cannot infer frequency.")
            return None

        freq = pd.infer_freq(index)
        if freq:
            freq_str = freq.upper()
            if freq_str.startswith(('D', 'B')): # Daily or Business Day
                print(f"Inferred frequency: Daily/Business (using {self.DEFAULT_DAILY_FREQ_MULT} periods/year)")
                return self.DEFAULT_DAILY_FREQ_MULT
            elif freq_str.startswith('M'): # Month End or Start
                print(f"Inferred frequency: Monthly (using {self.DEFAULT_MONTHLY_FREQ_MULT} periods/year)")
                return self.DEFAULT_MONTHLY_FREQ_MULT
            # Add more frequencies (e.g., Quarterly 'Q') if needed
        print("Could not infer frequency or frequency is unsupported (Supports Daily/Business, Monthly).")
        return None


    def _calculate_rolling_correlations(self) -> Optional[pd.DataFrame]:
        """
        Calculates rolling correlations between all pairs of columns.
        Internal method using self.df_returns and instance attributes.
        """
        if self.df_returns.empty:
            print("Warning: Returns DataFrame is empty. Skipping rolling correlations.")
            return None

        window_size = int(self.rolling_period_years * self.freq_multiplier)
        min_periods_required = max(2, int(window_size * 0.6)) # Require reasonable data

        if len(self.df_returns) < min_periods_required:
            print(f"Warning: Not enough data ({len(self.df_returns)} periods) for rolling window "
                  f"(size {window_size}, min required {min_periods_required}). Skipping rolling correlations.")
            return None

        print(f"Calculating rolling {self.rolling_period_years}-year correlations (window: {window_size} periods)...")

        rolling_corr = self.df_returns.rolling(window=window_size, min_periods=min_periods_required).corr()

        if rolling_corr.isnull().all().all() if isinstance(rolling_corr, pd.DataFrame) else rolling_corr.isnull().all():
              print(f"Warning: Rolling correlation calculation resulted in all NaNs (window size: {window_size}). Check data overlap.")
              return None

        rolling_corr_unstacked = rolling_corr.unstack(level=1)

        if isinstance(rolling_corr_unstacked.columns, pd.MultiIndex):
              rolling_corr_unstacked.columns = [
                  f"{col[0]}_{col[1]}" for col in rolling_corr_unstacked.columns.values
              ]

        rolling_corr_unstacked.columns = rolling_corr_unstacked.columns.astype(str)
        self_corr_pattern = r'(.+)_\1$'
        rolling_corr_unstacked = rolling_corr_unstacked.loc[:, ~rolling_corr_unstacked.columns.str.match(self_corr_pattern)]

        print("Finished rolling correlations.")
        return rolling_corr_unstacked.dropna(how='all')


    def _calculate_fixed_lookback_correlation(self, years: int) -> Optional[pd.DataFrame]:
        """
        Calculates the standard correlation matrix for a fixed lookback period.
        Internal method using self.df_returns and instance attributes.
        """
        if self.df_returns.empty:
             print(f"Info: Input returns DataFrame is empty for {years}-year lookback. Skipping.")
             return None

        lookback_periods = int(years * self.freq_multiplier)
        min_periods_required = max(2, int(lookback_periods * 0.6))

        if len(self.df_returns) < min_periods_required:
            print(f"Info: Not enough data ({len(self.df_returns)} periods) for {years}-year lookback "
                  f"({lookback_periods} periods, min required {min_periods_required}). Skipping.")
            return None

        df_period = self.df_returns.iloc[-lookback_periods:]
        print(f"Calculating {years}-year standard correlation ({len(df_period)} periods ending {self.df_returns.index[-1].date()})...")

        corr_matrix = df_period.corr()

        if corr_matrix.isnull().all().all():
             print(f"Warning: {years}-year correlation matrix contains only NaNs.")
             return None

        return corr_matrix


    def analyze(self) -> Dict[str, Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Performs rolling and fixed lookback correlation analysis and returns results.

        Returns:
            Dict[str, Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
                A dictionary containing the results:
                {
                    'rolling': DataFrame of rolling correlations (or None),
                    'fixed': Dictionary of fixed correlation matrices {name: DataFrame} (or empty dict)
                }
        """
        print("\n--- Running Correlation Analysis ---")

        # Calculate Rolling Correlations
        rolling_corr_df = self._calculate_rolling_correlations()

        # Calculate Fixed Lookback Correlations
        fixed_correlations: Dict[str, pd.DataFrame] = {}
        for years in self.fixed_lookback_config:
            corr_matrix = self._calculate_fixed_lookback_correlation(years)
            if corr_matrix is not None:
                fixed_correlations[f"{years}Y Fixed Correlation"] = corr_matrix

        results = {
            'rolling': rolling_corr_df,
            'fixed': fixed_correlations
        }

        if rolling_corr_df is None and not fixed_correlations:
            print("No correlation results generated.")
        else:
             print("--- Correlation analysis complete ---")

        return results

# Note: No __main__ block here, this is intended as an importable module.
# Example usage would involve:
# 1. Loading/Generating df_index (DataFrame with total return indices)
# 2. Instantiating analyzer = CorrelationAnalyzer(df_index=your_data)
# 3. Calling results = analyzer.analyze()
# 4. Accessing results['rolling'] and results['fixed']
