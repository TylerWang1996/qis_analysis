import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os # Added for file existence check

class CurrencyExposureCalculator:
    """
    Calculates the total net currency exposures of a portfolio strategy,
    given top-level weights on substrategies and the currency weights
    within each substrategy.
    """
    def __init__(self,
                 top_level_weights: pd.DataFrame,
                 substrat_weight_files_map: Dict[str, str]):
        """
        Initializes the CurrencyExposureCalculator.

        Args:
            top_level_weights (pd.DataFrame): DataFrame of daily EOD weights
                for the top-level strategy. Index=DateTimeIndex,
                Columns=Substrategy tickers.
            substrat_weight_files_map (Dict[str, str]): Dictionary mapping
                substrategy tickers (must match columns in top_level_weights)
                to the full file path of their corresponding Excel file
                containing currency weights.

        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If indices are not DatetimeIndex or not sorted,
                        or if top_level_weights columns don't match map keys.
        """
        if not isinstance(top_level_weights, pd.DataFrame):
            raise TypeError("top_level_weights must be a pandas DataFrame.")
        if not isinstance(top_level_weights.index, pd.DatetimeIndex):
            raise TypeError("top_level_weights index must be a DatetimeIndex.")
        if not top_level_weights.index.is_monotonic_increasing:
             raise ValueError("top_level_weights index must be sorted ascending.")
        if not isinstance(substrat_weight_files_map, dict):
             raise TypeError("substrat_weight_files_map must be a dictionary.")
        if not set(top_level_weights.columns) == set(substrat_weight_files_map.keys()):
             # Ensure keys exactly match columns for clarity, could relax later if needed
             raise ValueError("Columns in top_level_weights must exactly match the keys "
                              "in substrat_weight_files_map.")

        self.top_level_weights = top_level_weights.copy()
        self.substrat_files = substrat_weight_files_map
        self._validate_inputs() # Basic validation done, further checks if needed

    def _validate_inputs(self):
        """Placeholder for additional input validation if needed."""
        # Example: Check if files in the map actually exist
        for ticker, file_path in self.substrat_files.items():
             if not os.path.exists(file_path):
                  print(f"Warning: File not found for {ticker}: {file_path}")
                  # Decide if this should be a warning or raise an error
                  # raise FileNotFoundError(f"File not found for {ticker}: {file_path}")
        pass

    def _load_and_prepare_substrat_weights(self, common_dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """
        Loads currency weights for each substrategy from Excel files,
        validates, reindexes to daily frequency, and forward-fills.

        Args:
            common_dates (pd.DatetimeIndex): The target daily date index for alignment.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping substrategy tickers
                                     to their daily currency weight DataFrames.

        Raises:
            FileNotFoundError: If an Excel file specified in the map doesn't exist.
            ValueError: If the 'Weight' sheet is missing or data is invalid.
            RuntimeError: For other unexpected loading errors.
        """
        daily_substrat_weights = {}
        required_tickers = set(self.top_level_weights.columns)

        for ticker, file_path in self.substrat_files.items():
            if ticker not in required_tickers: # Should not happen due to init check, but safe
                continue

            if not os.path.exists(file_path):
                 # Raise error if file is missing for a required ticker
                 raise FileNotFoundError(f"Required weight file not found for substrat {ticker}: {file_path}")

            try:
                # Read specific sheet 'Weight', assume date is the first column (index_col=0)
                df = pd.read_excel(file_path, sheet_name='Weight', index_col=0, parse_dates=True)

                # --- Data Validation ---
                if not isinstance(df.index, pd.DatetimeIndex):
                     raise ValueError(f"Index of sheet 'Weight' in {file_path} for {ticker} is not DatetimeIndex.")
                if df.index.has_duplicates:
                     print(f"Warning: Duplicate dates found in weights for {ticker}. Keeping first entry.")
                     df = df[~df.index.duplicated(keep='first')]
                if df.isnull().values.any():
                     print(f"Warning: NaNs found in weights for {ticker} ({file_path}). Check source data. Forward-filling NaNs.")
                     # NaNs will be handled by ffill after reindexing

                # Ensure index is sorted
                df = df.sort_index()

                # Convert weights to numeric, coercing errors (e.g., text) to NaN
                df = df.apply(pd.to_numeric, errors='coerce')
                if df.isnull().values.any():
                     print(f"Warning: Non-numeric values found and converted to NaN in weights for {ticker} ({file_path}).")

                # --- Reindex & Forward Fill ---
                # Reindex to the common daily dates, keeping existing values
                # Then forward fill to propagate weights between rebalance dates
                df_reindexed = df.reindex(common_dates)
                df_filled = df_reindexed.ffill()

                # Optional: Handle initial NaNs if the first common_date is before the first substrat date
                # df_filled = df_filled.bfill() # Use with caution, might imply lookahead

                daily_substrat_weights[ticker] = df_filled

            except ValueError as e: # Catch sheet name errors, index errors etc.
                 raise ValueError(f"Error reading/validating sheet 'Weight' for {ticker} from {file_path}: {e}")
            except Exception as e:
                 raise RuntimeError(f"Unexpected error loading weights for {ticker} from {file_path}: {e}")

        # Final check: ensure we have data for all required tickers after loading
        loaded_tickers = set(daily_substrat_weights.keys())
        if loaded_tickers != required_tickers:
             missing = required_tickers - loaded_tickers
             # This should ideally not happen if FileNotFoundError is raised above
             raise RuntimeError(f"Failed to load weight data for required substrats: {missing}")

        return daily_substrat_weights


    def calculate_exposures(self) -> pd.DataFrame:
        """
        Calculates the daily total net currency exposures for the top-level strategy.

        Returns:
            pd.DataFrame: DataFrame indexed by trading date, with columns for each
                          currency (e.g., AUD, CAD, EUR, ..., USD), showing the
                          total net percentage exposure.
        """

        # Use the index from top_level_weights as the basis for dates
        common_dates = self.top_level_weights.index

        # Load and prepare all substrat weights aligned to these daily dates
        try:
            daily_substrat_weights_map = self._load_and_prepare_substrat_weights(common_dates)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
             print(f"Error preparing substrategy weights: {e}")
             # Return empty DataFrame or re-raise depending on desired handling
             return pd.DataFrame()


        # --- Alignment and Calculation ---
        # Shift top-level weights (use t-1 EOD weight for exposure on day t)
        shifted_top_weights = self.top_level_weights.shift(1)

        # Prepare a structure to hold results
        # Get all unique currency columns across all substrategies
        all_currencies = set()
        for df in daily_substrat_weights_map.values():
            # Exclude non-numeric columns if any slipped through (shouldn't happen with coerce)
            numeric_cols = df.select_dtypes(include=np.number).columns
            all_currencies.update(numeric_cols)
        all_currencies = sorted(list(all_currencies))
        if not all_currencies:
             print("Warning: No numeric currency columns found in substrategy weights.")
             return pd.DataFrame()

        # Use the date index resulting from the shift (drops first day)
        # Align calculation to dates where shifted top weights are not NaN
        result_index = shifted_top_weights.dropna(how='all').index
        if result_index.empty:
             print("Warning: No valid dates remain after shifting top-level weights.")
             return pd.DataFrame()

        # Initialize total exposure DataFrame
        total_exposure_df = pd.DataFrame(0.0, index=result_index, columns=all_currencies)

        # Iterate through each substrategy ticker present in the top-level weights
        for ticker in self.top_level_weights.columns:
            if ticker not in daily_substrat_weights_map:
                 print(f"Info: Skipping {ticker} as its weights were not loaded.")
                 continue # Should not happen if checks passed, but safety

            substrat_weights_df = daily_substrat_weights_map[ticker]

            # Shift substrat weights (use t-1 state for exposure on day t)
            shifted_substrat_weights = substrat_weights_df.shift(1)

            # Align the shifted top weight for this ticker with its shifted substrat weights
            # Use the common result_index derived from shifted_top_weights
            aligned_top_w = shifted_top_weights.loc[result_index, [ticker]]
            aligned_sub_w = shifted_substrat_weights.loc[result_index, :] # Align to same index

            # Multiply top weight (Series) by substrat currency weights (DataFrame)
            # Need element-wise multiplication, broadcasting the top weight across currency columns
            # Ensure alignment again just before multiplication
            aligned_top_w, aligned_sub_w = aligned_top_w.align(aligned_sub_w, join='inner', axis=0)

            if aligned_top_w.empty or aligned_sub_w.empty:
                 # This might happen if a substrat weight file starts much later
                 print(f"Warning: No overlapping dates for {ticker} after final alignment. Skipping contribution.")
                 continue

            # Perform the multiplication
            weighted_exposure = aligned_sub_w.multiply(aligned_top_w[ticker], axis=0)

            # Add this substrat's contribution to the total exposure
            # Use add with fill_value=0; align columns automatically
            total_exposure_df = total_exposure_df.add(weighted_exposure, fill_value=0)


        # --- Calculate USD Position ---
        # Sum exposures across *all* loaded non-USD currencies for each day
        non_usd_columns = [c for c in total_exposure_df.columns if c.upper() != 'USD']
        # Assumption: USD position makes the total portfolio sum to 100% (1.0)
        total_exposure_df['USD'] = 1.0 - total_exposure_df[non_usd_columns].sum(axis=1)

        # --- Final Formatting ---
        # Ensure USD column exists even if not originally present
        if 'USD' not in all_currencies:
             all_currencies.append('USD')

        # Reorder columns alphabetically (optional) and fill any NaNs (e.g., from sum) with 0
        final_df = total_exposure_df.reindex(columns=sorted(all_currencies)).fillna(0.0)

        # Drop rows where all exposures are zero (optional, might happen at start)
        # final_df = final_df.loc[(final_df != 0).any(axis=1)]

        return final_df

# Note: No __main__ block here, this is intended as an importable module.
# Example usage would involve:
# 1. Loading/Generating top_level_weights (e.g., from EqualVolatilityPortfolio)
# 2. Creating the substrat_weight_files_map dictionary
# 3. Ensuring the Excel files exist and are formatted correctly
# 4. Instantiating calculator = CurrencyExposureCalculator(...)
# 5. Calling exposures = calculator.calculate_exposures()
