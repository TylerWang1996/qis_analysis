import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os

class PerformanceAttribution:
    """
    Calculates performance attribution by substrategy and currency groupings
    (e.g., DM vs EM) over specified periods.

    Assumes sparse dollar attribution data for currencies within substrategies
    is provided periodically, and attribution is zero on intermediate days.
    """
    def __init__(self,
                 top_level_weights: pd.DataFrame,
                 substrat_tri: pd.DataFrame,
                 substrat_attrib_files_map: Dict[str, str],
                 currency_classification_map: Dict[str, str],
                 trading_days_per_year: int = 252):
        """
        Initializes the PerformanceAttribution calculator.

        Args:
            top_level_weights (pd.DataFrame): Daily EOD weights for the top-level strategy.
                                              Index=DateTimeIndex, Columns=Substrategy tickers.
            substrat_tri (pd.DataFrame): Total return indices for the substrategies.
                                         Index=DateTimeIndex, Columns=Substrategy tickers.
            substrat_attrib_files_map (Dict[str, str]): Dictionary mapping substrategy tickers
                                                        to the file path of their Excel file
                                                        containing currency attribution data
                                                        in a sheet named 'attrib'.
            currency_classification_map (Dict[str, str]): Dictionary mapping currency codes
                                                         (e.g., 'AUD', 'EUR', 'BRL') to
                                                         classification strings (e.g., 'DM', 'EM').
                                                         Must include all currencies found in
                                                         attribution files.
            trading_days_per_year (int): Trading days per year for reference. Defaults to 252.


        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If indices are invalid, columns/keys mismatch, or map is incomplete.
            FileNotFoundError: If an attribution file is missing.
        """
        # --- Input Validation ---
        if not isinstance(top_level_weights, pd.DataFrame) or \
           not isinstance(substrat_tri, pd.DataFrame):
            raise TypeError("top_level_weights and substrat_tri must be pandas DataFrames.")
        if not isinstance(top_level_weights.index, pd.DatetimeIndex) or \
           not isinstance(substrat_tri.index, pd.DatetimeIndex):
            raise TypeError("Input indices must be DatetimeIndex.")
        if not top_level_weights.index.is_monotonic_increasing or \
           not substrat_tri.index.is_monotonic_increasing:
             raise ValueError("Input indices must be sorted ascending.")
        if not isinstance(substrat_attrib_files_map, dict) or \
           not isinstance(currency_classification_map, dict):
             raise TypeError("substrat_attrib_files_map and currency_classification_map must be dictionaries.")
        if not set(top_level_weights.columns) == set(substrat_attrib_files_map.keys()):
             raise ValueError("top_level_weights columns must exactly match substrat_attrib_files_map keys.")
        if not set(top_level_weights.columns).issubset(set(substrat_tri.columns)):
             raise ValueError("top_level_weights columns must be a subset of substrat_tri columns.")

        self.top_level_weights = top_level_weights.copy()
        self.substrat_tri = substrat_tri.copy()
        self.substrat_files = substrat_attrib_files_map
        self.currency_map = {k.upper(): v.upper() for k, v in currency_classification_map.items()} # Ensure uppercase keys/values
        self.trading_days_per_year = trading_days_per_year
        print("PerformanceAttribution initialized.")

        # --- Prepare Intermediate Data ---
        print("Calculating substrategy returns...")
        self._substrat_returns = self._calculate_substrat_returns()

        # Determine common date range based on weights and returns needed for calc
        self._common_dates = self.top_level_weights.index.intersection(self._substrat_returns.index)
        if self._common_dates.empty:
             raise ValueError("No common dates found between top-level weights and substrat returns.")
        print(f"Common date range for analysis: {self._common_dates[0].date()} to {self._common_dates[-1].date()}")

        print("Loading and preparing currency attribution data...")
        self._daily_currency_attrib_map = self._load_and_prepare_attrib_data(self._common_dates)

        # Validate currency map covers all loaded currencies
        self._validate_currency_map()

        print("Calculating daily contributions...")
        # Calculate all daily contributions upon initialization
        self._daily_sub_contrib, self._daily_curr_contrib, self._daily_dm_em_contrib = self._calculate_daily_contributions()
        print("Daily contributions calculated.")


    def _validate_currency_map(self):
        """Checks if the currency map covers all currencies found in attribution data."""
        all_attrib_currencies = set()
        for df in self._daily_currency_attrib_map.values():
            # Ensure columns are strings before converting to upper
            all_attrib_currencies.update([str(c).upper() for c in df.columns])

        missing_in_map = all_attrib_currencies - set(self.currency_map.keys())
        # Allow USD to be missing from the map (it won't be classified as DM/EM)
        missing_in_map -= {'USD'}

        if missing_in_map:
            raise ValueError(f"Missing currency classifications for: {missing_in_map}. "
                             "Please update currency_classification_map.")

    def _calculate_substrat_returns(self) -> pd.DataFrame:
        """Calculates daily returns for substrategies."""
        if self.substrat_tri.shape[0] < 2:
            raise ValueError("Substrategy TRI needs at least 2 rows to calculate returns.")
        # Align TRI columns to top-level weight columns before calculating returns
        aligned_tri = self.substrat_tri[self.top_level_weights.columns]
        returns = aligned_tri.pct_change()
        return returns.iloc[1:] # Drop first NaN row

    def _load_and_prepare_attrib_data(self, common_dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """
        Loads currency attribution data for each substrategy from Excel files,
        validates, reindexes to daily frequency, and fills missing days with 0.
        """
        daily_attrib_data = {}
        for ticker, file_path in self.substrat_files.items():
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Required attribution file not found for substrat {ticker}: {file_path}")

            try:
                df = pd.read_excel(file_path, sheet_name='attrib', index_col=0, parse_dates=True)

                if not isinstance(df.index, pd.DatetimeIndex):
                     raise ValueError(f"Index of sheet 'attrib' in {file_path} for {ticker} is not DatetimeIndex.")
                if df.index.has_duplicates:
                     print(f"Warning: Duplicate dates found in attribution for {ticker}. Keeping first entry.")
                     df = df[~df.index.duplicated(keep='first')]

                df = df.sort_index()
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0) # Coerce non-numeric to NaN, then fill NaN with 0

                # --- Reindex & Fillna(0) ---
                # Reindex to common daily dates. Dates not present in original df get NaN.
                df_reindexed = df.reindex(common_dates)
                # Fill NaN values (days not in original file) with 0, per user requirement.
                df_filled = df_reindexed.fillna(0.0)

                daily_attrib_data[ticker] = df_filled

            except ValueError as e:
                 raise ValueError(f"Error reading/validating sheet 'attrib' for {ticker} from {file_path}: {e}")
            except Exception as e:
                 raise RuntimeError(f"Unexpected error loading attribution for {ticker} from {file_path}: {e}")

        return daily_attrib_data

    def _calculate_daily_contributions(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculates daily contributions for Substrategy, Currency, and DM/EM."""

        # --- 1. Substrategy Contributions ---
        # Align shifted top_weights with substrat_returns on the common index
        shifted_top_weights = self.top_level_weights.shift(1).reindex(self._common_dates)
        aligned_returns = self._substrat_returns.reindex(self._common_dates)

        # Element-wise multiplication after ensuring alignment
        # Result: DataFrame [Date x Substrategy], value = W_s(t-1) * R_s(t)
        daily_sub_contrib = (shifted_top_weights * aligned_returns).dropna(how='all') # Drop rows that are all NaN (e.g., first day)

        # --- 2. Currency Contributions ---
        # Use the prepared daily attribution map (already reindexed and filled with 0)
        # Align shifted top_weights with this map
        all_currencies = set()
        for df in self._daily_currency_attrib_map.values():
            all_currencies.update([str(c).upper() for c in df.columns]) # Ensure string and uppercase
        all_currencies = sorted(list(all_currencies))

        # Use index from daily_sub_contrib to ensure alignment after shift/dropna
        result_index = daily_sub_contrib.index
        daily_curr_contrib = pd.DataFrame(0.0, index=result_index, columns=all_currencies)

        for ticker in self.top_level_weights.columns:
            if ticker not in self._daily_currency_attrib_map:
                 continue # Should not happen if init checks passed

            # Get prepared daily attribution data for this substrat
            attrib_df = self._daily_currency_attrib_map[ticker]
            # Align attribution data to the result index (dates where sub contrib exists)
            aligned_attrib = attrib_df.reindex(result_index).fillna(0.0) # Ensure fillna just in case

            # Get shifted top weight for this ticker, aligned to result index
            aligned_top_w = shifted_top_weights.loc[result_index, [ticker]]

            # Multiply top weight (Series) by currency attributions (DataFrame)
            # Note: Based on user description, the 'attrib' data is already dollar attribution.
            # The simplest interpretation is to weight the *substrategy's* dollar attribution
            # by the *top-level weight* allocated to that substrategy.
            # Contrib_c(t) = Sum_s [ W_s(t-1) * Attrib_sc(t) ]
            # Here Attrib_sc(t) is the daily dollar attribution (filled with 0)
            weighted_attrib = aligned_attrib.multiply(aligned_top_w[ticker], axis=0)

            # Add contribution
            daily_curr_contrib = daily_curr_contrib.add(weighted_attrib, fill_value=0)

        # Ensure columns are uppercase
        daily_curr_contrib.columns = [str(c).upper() for c in daily_curr_contrib.columns]


        # --- 3. DM/EM Contributions ---
        daily_dm_em_contrib = pd.DataFrame(0.0, index=result_index, columns=['DM', 'EM', 'Other'])
        for currency in daily_curr_contrib.columns:
            # Skip USD for DM/EM classification
            if currency == 'USD':
                continue
            classification = self.currency_map.get(currency, 'Other').upper() # Default to 'Other' if not in map
            if classification == 'DM':
                daily_dm_em_contrib['DM'] += daily_curr_contrib[currency]
            elif classification == 'EM':
                daily_dm_em_contrib['EM'] += daily_curr_contrib[currency]
            else: # Add to 'Other' if classification is not DM or EM
                daily_dm_em_contrib['Other'] += daily_curr_contrib[currency]

        # Drop 'Other' column if it's all zero
        if (daily_dm_em_contrib['Other'] == 0).all():
             daily_dm_em_contrib = daily_dm_em_contrib.drop(columns=['Other'])


        return daily_sub_contrib, daily_curr_contrib, daily_dm_em_contrib


    def analyze_attribution(self,
        periods: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = None
        ) -> Dict[str, pd.DataFrame]:
        """
        Calculates and summarizes performance attribution over specified periods.

        Args:
            periods (dict, optional): Dictionary defining periods.
                Keys are period names (e.g., 'Last 3Y').
                Values define the period:
                    - None: Full sample period.
                    - str (e.g., '1Y', '3Y', '5Y', '3M'): Relative period from the end date.
                    - tuple (start_date_str, end_date_str): Specific date range.
                      Dates should be in 'YYYY-MM-DD' format or None.
                If None, defaults to {'Full Sample': None, 'Last 1Y': '1Y',
                                     'Last 3Y': '3Y', 'Last 5Y': '5Y'}.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing attribution summary DataFrames:
                - 'Substrategy': Rows=Substrategies, Columns=Periods, Values=Summed Contribution.
                - 'Currency': Rows=Currencies, Columns=Periods, Values=Summed Contribution.
                - 'DM_EM': Rows=['DM', 'EM', 'Other'], Columns=Periods, Values=Summed Contribution.
        """
        if periods is None:
            periods = {
                'Full Sample': None,
                'Last 1Y': '1Y',
                'Last 3Y': '3Y',
                'Last 5Y': '5Y'
            }

        # Get pre-calculated daily contributions
        daily_sub_contrib = self._daily_sub_contrib
        daily_curr_contrib = self._daily_curr_contrib
        daily_dm_em_contrib = self._daily_dm_em_contrib

        # Initialize result dictionaries
        sub_results = {}
        curr_results = {}
        dm_em_results = {}

        base_end_date = daily_sub_contrib.index[-1] # Use index from calculated contributions

        for name, period_def in periods.items():
            start_dt = None
            end_dt = base_end_date

            # --- Determine Period Start/End Dates ---
            try:
                if period_def is None: # Full Sample
                    start_dt = daily_sub_contrib.index[0]
                elif isinstance(period_def, str): # Relative 'XY' or 'XM'
                    num = int(period_def[:-1])
                    unit = period_def[-1].upper()
                    if unit == 'Y': offset = pd.DateOffset(years=num)
                    elif unit == 'M': offset = pd.DateOffset(months=num)
                    else: raise ValueError(f"Invalid relative period unit: {unit}. Use 'Y' or 'M'.")

                    start_dt_target = base_end_date - offset
                    available_dates = daily_sub_contrib.index[daily_sub_contrib.index >= start_dt_target]
                    start_dt = available_dates[0] if not available_dates.empty else None

                elif isinstance(period_def, tuple): # Date range (start, end)
                    start_str, end_str = period_def
                    if start_str:
                        start_dt_target = pd.to_datetime(start_str)
                        available_dates = daily_sub_contrib.index[daily_sub_contrib.index >= start_dt_target]
                        start_dt = available_dates[0] if not available_dates.empty else None
                    else: start_dt = daily_sub_contrib.index[0]

                    if end_str:
                        end_dt_target = pd.to_datetime(end_str)
                        available_dates = daily_sub_contrib.index[daily_sub_contrib.index <= end_dt_target]
                        end_dt = available_dates[-1] if not available_dates.empty else None
                    # else end_dt remains base_end_date
                else:
                    raise ValueError(f"Invalid period definition type: {type(period_def)}")

                # --- Slice and Sum Contributions ---
                if start_dt is None or end_dt is None or start_dt > end_dt:
                     print(f"Warning: Period '{name}' results in invalid date range. Skipping.")
                     continue

                # Slice and sum each contribution type
                sub_results[name] = daily_sub_contrib.loc[start_dt:end_dt].sum()
                curr_results[name] = daily_curr_contrib.loc[start_dt:end_dt].sum()
                dm_em_results[name] = daily_dm_em_contrib.loc[start_dt:end_dt].sum()

            except Exception as e:
                print(f"Error processing period '{name}': {e}")
                # Store NaNs for this period if calculation failed
                sub_results[name] = pd.Series(np.nan, index=daily_sub_contrib.columns)
                curr_results[name] = pd.Series(np.nan, index=daily_curr_contrib.columns)
                dm_em_results[name] = pd.Series(np.nan, index=daily_dm_em_contrib.columns)


        # --- Format Results ---
        final_results = {
            'Substrategy': pd.DataFrame(sub_results).fillna(0.0),
            'Currency': pd.DataFrame(curr_results).fillna(0.0),
            'DM_EM': pd.DataFrame(dm_em_results).fillna(0.0)
        }
        # Optional: Add Total row/column?
        for key, df in final_results.items():
             if not df.empty:
                  df['Total'] = df.sum(axis=1) # Sum contribution across periods for each contributor
                  # Add total contribution row for each period
                  # Ensure 'Total' row calculation handles potential NaNs if needed, though fillna(0) helps
                  df.loc['Total', :] = df.sum(axis=0)


        return final_results

# Note: No __main__ block here, this is intended as an importable module.
