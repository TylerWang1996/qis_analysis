import pandas as pd
import numpy as np
# Use explicit End offsets for clarity and future compatibility
from pandas.tseries.offsets import BDay, MonthEnd, QuarterEnd, YearEnd, DateOffset
from typing import Optional

class EqualVolatilityPortfolio:
    """
    Constructs a portfolio with weights inversely proportional to volatility.

    This class takes a DataFrame of total return indices, calculates daily
    weights based on an equal volatility (inverse volatility) strategy,
    and handles periodic rebalancing and daily weight drift.

    Attributes:
        total_return_index (pd.DataFrame): DataFrame of total return indices.
                                           Index must be DatetimeIndex.
        lookback_years (int): Number of years for volatility calculation.
        skip_recent_month (bool): Whether to skip the most recent month in
                                  volatility calculation.
        rebalance_freq (str): Rebalancing frequency ('M', 'Q', 'A', 'W').
        returns (pd.DataFrame): Daily returns calculated from total_return_index.
        trading_days_per_year (int): Assumed number of trading days per year
                                     for annualization.
    """

    def __init__(self,
                 total_return_index: pd.DataFrame,
                 lookback_years: int = 2,
                 skip_recent_month: bool = False,
                 rebalance_freq: str = 'M',
                 trading_days_per_year: int = 252):
        """
        Initializes the EqualVolatilityPortfolio object.

        Args:
            total_return_index (pd.DataFrame): DataFrame with dates as index,
                                               tickers as columns, and total
                                               return index values. Index must be
                                               a DatetimeIndex sorted ascending.
            lookback_years (int): Lookback period in years for volatility calc.
                                  Defaults to 2.
            skip_recent_month (bool): If True, excludes the most recent month
                                      from the volatility calculation window.
                                      Defaults to False.
            rebalance_freq (str): Rebalancing frequency. Supported values:
                                  'M' (Month End), 'Q' (Quarter End),
                                  'A' (Year End), 'W' (Week End - Friday).
                                  Defaults to 'M'.
            trading_days_per_year (int): Number of trading days used for
                                         annualizing volatility. Defaults to 252.

        Raises:
            TypeError: If total_return_index index is not a DatetimeIndex.
            ValueError: If inputs are invalid (e.g., negative lookback, empty data,
                        unsupported rebalance frequency, NaNs in input).
        """
        if not isinstance(total_return_index.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame index must be a DatetimeIndex.")
        if not total_return_index.index.is_monotonic_increasing:
             raise ValueError("Input DataFrame index must be sorted ascending.")

        self.total_return_index = total_return_index.copy()
        self.lookback_years = lookback_years
        self.skip_recent_month = skip_recent_month
        self.rebalance_freq = rebalance_freq.upper() # Ensure uppercase
        self.trading_days_per_year = trading_days_per_year
        self._validate_inputs() # Perform initial validation

        # Calculate returns *after* validation
        self.returns = self._calculate_daily_returns()

        # Final check on returns data
        if self.returns.empty or self.returns.iloc[1:].empty: # Check if returns calculation yielded results
             raise ValueError("Calculated returns DataFrame is empty or has only NaNs. "
                              "Check input total_return_index (needs at least 2 rows).")


    def _validate_inputs(self):
        """Performs validation checks on initialization parameters."""
        if self.lookback_years <= 0:
            raise ValueError("Lookback years must be positive.")
        if self.rebalance_freq not in ['M', 'Q', 'A', 'W']:
             raise ValueError("Invalid rebalance frequency. Use 'M', 'Q', 'A', or 'W'.")
        if self.total_return_index.empty:
            raise ValueError("Input total_return_index DataFrame cannot be empty.")
        if self.total_return_index.shape[0] < 2:
             raise ValueError("Input total_return_index DataFrame must have at least 2 rows to calculate returns.")
        if self.total_return_index.isnull().values.any():
            # Consider adding an option to handle NaNs (e.g., ffill) if needed.
            raise ValueError("Input total_return_index contains NaNs. Please clean data first.")
        if self.trading_days_per_year <= 0:
             raise ValueError("Trading days per year must be positive.")


    def _calculate_daily_returns(self) -> pd.DataFrame:
        """
        Calculates daily percentage returns from the total return index.

        Returns:
            pd.DataFrame: DataFrame of daily returns. The first row will contain NaNs.
        """
        # pct_change calculates (current - previous) / previous
        return self.total_return_index.pct_change() # Keep the first NaN row for alignment


    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """
        Determines the rebalance dates (last trading day of the period).

        These are the dates *at the end of which* the target weights are calculated,
        to be applied on the *next* trading day.

        Returns:
            pd.DatetimeIndex: Sorted list of rebalance dates within the returns index.

        Raises:
            ValueError: If not enough data exists to determine any rebalance dates
                        after accounting for the required lookback period.
        """
        # Use returns index excluding the first NaN row for date calculations
        valid_returns_index = self.returns.index[1:]
        if valid_returns_index.empty:
             raise ValueError("Not enough data to calculate returns or determine rebalance dates.")

        min_data_date = valid_returns_index[0]

        # Calculate the total lookback offset in months
        total_months_offset = self.lookback_years * 12
        if self.skip_recent_month:
            total_months_offset += 1
        # Use a single month-based offset
        required_offset = pd.DateOffset(months=total_months_offset)

        # The first date we can *calculate* volatility for (using data up to this date)
        first_possible_vol_calc_date = min_data_date + required_offset

        # We need to find the first trading day *on or after* this date in our index
        valid_start_mask = valid_returns_index >= first_possible_vol_calc_date
        if not valid_start_mask.any():
             raise ValueError(f"Not enough data history for the lookback period. "
                              f"Lookback requires data up to {first_possible_vol_calc_date.date()}, "
                              f"but latest return date is {valid_returns_index[-1].date()}.")
        first_valid_calc_date = valid_returns_index[valid_start_mask][0]


        # Generate potential end-of-period dates based on frequency
        # Use the index of the *returns* DataFrame for date range
        date_index = valid_returns_index # Use index with actual returns
        freq_map = {'M': 'ME', 'Q': 'QE', 'A': 'YE', 'W': 'W-FRI'}
        pandas_freq = freq_map.get(self.rebalance_freq, 'ME') # Default to MonthEnd

        potential_dates = pd.date_range(start=date_index[0], end=date_index[-1], freq=pandas_freq)

        # Map potential EOP dates to the *last actual trading day* on or before that date
        rebalance_dates = []
        available_dates_index = date_index # Alias for clarity
        for date in potential_dates:
            # Find the index of the latest trading day <= potential EOP date
            idx = available_dates_index.searchsorted(date, side='right')
            if idx > 0:
                actual_date = available_dates_index[idx - 1]
                # Ensure we don't add duplicates if multiple potential dates map to the same trading day
                if not rebalance_dates or actual_date != rebalance_dates[-1]:
                    rebalance_dates.append(actual_date)

        # Convert to DatetimeIndex (already sorted)
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)

        # Filter: Only keep rebalance dates on or after the first date
        # for which we can actually calculate the volatility.
        rebalance_dates = rebalance_dates[rebalance_dates >= first_valid_calc_date]

        if rebalance_dates.empty:
            raise ValueError(f"Not enough data for any rebalances after {first_valid_calc_date.date()} "
                             f"with lookback {self.lookback_years} years and freq '{self.rebalance_freq}'.")

        return rebalance_dates


    def _calculate_volatility(self, calculation_end_date: pd.Timestamp) -> pd.Series:
        """
        Calculates annualized volatility for each asset up to a given date (t-1).

        Args:
            calculation_end_date (pd.Timestamp): The last date of returns data
                                                 to include for volatility calculation.
                                                 This should be a valid date in self.returns.index.

        Returns:
            pd.Series: Annualized volatility for each asset (index=asset tickers).
                       Returns NaN for an asset if its volatility cannot be calculated
                       (e.g., insufficient data points in the window).
        """
        # Use returns index excluding the first NaN row
        valid_returns_index = self.returns.index[1:]

        # Determine the actual end date for the volatility window slice
        vol_window_end = calculation_end_date
        if self.skip_recent_month:
            # Target end date is one month before the calculation_end_date
            target_end = calculation_end_date - pd.DateOffset(months=1)
            # Find the closest trading day <= target_end within the valid returns index
            valid_end_mask = valid_returns_index <= target_end
            if not valid_end_mask.any():
                 # Not enough history to skip a month
                 return pd.Series(np.nan, index=self.returns.columns, name=calculation_end_date)
            vol_window_end = valid_returns_index[valid_end_mask][-1]

        # Determine the start date of the volatility window slice using a simple month offset
        vol_window_start_offset = pd.DateOffset(months=self.lookback_years * 12)
        target_start = vol_window_end - vol_window_start_offset

        # Find the closest trading day >= target_start within the valid returns index
        valid_start_mask = valid_returns_index >= target_start
        if not valid_start_mask.any():
            # Not enough history for the lookback
            return pd.Series(np.nan, index=self.returns.columns, name=calculation_end_date)
        vol_window_start = valid_returns_index[valid_start_mask][0]

        # Select the returns data for the volatility window (use .loc for label indexing)
        # Ensure we select from the original returns DF which includes NaNs if needed,
        # but the window start/end dates are based on valid return dates.
        returns_window = self.returns.loc[vol_window_start:vol_window_end]

        # --- Data Adequacy Check ---
        # Require a minimum number of data points for a reliable std dev calculation
        expected_days = self.trading_days_per_year * self.lookback_years
        # Use a fraction of expected days or an absolute minimum
        min_required_days = max(30, int(expected_days * 0.6))

        # Calculate daily standard deviation only if enough data points exist
        # Count non-NaN values for each column
        valid_counts = returns_window.count()
        daily_std = returns_window.std(ddof=1) # Use sample standard deviation (N-1 denominator)

        # Set std dev to NaN if fewer than min_required_days are available
        sufficient_data_mask = valid_counts >= min_required_days
        daily_std[~sufficient_data_mask] = np.nan

        # Annualize volatility
        annualized_vol = daily_std * np.sqrt(self.trading_days_per_year)

        return annualized_vol.rename(calculation_end_date) # Rename series for clarity


    def _calculate_target_weights(self, calculation_end_date: pd.Timestamp) -> pd.Series:
        """
        Calculates target weights based on inverse volatility as of calculation_end_date.

        Args:
            calculation_end_date (pd.Timestamp): The date based on which to
                                                 calculate volatility (t-1).

        Returns:
            pd.Series: Target weights for each asset (index=asset tickers).
                       Assets with NaN or zero volatility get zero weight.
                       Returns a series of zeros if no valid volatility is found.
        """
        volatility = self._calculate_volatility(calculation_end_date)

        # Replace zero volatility with NaN to avoid division by zero and treat appropriately
        volatility = volatility.replace(0, np.nan)

        # Calculate inverse volatility - assets with NaN vol get NaN inv_vol
        inv_vol = 1 / volatility

        # Handle cases where all volatilities are NaN (e.g., start of series)
        if inv_vol.isnull().all():
            # Return zero weights if no valid volatility could be calculated
            print(f"Warning: Could not calculate volatility for any asset on {calculation_end_date.date()}. Returning zero weights.")
            return pd.Series(0.0, index=self.returns.columns, name=calculation_end_date)

        # Calculate weights only for assets with valid (non-NaN) inverse volatility
        valid_inv_vol = inv_vol.dropna()
        total_inv_vol = valid_inv_vol.sum()

        # Calculate weights for valid assets
        if total_inv_vol > 1e-10: # Use tolerance for floating point comparison
            target_weights_valid = valid_inv_vol / total_inv_vol
        else:
             # If sum is effectively zero (e.g., all valid vols were huge, inv_vols near zero)
             # or negative (shouldn't happen), assign zero weights.
             target_weights_valid = pd.Series(0.0, index=valid_inv_vol.index)


        # Create the full weight series, filling NaNs with 0 weight
        target_weights = pd.Series(0.0, index=self.returns.columns, name=calculation_end_date)
        target_weights.update(target_weights_valid)

        return target_weights


    def construct_portfolio(self) -> pd.DataFrame:
        """
        Constructs the portfolio by calculating daily weights according to the strategy.

        Handles initial weight calculation, rebalancing, and daily drift.

        Returns:
            pd.DataFrame: DataFrame with daily weights for each asset.
                          Index is DatetimeIndex, columns are asset tickers.
                          The DataFrame starts on the first date for which weights
                          can be determined based on the lookback period.

        Raises:
            ValueError: If initial weights cannot be calculated due to insufficient data.
        """
        rebalance_dates = self._get_rebalance_dates()
        # The first date we *calculate* weights based on is the first rebalance date.
        first_rebalance_date = rebalance_dates[0]

        # Find the first trading day *after* the first calculation date in the full returns index
        # This is the first day the weights will actually be applied.
        full_returns_index = self.returns.index
        first_weight_application_date_idx = full_returns_index.searchsorted(first_rebalance_date, side='right')

        if first_weight_application_date_idx >= len(full_returns_index):
             raise ValueError("Cannot determine first weight application date. "
                              "Not enough data after the first rebalance date.")
        first_weight_application_date = full_returns_index[first_weight_application_date_idx]


        # Initialize weights DataFrame starting from the first application date, using the full index
        output_dates = full_returns_index[full_returns_index >= first_weight_application_date]
        weights_df = pd.DataFrame(index=output_dates, columns=self.returns.columns, dtype=float)

        # --- Initial Weight Calculation ---
        # Weights for the first day are based on volatility calculated up to the first rebalance date.
        initial_calc_end_date = first_rebalance_date
        current_weights = self._calculate_target_weights(initial_calc_end_date)

        # Check if initial calculation resulted in all zeros (due to warnings in target weights)
        if current_weights.sum() < 1e-10:
             # This indicates an issue with the very first volatility calculation
             raise ValueError(f"Could not calculate valid non-zero initial weights for date "
                              f"{first_weight_application_date.date()} based on data up to "
                              f"{initial_calc_end_date.date()}. Check data adequacy/volatility.")

        weights_df.loc[first_weight_application_date] = current_weights

        # Convert rebalance_dates to a set for efficient lookup
        rebalance_dates_set = set(rebalance_dates)

        # --- Loop Through Remaining Days ---
        # Iterate from the second day of the output period
        for t in range(1, len(output_dates)):
            current_date = output_dates[t]
            prev_date = output_dates[t-1] # Date for which weights are already set

            # Get weights from the previous day
            prev_weights = weights_df.loc[prev_date].copy() # Use copy to avoid modifying df
            # Get returns for prev_date (used for drift calculation)
            # Handle potential NaN returns on prev_date (e.g., if it was the first day of original data)
            # Use fillna(0.0) as 0% return won't affect drift if weights exist
            returns_prev_day = self.returns.loc[prev_date].fillna(0.0)

            # Check if prev_date was a rebalance date. If so, calculate new target weights for current_date.
            if prev_date in rebalance_dates_set:
                # Recalculate target weights based on data up to prev_date
                target_weights = self._calculate_target_weights(prev_date)
                # Note: _calculate_target_weights handles NaNs by returning 0 weights
                current_weights = target_weights
                weights_df.loc[current_date] = current_weights
            else:
                # Drift the weights based on previous day's returns
                portfolio_return_prev_day = (prev_weights * returns_prev_day).sum()
                denominator = 1.0 + portfolio_return_prev_day

                if np.isclose(denominator, 0):
                     print(f"Warning: Portfolio return was -100% on {prev_date.date()}. Holding previous weights for {current_date.date()}.")
                     drifted_weights = prev_weights
                elif np.isnan(denominator) or denominator < 1e-10: # Also handle cases where denominator is extremely small
                     print(f"Warning: Drift denominator near zero or NaN on {current_date.date()}. Holding previous weights.")
                     drifted_weights = prev_weights
                else:
                    # w_i(t) = w_i(t-1) * (1 + R_i(t-1)) / (1 + Port_Return(t-1))
                    drifted_weights = prev_weights * (1 + returns_prev_day) / denominator

                # Store the drifted weights
                weights_df.loc[current_date] = drifted_weights
                # Update current_weights (though not strictly needed as we read from df)
                current_weights = drifted_weights


        # --- Final Processing ---
        # Fill any potential NaNs that might have slipped through (e.g., from drift issues)
        weights_df = weights_df.fillna(0.0)

        # Normalize row-wise to ensure weights sum to 1, handling rows that sum to 0
        row_sums = weights_df.sum(axis=1)
        # Avoid division by zero for rows where all weights became zero
        safe_row_sums = row_sums.replace(0, 1.0) # Replace 0 with 1 to avoid NaN after division
        weights_df = weights_df.div(safe_row_sums, axis=0)
        # Ensure rows that originally summed to zero remain zero
        weights_df[row_sums == 0] = 0.0

        return weights_df
