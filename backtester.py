import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from typing import Optional, Dict, Union, Tuple

class Backtester:
    """
    Performs backtesting analysis on a strategy given weights and asset returns.

    Calculates strategy returns and various performance metrics over specified periods,
    using both daily and monthly return frequencies.
    """
    def __init__(self,
                 strategy_weights: pd.DataFrame,
                 substrategy_total_return_index: pd.DataFrame,
                 trading_days_per_year: int = 252,
                 months_per_year: int = 12): # Added months_per_year
        """
        Initializes the Backtester.

        Args:
            strategy_weights (pd.DataFrame): DataFrame of daily EOD strategy weights.
                                             Index=DateTimeIndex, Columns=Asset tickers.
            substrategy_total_return_index (pd.DataFrame): DataFrame of total return indices
                                                          for the substrategies.
                                                          Index=DateTimeIndex, Columns=Asset tickers.
            trading_days_per_year (int): Number of trading days assumed per year. Defaults to 252.
            months_per_year (int): Number of months assumed per year. Defaults to 12.

        Raises:
            ValueError: If inputs are invalid (e.g., mismatched columns, no date overlap).
        """
        if not isinstance(strategy_weights.index, pd.DatetimeIndex) or \
           not isinstance(substrategy_total_return_index.index, pd.DatetimeIndex):
            raise TypeError("Input indices must be DatetimeIndex.")

        if not strategy_weights.index.is_monotonic_increasing or \
           not substrategy_total_return_index.index.is_monotonic_increasing:
             raise ValueError("Input indices must be sorted ascending.")

        self.trading_days_per_year = trading_days_per_year
        self.months_per_year = months_per_year # Store months per year
        self._strategy_weights = strategy_weights.copy()
        self._substrat_tri = substrategy_total_return_index.copy()

        # Ensure columns match (allow weights to be a subset of TRI columns)
        if not set(self._strategy_weights.columns).issubset(set(self._substrat_tri.columns)):
             raise ValueError("Strategy weight columns must be a subset of substrategy TRI columns.")

        # Calculate substrategy daily returns
        self._substrat_returns = self._substrat_tri.pct_change()

        self.daily_contributions = self._calculate_daily_contributions()

        # Align data and calculate strategy daily returns
        self.strategy_daily_returns = self._calculate_strategy_returns()

        if self.strategy_daily_returns.empty:
            raise ValueError("Strategy daily returns calculation resulted in an empty series. "
                             "Check date alignment and overlap between weights and returns.")

        # Calculate strategy monthly returns
        self.strategy_monthly_returns = self._calculate_strategy_monthly_returns()
        # Ensure monthly returns don't have NaNs right after calculation
        if self.strategy_monthly_returns.isnull().any():
             print("Warning: NaNs detected in initial monthly return calculation. Check resampling.")
             # It's possible the first/last month has issues if daily data is sparse there
             # self.strategy_monthly_returns = self.strategy_monthly_returns.dropna()

    def _calculate_daily_contributions(self) -> pd.DataFrame:
        """
        Calculates the daily arithmetic contribution of each substrategy.
        Contribution = Weight(t-1) * Return(t)
        """
        weights_aligned, returns_aligned = self._strategy_weights.align(
            self._substrat_returns, join='inner', axis=1
        )
        shifted_weights = weights_aligned.shift(1)
        
        # Align again after shifting to ensure dates match perfectly
        shifted_weights, returns_aligned = shifted_weights.align(returns_aligned, join='inner', axis=0)
        
        # Element-wise multiplication
        daily_contrib = shifted_weights * returns_aligned
        
        return daily_contrib.dropna(how='all') # Drop rows that are all NaN (e.g., first day)

    def _calculate_strategy_returns(self) -> pd.Series:
        """
        Calculates the daily returns of the strategy based on t-1 weights.
        """
        # Align weights and returns (inner join on columns)
        weights_aligned, returns_aligned = self._strategy_weights.align(
            self._substrat_returns, join='inner', axis=1
        )

        # Shift weights forward by one day to use t-1 weights for t returns
        shifted_weights = weights_aligned.shift(1)

        # Combine shifted weights and returns, keeping only dates where both exist
        # Use 'inner' join to handle potential NaNs from shifting or start/end differences
        combined = pd.concat([shifted_weights, returns_aligned], axis=1, keys=['weights', 'returns'], join='inner')

        # Separate aligned weights and returns after ensuring date alignment
        weights_final = combined['weights']
        returns_final = combined['returns']

        # Calculate daily portfolio return: sum(weight[t-1] * return[t])
        strategy_returns = (weights_final * returns_final).sum(axis=1)

        # Drop any potential leading NaNs (already handled by inner join, but safe)
        strategy_returns = strategy_returns.dropna()

        return strategy_returns.rename("strategy_returns")

    def _calculate_strategy_monthly_returns(self) -> pd.Series:
        """Calculates monthly returns from daily returns."""
        if self.strategy_daily_returns.empty:
            return pd.Series(dtype=float)
        # Resample daily returns to monthly using Month End frequency ('ME')
        # Compound daily returns within each month
        monthly_returns = (1 + self.strategy_daily_returns).resample('ME').prod() - 1
        # *** FIX: Add dropna() to handle potential NaNs from resampling edge cases ***
        monthly_returns = monthly_returns.dropna()
        return monthly_returns.rename("strategy_monthly_returns")


    def get_strategy_returns(self, frequency: str = 'daily') -> pd.Series:
        """
        Returns the calculated strategy returns.

        Args:
            frequency (str): 'daily' or 'monthly'. Defaults to 'daily'.

        Returns:
            pd.Series: Strategy returns for the specified frequency.
        """
        if frequency == 'monthly':
            return self.strategy_monthly_returns.copy()
        elif frequency == 'daily':
            return self.strategy_daily_returns.copy()
        else:
            raise ValueError("Frequency must be 'daily' or 'monthly'")

    def get_strategy_index(self, initial_value: float = 100.0, frequency: str = 'daily') -> pd.Series:
        """
        Calculates the strategy's total return index.

        Args:
            initial_value (float): The starting value for the index. Defaults to 100.0.
            frequency (str): 'daily' or 'monthly' to determine index frequency. Defaults to 'daily'.

        Returns:
            pd.Series: Strategy total return index.
        """
        returns = self.get_strategy_returns(frequency=frequency)
        if returns.empty:
            return pd.Series(dtype=float) # Return empty series if no returns

        cumulative_returns = (1 + returns).cumprod()
        strategy_index = initial_value * cumulative_returns

        # Insert the initial value at the beginning for a complete index series
        first_return_date = returns.index[0]
        original_index = self._substrat_tri.index # Use original TRI for better date finding
        loc = original_index.searchsorted(first_return_date)

        start_date_for_index = None
        if loc > 0:
            # Try to find the business day before the first return date in the original index
            potential_start = original_index[loc-1]
            # Check if it's truly before (handles cases where first return date is first in TRI)
            if potential_start < first_return_date:
                 start_date_for_index = potential_start

        if start_date_for_index is None:
            # Fallback: Use the day before the first return date, adjusting frequency
            offset = pd.offsets.BDay(1) if frequency == 'daily' else pd.offsets.MonthBegin(1)
            try:
                 start_date_for_index = first_return_date - offset
            except: # If offset fails, cannot reliably prepend
                 start_date_for_index = None

        if start_date_for_index and start_date_for_index not in strategy_index.index:
             initial_value_series = pd.Series([initial_value], index=[start_date_for_index])
             strategy_index = pd.concat([initial_value_series, strategy_index])

        return strategy_index.rename(f"strategy_index_{frequency}")


    def _calculate_drawdown_details(self, cumulative_index: pd.Series) -> Dict[str, Union[float, Optional[pd.Timestamp]]]:
        """Calculates max drawdown and related dates from a cumulative index series."""
        if cumulative_index.empty or cumulative_index.isnull().all() or len(cumulative_index) < 2:
            return {'max_drawdown': np.nan, 'peak_date': None, 'trough_date': None, 'recovery_date': None}

        running_max = cumulative_index.cummax()
        drawdown = (cumulative_index / running_max) - 1.0

        # Handle cases where index is constant or always increasing
        if not (drawdown < -1e-9).any(): # Use tolerance
             return {'max_drawdown': 0.0, 'peak_date': None, 'trough_date': None, 'recovery_date': None}

        max_drawdown = drawdown.min()
        trough_date = drawdown.idxmin()

        # Find peak date *before or on* the trough date
        peak_date = running_max.loc[:trough_date].idxmax()

        # Find recovery date
        recovery_date = None
        post_trough_index = cumulative_index.loc[trough_date:]
        peak_value = cumulative_index.loc[peak_date]
        try:
            # Find first date *after* trough where index value >= peak value
            recovery_candidates = post_trough_index[post_trough_index >= peak_value * (1 - 1e-9)] # Tolerance
            if not recovery_candidates.empty:
                 # Ensure recovery date is strictly after trough date if possible
                 possible_recovery_dates = recovery_candidates.index[recovery_candidates.index > trough_date]
                 if not possible_recovery_dates.empty:
                      recovery_date = possible_recovery_dates[0]
                 elif recovery_candidates.index[0] >= trough_date : # Allow recovery on trough date only if it's the only option >= peak
                      recovery_date = recovery_candidates.index[0]

        except IndexError:
            recovery_date = None # No recovery found in the period

        return {
            'max_drawdown': max_drawdown,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date
        }

    def _calculate_metrics_for_period(self, daily_returns_period: pd.Series) -> Dict[str, float]:
        """Helper to calculate all metrics using DAILY returns for a given period."""
        metrics = {}
        n_days = len(daily_returns_period)
        metric_names = ['Ann Return', 'Ann Volatility', 'Information Ratio',
                        'Skewness', 'Kurtosis', 'Sortino Ratio', 'Max Drawdown',
                        'Calmar Ratio', '% Positive Months', 'DD Recovery Months']

        min_days_threshold = 21 # Approx 1 month
        if n_days < min_days_threshold:
             return {m: np.nan for m in metric_names}

        clean_returns = daily_returns_period.dropna()
        n_days_clean = len(clean_returns)
        if n_days_clean < min_days_threshold:
             return {m: np.nan for m in metric_names}

        # --- Return & Vol ---
        cumulative_return_period = (1 + clean_returns).prod() - 1
        years_in_period = n_days_clean / self.trading_days_per_year
        metrics['Ann Return'] = (1 + cumulative_return_period) ** (1 / years_in_period) - 1 if years_in_period > 0 else 0.0
        daily_std_dev = clean_returns.std(ddof=1)
        metrics['Ann Volatility'] = daily_std_dev * np.sqrt(self.trading_days_per_year)

        # --- Ratios ---
        vol = metrics['Ann Volatility']
        ret = metrics['Ann Return']
        metrics['Information Ratio'] = ret / vol if vol != 0 else (np.inf if ret > 1e-9 else (0.0 if abs(ret) < 1e-9 else -np.inf))

        downside_returns = clean_returns[clean_returns < 0]
        downside_dev = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 0.0
        annualized_downside_dev = downside_dev * np.sqrt(self.trading_days_per_year)
        metrics['Sortino Ratio'] = ret / annualized_downside_dev if annualized_downside_dev != 0 else (np.inf if ret > 1e-9 else (0.0 if abs(ret) < 1e-9 else -np.inf))

        # --- Distribution ---
        metrics['Skewness'] = clean_returns.skew()
        metrics['Kurtosis'] = clean_returns.kurt() # Fisher's (excess) kurtosis

        # --- Drawdown ---
        cumulative_index_period = (1 + clean_returns).cumprod() * 100
        dd_details = self._calculate_drawdown_details(cumulative_index_period)
        metrics['Max Drawdown'] = dd_details['max_drawdown']
        max_dd = metrics['Max Drawdown']
        metrics['Calmar Ratio'] = ret / abs(max_dd) if abs(max_dd) > 1e-9 else (np.inf if ret > 1e-9 else 0.0)

        if dd_details['recovery_date'] is not None and dd_details['trough_date'] is not None:
             time_delta = dd_details['recovery_date'] - dd_details['trough_date']
             metrics['DD Recovery Months'] = time_delta.days / 30.4375
        else:
             metrics['DD Recovery Months'] = np.nan

        # --- Monthly Stats ---
        try:
            monthly_returns = (1 + clean_returns).resample('ME').prod() - 1
            if monthly_returns.empty or len(monthly_returns) < 1:
                 metrics['% Positive Months'] = np.nan
            else:
                 metrics['% Positive Months'] = (monthly_returns > 0).mean() * 100.0
        except ValueError:
            metrics['% Positive Months'] = np.nan

        return metrics

    def _calculate_monthly_metrics_for_period(self, monthly_returns_period: pd.Series) -> Dict[str, float]:
        """Helper to calculate all metrics using MONTHLY returns for a given period."""
        metrics = {}
        n_months = len(monthly_returns_period)
        metric_names = ['Ann Return', 'Ann Volatility', 'Information Ratio',
                        'Skewness', 'Kurtosis', 'Sortino Ratio', 'Max Drawdown',
                        'Calmar Ratio', '% Positive Months', 'DD Recovery Months']

        min_months_threshold = 12 # Require at least 1 year for meaningful annualized monthly stats
        if n_months < min_months_threshold:
            return {m: np.nan for m in metric_names}

        clean_returns = monthly_returns_period.dropna()
        n_months_clean = len(clean_returns)
        if n_months_clean < min_months_threshold:
            return {m: np.nan for m in metric_names}

        # --- Return & Vol ---
        cumulative_return_period = (1 + clean_returns).prod() - 1
        years_in_period = n_months_clean / self.months_per_year
        metrics['Ann Return'] = (1 + cumulative_return_period) ** (1 / years_in_period) - 1 if years_in_period > 0 else 0.0
        monthly_std_dev = clean_returns.std(ddof=1)
        metrics['Ann Volatility'] = monthly_std_dev * np.sqrt(self.months_per_year)

        # --- Ratios ---
        vol = metrics['Ann Volatility']
        ret = metrics['Ann Return']
        metrics['Information Ratio'] = ret / vol if vol != 0 else (np.inf if ret > 1e-9 else (0.0 if abs(ret) < 1e-9 else -np.inf))

        downside_returns = clean_returns[clean_returns < 0]
        downside_dev = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 0.0
        annualized_downside_dev = downside_dev * np.sqrt(self.months_per_year) # Use months/year
        metrics['Sortino Ratio'] = ret / annualized_downside_dev if annualized_downside_dev != 0 else (np.inf if ret > 1e-9 else (0.0 if abs(ret) < 1e-9 else -np.inf))

        # --- Distribution ---
        metrics['Skewness'] = clean_returns.skew()
        metrics['Kurtosis'] = clean_returns.kurt() # Fisher's (excess) kurtosis

        # --- Drawdown (from monthly returns) ---
        cumulative_index_period = (1 + clean_returns).cumprod() * 100
        dd_details = self._calculate_drawdown_details(cumulative_index_period)
        metrics['Max Drawdown'] = dd_details['max_drawdown']
        max_dd = metrics['Max Drawdown']
        metrics['Calmar Ratio'] = ret / abs(max_dd) if abs(max_dd) > 1e-9 else (np.inf if ret > 1e-9 else 0.0)

        # DD Recovery Time (in months directly)
        if dd_details['recovery_date'] is not None and dd_details['trough_date'] is not None:
             # Calculate difference in months between recovery and trough dates
             # Use the number of periods in the monthly series index for simplicity
             try:
                  recovery_idx = clean_returns.index.get_loc(dd_details['recovery_date'])
                  trough_idx = clean_returns.index.get_loc(dd_details['trough_date'])
                  metrics['DD Recovery Months'] = recovery_idx - trough_idx # Already in months
             except KeyError: # Handle case where date might not be exactly in index after cleaning
                  metrics['DD Recovery Months'] = np.nan
        else:
             metrics['DD Recovery Months'] = np.nan

        # --- Monthly Stats ---
        # % Positive Months is simpler here
        metrics['% Positive Months'] = (clean_returns > 0).mean() * 100.0

        return metrics


    def _get_period_returns(self, frequency: str, period_def: Optional[Union[str, Tuple[Optional[str], Optional[str]]]]) -> pd.Series:
        """Slices the returns series based on the period definition."""
        base_returns = self.get_strategy_returns(frequency=frequency)
        if base_returns.empty:
            return pd.Series(dtype=float)

        end_date = base_returns.index[-1]
        start_dt = None
        end_dt = end_date # Default end date

        if period_def is None: # Full Sample
            start_dt = base_returns.index[0]
        elif isinstance(period_def, str): # Relative 'XY' or 'XM'
            num = int(period_def[:-1])
            unit = period_def[-1].upper()
            if unit == 'Y':
                 offset = pd.DateOffset(years=num)
            elif unit == 'M':
                 offset = pd.DateOffset(months=num)
            else:
                 raise ValueError(f"Invalid relative period unit: {unit}. Use 'Y' or 'M'.")

            start_dt_target = end_date - offset
            # Find the first available date >= target start date
            available_dates_after_target = base_returns.index[base_returns.index >= start_dt_target]
            if not available_dates_after_target.empty:
                 start_dt = available_dates_after_target[0]
            else: # Requested period is entirely before data starts
                 start_dt = None # Will result in empty slice

        elif isinstance(period_def, tuple): # Date range (start, end)
            start_str, end_str = period_def
            if start_str:
                start_dt_target = pd.to_datetime(start_str)
                available_dates_after_target = base_returns.index[base_returns.index >= start_dt_target]
                if not available_dates_after_target.empty:
                    start_dt = available_dates_after_target[0]
                else: # Requested start is after all data
                     start_dt = None
            else: # Start from beginning
                start_dt = base_returns.index[0]

            if end_str:
                end_dt_target = pd.to_datetime(end_str)
                available_dates_before_target = base_returns.index[base_returns.index <= end_dt_target]
                if not available_dates_before_target.empty:
                     end_dt = available_dates_before_target[-1]
                else: # Requested end is before all data
                     end_dt = None # Will result in empty slice
            # else end_dt remains the series end_date
        else:
            raise ValueError(f"Invalid period definition type: {type(period_def)}")

        # Slice returns for the period
        if start_dt is None or end_dt is None or start_dt > end_dt:
             period_returns = pd.Series(dtype=float) # Empty series
        else:
             period_returns = base_returns.loc[start_dt:end_dt]

        return period_returns


    def calculate_performance_summary(self,
        periods: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = None,
        frequency: str = 'daily'
        ) -> pd.DataFrame:
        """
        Calculates performance metrics over specified periods for a given frequency.

        Args:
            periods (dict, optional): Dictionary defining periods. See _get_period_returns.
                                      If None, defaults to {'Full Sample': None, 'Last 1Y': '1Y',
                                                           'Last 3Y': '3Y', 'Last 5Y': '5Y'}.
            frequency (str): 'daily' or 'monthly'. Determines which returns and metrics function to use.
                             Defaults to 'daily'.

        Returns:
            pd.DataFrame: DataFrame with metrics as rows and periods as columns.
        """
        if frequency not in ['daily', 'monthly']:
            raise ValueError("Frequency must be 'daily' or 'monthly'")

        if periods is None:
            periods = {
                'Full Sample': None,
                'Last 1Y': '1Y',
                'Last 3Y': '3Y',
                'Last 5Y': '5Y'
            }

        all_metrics = {}
        metric_func = self._calculate_metrics_for_period if frequency == 'daily' else self._calculate_monthly_metrics_for_period

        # print(f"\nCalculating {frequency} metrics for periods:", list(periods.keys())) # Reduce noise

        for name, period_def in periods.items():
            # print(f"  Period: {name}...", end="") # Reduce noise
            try:
                period_returns = self._get_period_returns(frequency, period_def)

                # Calculate metrics if data exists for the period
                if not period_returns.empty:
                    metrics = metric_func(period_returns)
                    all_metrics[name] = metrics
                    # print(" Done.") # Reduce noise
                # else:
                    # print(f" No data. Skipping.") # Reduce noise

            except Exception as e:
                print(f"\nError calculating {frequency} metrics for period '{name}': {e}")
                # traceback.print_exc() # Uncomment for debug
                # Add NaN metrics for this period if calculation failed
                metric_names = ['Ann Return', 'Ann Volatility', 'Information Ratio',
                                'Skewness', 'Kurtosis', 'Sortino Ratio', 'Max Drawdown',
                                'Calmar Ratio', '% Positive Months', 'DD Recovery Months']
                all_metrics[name] = {m: np.nan for m in metric_names}


        # Format results into a DataFrame
        if not all_metrics:
            return pd.DataFrame() # Return empty if no periods were successful

        summary_df = pd.DataFrame(all_metrics)
        # Define a standard order for metrics
        metric_order = [
            'Ann Return', 'Ann Volatility', 'Information Ratio', 'Sortino Ratio',
            'Skewness', 'Kurtosis', 'Max Drawdown', 'Calmar Ratio',
            '% Positive Months', 'DD Recovery Months'
        ]
        # Reindex, adding missing metrics as NaN
        summary_df = summary_df.reindex(metric_order)
        # Ensure columns are in the order they were specified in the input dict
        summary_df = summary_df.reindex(columns=periods.keys())

        return summary_df

    # --- Convenience method for monthly summary ---
    def calculate_monthly_performance_summary(self,
        periods: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = None
        ) -> pd.DataFrame:
        """
        Convenience method to calculate performance metrics using MONTHLY returns.

        Args:
            periods (dict, optional): Dictionary defining periods. See calculate_performance_summary.

        Returns:
            pd.DataFrame: DataFrame with monthly-based metrics as rows and periods as columns.
        """
        return self.calculate_performance_summary(periods=periods, frequency='monthly')

    def calculate_contribution_summary(self,
        periods: Optional[Dict[str, Union[str, Tuple[Optional[str], Optional[str]]]]] = None
        ) -> pd.DataFrame:
        """
        Calculates the total arithmetic contribution of each substrategy over specified periods.

        Args:
            periods (dict, optional): Dictionary defining periods. See calculate_performance_summary.

        Returns:
            pd.DataFrame: DataFrame with substrategies as rows and periods as columns,
                          showing total contribution.
        """
        if self.daily_contributions.empty:
            print("Warning: Daily contributions data is empty. Cannot calculate summary.")
            return pd.DataFrame()

        if periods is None:
            periods = {
                'Full Sample': None, 'Last 1Y': '1Y', 'Last 3Y': '3Y', 'Last 5Y': '5Y'
            }
        
        all_contributions = {}
        base_end_date = self.daily_contributions.index[-1]

        for name, period_def in periods.items():
            start_dt = None
            end_dt = base_end_date

            try:
                if period_def is None: # Full Sample
                    start_dt = self.daily_contributions.index[0]
                elif isinstance(period_def, str): # Relative 'XY' or 'XM'
                    num = int(period_def[:-1])
                    unit = period_def[-1].upper()
                    if unit == 'Y': offset = pd.DateOffset(years=num)
                    elif unit == 'M': offset = pd.DateOffset(months=num)
                    else: raise ValueError(f"Invalid relative period unit: {unit}. Use 'Y' or 'M'.")
                    
                    start_dt_target = base_end_date - offset
                    available_dates = self.daily_contributions.index[self.daily_contributions.index >= start_dt_target]
                    start_dt = available_dates[0] if not available_dates.empty else None

                elif isinstance(period_def, tuple): # Date range
                    start_str, end_str = period_def
                    if start_str:
                        start_dt_target = pd.to_datetime(start_str)
                        available_dates = self.daily_contributions.index[self.daily_contributions.index >= start_dt_target]
                        start_dt = available_dates[0] if not available_dates.empty else None
                    else:
                        start_dt = self.daily_contributions.index[0]

                    if end_str:
                        end_dt_target = pd.to_datetime(end_str)
                        available_dates = self.daily_contributions.index[self.daily_contributions.index <= end_dt_target]
                        end_dt = available_dates[-1] if not available_dates.empty else None
                
                # Slice and sum contributions for the period
                if start_dt is not None and end_dt is not None and start_dt <= end_dt:
                    period_contributions = self.daily_contributions.loc[start_dt:end_dt]
                    all_contributions[name] = period_contributions.sum()
                else:
                    # If period is invalid, fill with NaNs
                    all_contributions[name] = pd.Series(np.nan, index=self.daily_contributions.columns)

            except Exception as e:
                print(f"\nError calculating contribution for period '{name}': {e}")
                all_contributions[name] = pd.Series(np.nan, index=self.daily_contributions.columns)

        # Format results into a DataFrame
        summary_df = pd.DataFrame(all_contributions)
        
        # Add a 'Total' row to verify it sums to the portfolio's arithmetic return
        if not summary_df.empty:
            summary_df.loc['Total Portfolio Return'] = summary_df.sum(axis=0)

        return summary_df

# Note: The __main__ block has been removed and placed in the testing file.
