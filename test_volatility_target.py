# test_volatility_target.py (Enhanced & Fixed v3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- FIX 1: Add BDay import ---
from pandas.tseries.offsets import BDay, DateOffset
# --- END FIX 1 ---

# Ensure modules are importable (adjust path if necessary)
try:
    # Use the version with skip_recent_month included
    from portfolio_construction import EqualVolatilityPortfolio, VolatilityTargetPortfolio
    from backtester import Backtester
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure portfolio_construction.py and backtester.py are in the Python path.")
    exit()

# --- 1. Data Simulation ---

def generate_simulated_data(
    n_assets: int = 4,
    n_years: float = 8.0, # Allow float for short history test
    base_vol: float = 0.20,
    trading_days: int = 252,
    add_extreme_periods: bool = False
) -> pd.DataFrame:
    """
    Generates a DataFrame of simulated daily total return indices.
    Optionally includes periods of high and low volatility.
    """
    print(f"Generating simulated data ({'with extreme periods' if add_extreme_periods else 'standard'}, {n_years:.1f} years)...")

    # --- FIX 2: Ensure n_days is integer ---
    n_days = int(n_years * trading_days)
    # --- END FIX 2 ---

    if n_days <= 0:
        print("Warning: n_days is zero or negative, returning empty DataFrame.")
        return pd.DataFrame()

    dates = pd.date_range(start='2017-01-01', periods=n_days, freq='B')

    daily_vol = base_vol / np.sqrt(trading_days)
    vols = np.random.uniform(0.8, 1.2, n_assets) * daily_vol
    means = np.random.uniform(-0.0001, 0.0005, n_assets)

    returns_df = pd.DataFrame(index=dates, columns=[f'Asset{i+1}' for i in range(n_assets)])

    if add_extreme_periods and n_years >= 6:
        p1_end = int(n_days * 0.3)
        p2_start = p1_end
        p2_end = int(n_days * 0.5)
        p3_start = p2_end
        p3_end = int(n_days * 0.7)
        p4_start = p3_end

        periods = [
            (0, p1_end, 1.0), (p2_start, p2_end, 0.1),
            (p3_start, p3_end, 3.0), (p4_start, n_days, 1.0)
        ]
    else:
        periods = [(0, n_days, 1.0)]

    for i in range(n_assets):
        asset_returns = np.array([])
        for start, end, vol_multiplier in periods:
            period_len = end - start
            if period_len <= 0: continue
            asset_returns = np.concatenate([
                asset_returns,
                np.random.normal(loc=means[i], scale=vols[i] * vol_multiplier, size=period_len)
            ])
        returns_df[f'Asset{i+1}'] = asset_returns

    tri = 100 * (1 + returns_df).cumprod()

    if tri.isnull().values.any():
        warnings.warn("Simulated data contains NaNs! This shouldn't happen.")
        tri = tri.ffill().bfill()

    print("Simulated data generated.")
    return tri

# --- 2. Test Functions ---

def run_basic_test(underlying_asset_tri, target_volatility, vol_lookback_years, trading_days, tolerance):
    """Runs the original set of tests."""
    print("\n--- Running Test: Basic Execution & Volatility ---")
    try:
        evp = EqualVolatilityPortfolio(
            total_return_index=underlying_asset_tri, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        equal_vol_weights = evp.construct_portfolio()
        assert not equal_vol_weights.empty, "EVP failed."

        vtp = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights, underlying_asset_tri=underlying_asset_tri,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            rebalance_freq='M', trading_days_per_year=trading_days, max_leverage=3.0, min_leverage=0.1,
            skip_recent_month=False # Ensure it runs with default
        )
        vol_target_weights = vtp.construct_target_vol_weights().dropna()
        assert not vol_target_weights.empty, "VTP failed."

        bt = Backtester(
            strategy_weights=vol_target_weights,
            substrategy_total_return_index=underlying_asset_tri.loc[vol_target_weights.index],
            trading_days_per_year=trading_days
        )
        final_returns = bt.get_strategy_returns(frequency='daily')
        assert not final_returns.empty, "Backtester failed."

        rolling_vol = final_returns.rolling(window=trading_days).std() * np.sqrt(trading_days)
        rolling_vol = rolling_vol.dropna().replace(0, np.nan).dropna()
        assert not rolling_vol.empty, "Could not calculate rolling vol."

        mean_realized_vol = rolling_vol.mean()
        lower_bound = target_volatility * (1 - tolerance)
        upper_bound = target_volatility * (1 + tolerance)
        assert lower_bound <= mean_realized_vol <= upper_bound, \
            f"Mean vol {mean_realized_vol:.2%} outside tolerance [{lower_bound:.2%}, {upper_bound:.2%}]"

        leverage = vol_target_weights.sum(axis=1)
        correlation = leverage.corr(final_returns.loc[leverage.index])
        assert abs(correlation) < 0.15, f"Leverage/Return correlation too high: {correlation:.4f}"

        print("Basic Execution & Volatility Test: PASSED")
        # Return VTP instance for other tests
        return True, vol_target_weights, underlying_asset_tri, vtp
    except Exception as e:
        print(f"Basic Execution & Volatility Test: FAILED ({e})")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def test_single_date_step_through(vol_target_weights, underlying_asset_tri, target_volatility, vol_lookback_years, trading_days, vtp_instance):
    """Tests if a single date calculation uses only past data (with tolerance)."""
    print("\n--- Running Test: Single-Date Step-Through (Lookahead) ---")
    if vol_target_weights is None:
        print("Skipping: Basic test failed.")
        return

    try:
        test_date = vol_target_weights.index[trading_days]
        calc_date = test_date - BDay(1)

        leverage_full_run = vol_target_weights.sum(axis=1).loc[test_date]

        tri_subset = underlying_asset_tri.loc[:calc_date]
        evp_weights_full = EqualVolatilityPortfolio(
             total_return_index=underlying_asset_tri, lookback_years=1, rebalance_freq='Q'
        ).construct_portfolio()

        vtp_subset = VolatilityTargetPortfolio(
            equal_vol_weights=evp_weights_full,
            underlying_asset_tri=tri_subset,
            target_volatility=target_volatility,
            volatility_lookback_years=vol_lookback_years,
            trading_days_per_year=trading_days,
            skip_recent_month=False # Match the basic run setting
        )

        leverage_subset = vtp_subset._calculate_leverage(calc_date)

        assert np.isclose(leverage_full_run, leverage_subset, atol=0.05), \
            f"Leverage mismatch on {test_date.date()}: Full={leverage_full_run:.4f}, Subset={leverage_subset:.4f}"
        print("Single-Date Step-Through Test: PASSED (with tolerance)")
    except Exception as e:
        print(f"Single-Date Step-Through Test: FAILED ({e})")
        import traceback
        traceback.print_exc()


def test_data_truncation_lookahead(underlying_asset_tri, target_volatility, vol_lookback_years, trading_days):
    """Tests if truncating future data affects past weights."""
    print("\n--- Running Test: Data Truncation (Lookahead) ---")
    try:
        evp_full = EqualVolatilityPortfolio(
            total_return_index=underlying_asset_tri, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        evp_weights_full = evp_full.construct_portfolio()
        vtp_full = VolatilityTargetPortfolio(
            equal_vol_weights=evp_weights_full, underlying_asset_tri=underlying_asset_tri,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            trading_days_per_year=trading_days, skip_recent_month=False
        )
        weights_full = vtp_full.construct_target_vol_weights().dropna()

        trunc_idx = int(len(weights_full) * 0.7)
        if trunc_idx <= 0 : raise ValueError("Not enough data to truncate.")
        t_trunc = weights_full.index[trunc_idx]
        print(f"Truncating data after {t_trunc.date()}...")

        tri_trunc = underlying_asset_tri.loc[:t_trunc]

        evp_trunc = EqualVolatilityPortfolio(
            total_return_index=tri_trunc, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        evp_weights_trunc = evp_trunc.construct_portfolio()
        vtp_trunc = VolatilityTargetPortfolio(
            equal_vol_weights=evp_weights_trunc, underlying_asset_tri=tri_trunc,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            trading_days_per_year=trading_days, skip_recent_month=False
        )
        weights_trunc = vtp_trunc.construct_target_vol_weights().dropna()

        last_trunc_date = weights_trunc.index[-1]
        weights_full_subset = weights_full.loc[:last_trunc_date]

        weights_full_aligned, weights_trunc_aligned = weights_full_subset.align(weights_trunc, join='inner', axis=0)

        assert not weights_full_aligned.empty, "Alignment failed."
        assert weights_full_aligned.equals(weights_trunc_aligned), \
            "Weights changed when future data was truncated!"

        print("Data Truncation Test: PASSED")
    except Exception as e:
        print(f"Data Truncation Test: FAILED ({e})")
        import traceback
        traceback.print_exc()

def test_extreme_volatility(target_volatility, vol_lookback_years, trading_days):
    """Tests leverage behaviour during high/low vol."""
    print("\n--- Running Test: Extreme Volatility ---")
    try:
        tri_extreme = generate_simulated_data(n_years=10, add_extreme_periods=True)

        evp = EqualVolatilityPortfolio(
            total_return_index=tri_extreme, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        equal_vol_weights = evp.construct_portfolio()

        vtp = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights, underlying_asset_tri=tri_extreme,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            rebalance_freq='M', trading_days_per_year=trading_days, max_leverage=3.0, min_leverage=0.1
        )
        vol_target_weights = vtp.construct_target_vol_weights().dropna()
        leverage = vol_target_weights.sum(axis=1)

        n_days = len(tri_extreme)
        low_vol_start = tri_extreme.index[int(n_days * 0.3)]
        low_vol_end = tri_extreme.index[int(n_days * 0.5)]
        high_vol_start = tri_extreme.index[int(n_days * 0.5)]
        high_vol_end = tri_extreme.index[int(n_days * 0.7)]

        check_low_start = low_vol_start + DateOffset(years=vol_lookback_years)
        check_high_start = high_vol_start + DateOffset(years=vol_lookback_years)
        check_low_end = low_vol_end + DateOffset(years=vol_lookback_years)
        check_high_end = high_vol_end + DateOffset(years=vol_lookback_years)


        leverage_in_low_vol = leverage.loc[check_low_start:check_low_end].mean()
        leverage_in_high_vol = leverage.loc[check_high_start:check_high_end].mean()

        print(f"Mean Leverage during Low Vol period (approx): {leverage_in_low_vol:.2f}")
        print(f"Mean Leverage during High Vol period (approx): {leverage_in_high_vol:.2f}")

        assert leverage_in_low_vol > leverage_in_high_vol, "Leverage didn't respond correctly to vol changes."
        print(f"Max leverage hit: {np.isclose(leverage.max(), 3.0)}. Min leverage hit: {np.isclose(leverage.min(), 0.1)}")

        print("Extreme Volatility Test: PASSED")
    except Exception as e:
        print(f"Extreme Volatility Test: FAILED ({e})")
        import traceback
        traceback.print_exc()

def test_rebalance_frequency(underlying_asset_tri, target_volatility, vol_lookback_years, trading_days):
    """Tests if leverage changes only happen on rebalance dates."""
    print("\n--- Running Test: Rebalance Frequency (Monthly) ---")
    try:
        evp = EqualVolatilityPortfolio(
            total_return_index=underlying_asset_tri, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        equal_vol_weights = evp.construct_portfolio()

        vtp = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights, underlying_asset_tri=underlying_asset_tri,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            rebalance_freq='M',
            trading_days_per_year=trading_days
        )
        vol_target_weights = vtp.construct_target_vol_weights().dropna()
        leverage = vol_target_weights.sum(axis=1)
        leverage_diff = leverage.diff().dropna()

        reb_calc_dates = vtp._get_leverage_rebalance_dates()
        reb_apply_dates_map = {}
        for r_calc in reb_calc_dates:
            apply_date_idx = leverage.index.searchsorted(r_calc, side='right')
            if apply_date_idx < len(leverage.index):
                reb_apply_dates_map[leverage.index[apply_date_idx]] = True

        for date, diff_value in leverage_diff.items():
            if date not in reb_apply_dates_map:
                 assert np.isclose(diff_value, 0.0, atol=1e-6), \
                     f"Leverage changed on non-rebalance date {date.date()} by {diff_value:.6f}"

        print("Rebalance Frequency Test: PASSED")
    except Exception as e:
        print(f"Rebalance Frequency Test: FAILED ({e})")
        import traceback
        traceback.print_exc()


def test_insufficient_history(vol_lookback_years, trading_days):
    """Tests if the class handles insufficient history correctly."""
    print("\n--- Running Test: Insufficient History ---")
    try:
        tri_short = generate_simulated_data(n_years=vol_lookback_years - 0.5, trading_days=trading_days)

        try:
             evp = EqualVolatilityPortfolio(
                 total_return_index=tri_short, lookback_years=1,
                 rebalance_freq='Q', trading_days_per_year=trading_days
             )
             equal_vol_weights = evp.construct_portfolio()
        except ValueError:
             print("EVP correctly raised ValueError for short history.")
             equal_vol_weights = pd.DataFrame(1/tri_short.shape[1], index=tri_short.index, columns=tri_short.columns)

        vtp = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights,
            underlying_asset_tri=tri_short,
            target_volatility=0.15,
            volatility_lookback_years=vol_lookback_years,
            trading_days_per_year=trading_days
        )
        weights = vtp.construct_target_vol_weights()
        assert weights.dropna().empty, "VTP produced weights despite insufficient history."
        print("Insufficient History Test: PASSED (VTP correctly produced no valid weights)")

    except ValueError as e:
        print(f"Insufficient History Test: PASSED (Correctly raised ValueError: {e})")
    except Exception as e:
        print(f"Insufficient History Test: FAILED (Unexpected error: {e})")
        import traceback
        traceback.print_exc()

# --- NEW TEST ---
def test_skip_recent_month(underlying_asset_tri, target_volatility, vol_lookback_years, trading_days):
    """Tests if the skip_recent_month feature works."""
    print("\n--- Running Test: Skip Recent Month ---")
    try:
        evp = EqualVolatilityPortfolio(
            total_return_index=underlying_asset_tri, lookback_years=1, rebalance_freq='Q',
            trading_days_per_year=trading_days
        )
        equal_vol_weights = evp.construct_portfolio()

        # Run with skip_recent_month=False
        vtp_no_skip = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights, underlying_asset_tri=underlying_asset_tri,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            rebalance_freq='M', trading_days_per_year=trading_days,
            skip_recent_month=False # Explicitly False
        )
        weights_no_skip = vtp_no_skip.construct_target_vol_weights().dropna()

        # Run with skip_recent_month=True
        vtp_skip = VolatilityTargetPortfolio(
            equal_vol_weights=equal_vol_weights, underlying_asset_tri=underlying_asset_tri,
            target_volatility=target_volatility, volatility_lookback_years=vol_lookback_years,
            rebalance_freq='M', trading_days_per_year=trading_days,
            skip_recent_month=True # Explicitly True
        )
        weights_skip = vtp_skip.construct_target_vol_weights().dropna()

        # Align for comparison
        weights_no_skip_aligned, weights_skip_aligned = weights_no_skip.align(weights_skip, join='inner')

        assert not weights_no_skip_aligned.empty, "Alignment resulted in empty DataFrames."
        # Assert that the weights are NOT equal. They should differ.
        assert not weights_no_skip_aligned.equals(weights_skip_aligned), \
            "Weights are identical with and without skipping recent month - feature might not be working."

        print("Skip Recent Month Test: PASSED")
    except Exception as e:
        print(f"Skip Recent Month Test: FAILED ({e})")
        import traceback
        traceback.print_exc()
# --- END NEW TEST ---

# --- 3. Main Execution ---

if __name__ == '__main__':
    # --- Configuration ---
    N_ASSETS = 5
    N_YEARS = 8
    TARGET_VOLATILITY = 0.15
    VOL_LOOKBACK_YEARS = 1
    TRADING_DAYS_PER_YEAR = 252
    VOL_TOLERANCE = 0.35

    print("="*50)
    simulated_tri_std = generate_simulated_data(
        n_assets=N_ASSETS, n_years=N_YEARS,
        trading_days=TRADING_DAYS_PER_YEAR, add_extreme_periods=False
    )
    print("="*50)

    success, weights, tri, vtp_instance = run_basic_test(
        simulated_tri_std, TARGET_VOLATILITY, VOL_LOOKBACK_YEARS,
        TRADING_DAYS_PER_YEAR, VOL_TOLERANCE
    )

    if success:
        test_single_date_step_through(weights, tri, TARGET_VOLATILITY, VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR, vtp_instance)
        test_data_truncation_lookahead(tri, TARGET_VOLATILITY, VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR)
        test_rebalance_frequency(tri, TARGET_VOLATILITY, VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR)
        # Add call to the new test
        test_skip_recent_month(tri, TARGET_VOLATILITY, VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR)
    else:
        print("\nSkipping additional tests due to basic test failure.")

    test_extreme_volatility(TARGET_VOLATILITY, VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR)
    test_insufficient_history(VOL_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR)

    print("\n" + "="*50)
    print("All Tests Completed.")
    print("="*50)