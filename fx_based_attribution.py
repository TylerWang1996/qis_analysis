import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional

# ==============================================================================
# Core Analysis Function with Detailed Currency Grouping
# ==============================================================================

def analyze_fx_strategy(
    excel_file_path: str,
    returns_sheet_name: str,
    weights_sheet_name: str,
    mapping_file_path: str,
    mapping_sheet_name: str,
    ticker_col_name: str,
    currency_col_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Analyzes FX strategy returns using a detailed currency grouping and lagged weights.

    This version replaces the binary G10/EM split with a more granular
    multi-category grouping defined in the CURRENCY_GROUPS dictionary.

    Handles sparse weight data (only on rebalance dates) by forward-filling.
    Aggregates returns based on:
    1. Currency Group: As defined in the CURRENCY_GROUPS dictionary.
    2. Position Direction: Short USD (positive weight) vs. Long USD (negative weight),
       using forward-filled weights from the *previous* period (t-1) for returns at period t.

    Args:
        excel_file_path (str): Path to the Excel file with returns/weights data.
        returns_sheet_name (str): Sheet name for return attribution data.
        weights_sheet_name (str): Sheet name for weight data (can be sparse).
        mapping_file_path (str): Path to the Excel file with ticker-to-currency mapping.
        mapping_sheet_name (str): Sheet name for the mapping table.
        ticker_col_name (str): Column name for tickers in the mapping sheet.
        currency_col_name (str): Column name for currency pairs (e.g., 'EURUSD').
        start_date (Optional[str]): Optional start date for analysis ('YYYY-MM-DD').
        end_date (Optional[str]): Optional end date for analysis ('YYYY-MM-DD').

    Returns:
        Dict[str, float]: Dictionary of aggregated returns for all defined categories.
    """

    # --- Configuration: Define Detailed Currency Groups ---
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
    
    # Create a reverse map for quick lookup: { 'EUR': 'European Majors', 'AUD': 'Antipodeans', ... }
    currency_to_group_map = {
        currency: group for group, currencies in CURRENCY_GROUPS.items() for currency in currencies
    }
    print("Using detailed currency group mapping for attribution.")


    # --- Step 1a: Load Ticker-to-Currency Mapping ---
    print(f"Loading ticker mapping from: {mapping_file_path} (Sheet: '{mapping_sheet_name}')")
    try:
        mapping_df = pd.read_excel(
            mapping_file_path,
            sheet_name=mapping_sheet_name
        )
        if ticker_col_name not in mapping_df.columns or currency_col_name not in mapping_df.columns:
            print(f"Error: Required mapping columns ('{ticker_col_name}', '{currency_col_name}') not found.")
            return {}
        
        mapping_df = mapping_df.dropna(subset=[ticker_col_name, currency_col_name])
        mapping_df = mapping_df.drop_duplicates(subset=[ticker_col_name], keep='first')
        ticker_to_currency_map: Dict[str, str] = pd.Series(
            mapping_df[currency_col_name].values,
            index=mapping_df[ticker_col_name]
        ).to_dict()
        print(f"Successfully loaded mapping for {len(ticker_to_currency_map)} unique tickers.")

    except FileNotFoundError:
        print(f"Error: Mapping file not found at '{mapping_file_path}'")
        return {}
    except Exception as e:
        print(f"Error reading mapping file '{mapping_file_path}': {e}")
        return {}

    # --- Step 1b: Load Returns and Weights Data ---
    print(f"Loading returns/weights data from: {excel_file_path}")
    try:
        returns_df = pd.read_excel(excel_file_path, sheet_name=returns_sheet_name, index_col=0, parse_dates=True)
        weights_df = pd.read_excel(excel_file_path, sheet_name=weights_sheet_name, index_col=0, parse_dates=True)
        print(f"Successfully loaded sheets: '{returns_sheet_name}' and '{weights_sheet_name}'")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{excel_file_path}'")
        return {}
    except Exception as e:
        print(f"Error reading data file '{excel_file_path}': {e}")
        return {}

    # --- Step 2: Data Validation, Alignment, Forward Fill, and Shift ---
    common_tickers = returns_df.columns.intersection(weights_df.columns)
    if not common_tickers.tolist():
        print("Error: No common tickers found between returns and weights sheets.")
        return {}
    
    returns_df, weights_df = returns_df[common_tickers], weights_df[common_tickers]

    print("Aligning weights, forward-filling, and shifting for t-1 logic...")
    weights_aligned = weights_df.reindex(returns_df.index)
    weights_filled = weights_aligned.ffill()
    weights_final = weights_filled.shift(1)
    
    common_index = returns_df.index.intersection(weights_final.index)
    returns_df = returns_df.loc[common_index]
    weights_final = weights_final.loc[common_index]
    print(f"Aligned data post-processing: {len(common_index)} dates and {len(common_tickers)} common tickers.")
    
    # --- Step 3: Filtering by Date ---
    if start_date:
        returns_df = returns_df.loc[start_date:]
        weights_final = weights_final.loc[start_date:]
    if end_date:
        returns_df = returns_df.loc[:end_date]
        weights_final = weights_final.loc[:end_date]

    if returns_df.empty:
        print(f"Warning: No data available for the specified date range ({start_date} to {end_date}).")
        return {}
    print(f"Data filtered: {len(returns_df.index)} dates remaining.")

    # --- Step 4: Categorization and Aggregation ---
    print("Categorizing positions using detailed groups and aggregating returns...")
    is_short_usd_mask = weights_final > 0
    is_long_usd_mask = weights_final < 0

    # Initialize a results dictionary dynamically from the currency groups (excluding USD itself)
    results: Dict[str, float] = {
        f"{group} {direction}": 0.0
        for group in CURRENCY_GROUPS if group != 'USD'
        for direction in ["Short USD", "Long USD"]
    }

    # Iterate through each common ticker to attribute its returns
    for ticker in returns_df.columns:
        currency_pair = ticker_to_currency_map.get(ticker)
        if not currency_pair or len(currency_pair) < 6: # e.g., 'EURUSD'
            continue

        # Extract the base currency (the first 3 letters) from the pair
        base_currency = currency_pair[:3].upper()
        group_name = currency_to_group_map.get(base_currency)

        # Proceed only if the currency belongs to a defined group (and is not USD)
        if group_name and group_name != 'USD':
            ticker_returns = returns_df[ticker]
            
            # Calculate returns for short and long positions based on the t-1 weight masks
            short_usd_returns = ticker_returns.where(is_short_usd_mask[ticker]).sum()
            long_usd_returns = ticker_returns.where(is_long_usd_mask[ticker]).sum()
            
            # Add the calculated returns to the correct category totals
            results[f"{group_name} Short USD"] += short_usd_returns
            results[f"{group_name} Long USD"] += long_usd_returns

    # --- Step 5: Format and Return Results ---
    final_results = {k: v if pd.notna(v) else 0.0 for k, v in results.items()}
    print("Analysis complete.")
    return final_results

# ==============================================================================
# Script Execution Block
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution block:
    - Configure file paths, sheet names, mapping details, and analysis period.
    - Calls the analysis function.
    - Prints the results.
    - Generates and displays a bar chart visualization.
    """

    # --- !!! USER CONFIGURATION REQUIRED !!! ---

    # --- 1. Data File Details ---
    # IMPORTANT: Use raw string literals (r'...') or forward slashes ('/') for paths to avoid issues.
    YOUR_EXCEL_FILE = r'path/to/your/fx_data.xlsx' # <--- CHANGE THIS
    RETURNS_SHEET = 'Returns'                      # <--- CHANGE THIS
    WEIGHTS_SHEET = 'Weights'                      # <--- CHANGE THIS (Can be sparse)

    # --- 2. Ticker Mapping Details ---
    MAPPING_FILE = r'path/to/your/mapping_data.xlsx' # <--- CHANGE THIS
    MAPPING_SHEET = 'Mapping'                         # <--- CHANGE THIS
    TICKER_COLUMN = 'Substrategy Ticker'              # <--- CHANGE THIS
    CURRENCY_COLUMN = 'Currency Pair'                 # <--- CHANGE THIS

    # --- 3. Analysis Period (Optional) ---
    START_DATE = '2023-01-01' # Example start date (or None to use all data)
    END_DATE = '2023-12-31'   # Example end date (or None to use all data)
    # START_DATE = None
    # END_DATE = None

    # --- End of User Configuration ---


    # --- Run the analysis using configured parameters ---
    print("="*60)
    print("Starting FX Strategy Return Attribution Analysis")
    print("Using Detailed Currency Grouping and Lagged Weights")
    print("="*60)
    
    # Check if placeholder paths have been changed
    if 'path/to/your' in YOUR_EXCEL_FILE or 'path/to/your' in MAPPING_FILE:
        print("!!! CONFIGURATION ERROR !!!")
        print("Please update the placeholder file paths (YOUR_EXCEL_FILE, MAPPING_FILE) before running.")
    else:
        aggregated_returns = analyze_fx_strategy(
            excel_file_path=YOUR_EXCEL_FILE,
            returns_sheet_name=RETURNS_SHEET,
            weights_sheet_name=WEIGHTS_SHEET,
            mapping_file_path=MAPPING_FILE,
            mapping_sheet_name=MAPPING_SHEET,
            ticker_col_name=TICKER_COLUMN,
            currency_col_name=CURRENCY_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE
        )

        # --- Print & Visualize Results ---
        if aggregated_returns:
            print("\n--- Aggregated FX Strategy Returns by Detailed Group ---")
            period_str = f"{START_DATE or 'Start'} to {END_DATE or 'End'}"
            print(f"Period: {period_str}")
            print("-" * 50)
            total_return = 0.0
            
            # Sort results for cleaner presentation
            sorted_results = dict(sorted(aggregated_returns.items()))
            
            for category, ret in sorted_results.items():
                print(f"{category:<25}: {ret:,.2f}")
                total_return += ret
            print("-" * 50)
            print(f"{'Total Return':<25}: {total_return:,.2f}")
            print("-" * 50)

            # --- Visualization (Enhanced for more categories) ---
            print("\nGenerating plot...")
            categories = list(sorted_results.keys())
            values = list(sorted_results.values())

            plt.figure(figsize=(16, 9)) # Wider figure for more labels
            
            # Use a colormap for better visual distinction between many bars
            color_map = plt.get_cmap('viridis') 
            colors = color_map(np.linspace(0, 1, len(categories)))

            bars = plt.bar(categories, values, color=colors)

            plt.ylabel("Aggregated Return (USD)")
            plt.title(f"FX Strategy Return Attribution ({period_str})\nUsing Lagged Weights & Detailed Currency Groups", fontsize=16)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels for readability
            plt.axhline(0, color='grey', linewidth=0.8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on bars
            for bar in bars:
                yval = bar.get_height()
                # Position label inside the bar if it's tall enough, else outside
                va_pos = 'bottom' if yval >= 0 else 'top'
                label_y_offset = yval * 0.02 if yval >=0 else yval*0.1
                plt.text(
                    x=bar.get_x() + bar.get_width() / 2.0, 
                    y=yval - label_y_offset if yval < 0 else yval + label_y_offset,
                    s=f'{yval:,.0f}', 
                    va=va_pos, 
                    ha='center', 
                    fontsize=9
                )

            plt.tight_layout()
            print("Displaying bar chart visualization (close the plot window to exit script)...")
            plt.show()

        else:
            print("\nAnalysis did not produce results. Please check input files, sheet names, column names, dates, and error messages above.")

    print("\nScript finished.")
