"""
merge_trades.py
---------------
This script combines multiple CSV files containing aggregated trade data into a single Parquet file.
It reads each CSV, concatenates them, sorts by timestamp, and writes the output using PyArrow.
"""

import glob
import os
import pandas as pd

def combine_csv_to_parquet(csv_folder: str, output_file: str):
    """
    Combine multiple CSV files into one Parquet file.

    Parameters:
        csv_folder (str): Folder containing CSV files.
        output_file (str): Path for the output Parquet file.
    """
    csv_pattern = os.path.join(csv_folder, 'BTCUSDT-trades-aggregated-*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in folder '{csv_folder}' with pattern 'BTCUSDT-trades-aggregated-*.csv'")
    df_list = []
    # Read each CSV file.
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            df_list.append(df)
            print(f"Loaded {file} with {len(df)} rows.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    # Concatenate all dataframes.
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined DataFrame has {len(combined_df)} rows.")
    # Sort by timestamp.
    combined_df.sort_values(by='timestamp', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    try:
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        print(f"Successfully wrote combined data to {output_file}")
    except Exception as e:
        print(f"Error writing to Parquet file: {e}")

if __name__ == "__main__":
    csv_directory = 'data/'
    parquet_output = 'output_parquet/BTCUSDT-trades.parquet'
    combine_csv_to_parquet(csv_folder=csv_directory, output_file=parquet_output)
