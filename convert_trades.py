"""
convert_trades.py
-----------------
This script converts multiple CSV files containing trade data into Parquet format.
It uses glob to locate CSV files, reads them into pandas DataFrames, and writes them
out using PyArrow with snappy compression.
"""

import glob
import os
import pandas as pd

def csv_to_parquet(csv_folder: str, output_directory: str):
    """
    Convert CSV files to Parquet format.

    Parameters:
        csv_folder (str): Folder containing CSV files.
        output_directory (str): Destination folder for Parquet files.
    """
    # Create a file pattern to match the trade CSV files.
    csv_pattern = os.path.join(csv_folder, 'BTCUSDT-trades-aggregated-*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in folder '{csv_folder}' with pattern 'BTCUSDT-trades-aggregated-*.csv'")
    # Loop through each file and process it.
    for file in csv_files:
        try:
            # Read the CSV file and parse the 'timestamp' column as datetime.
            df = pd.read_csv(file, parse_dates=['timestamp'])
            # Sort the DataFrame by timestamp.
            df.sort_values(by='timestamp', inplace=True)
            # Reset the index for consistency.
            df.reset_index(drop=True, inplace=True)
            print(f"Loaded {file} with {len(df)} rows.")
            try:
                # Write the DataFrame to a Parquet file using snappy compression.
                df.to_parquet(output_directory + file, engine='pyarrow', compression='snappy')
                print(f"Successfully wrote combined data to {output_directory + file}")
            except Exception as e:
                print(f"Error writing to Parquet file: {e}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    csv_directory = 'data/'
    parquet_output = 'trades_history/'
    csv_to_parquet(csv_folder=csv_directory, output_directory=parquet_output)
