"""
merge_funding_history.py
------------------------
This script merges multiple CSV files containing funding history into a single Parquet file.
It uses glob to find files, reads them into pandas DataFrames, concatenates them, removes duplicates,
and writes the combined data using PyArrow.
"""

import os
from glob import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def merge_funding_history(data_directory, output_file):
    """
    Merge funding history CSV files into one Parquet file.

    Parameters:
        data_directory (str): Directory containing funding history CSV files.
        output_file (str): Output Parquet file path.
    """
    if not os.path.exists(data_directory):
        print(f"The directory {data_directory} does not exist.")
        return
    # Find all CSV files matching the funding history pattern.
    all_files = glob(os.path.join(data_directory, 'Funding History 2024-11-*.csv'))
    if not all_files:
        print("No funding history files found matching the pattern.")
        return
    df_list = []
    print("Reading CSV files:")
    # Loop over each file with a progress bar.
    for file in tqdm(all_files):
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'],
                             dtype={'symbol': str, 'fundingInterval': str,
                                    'fundingRate': float, 'fundingRateDaily': float})
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    if df_list:
        # Concatenate all DataFrames.
        combined_df = pd.concat(df_list, ignore_index=True)
        # Remove duplicates based on the 'timestamp' column.
        combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
        # Sort the combined DataFrame by timestamp.
        combined_df.sort_values(by='timestamp', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        try:
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, output_file)
            print(f"Saved combined data to {output_file}")
        except Exception as e:
            print(f"Error saving to Parquet file: {e}")
    else:
        print("No data was read from the CSV files.")

if __name__ == '__main__':
    data_directory = 'data/funding'
    output_file = 'output_parquet/funding_history.parquet'
    merge_funding_history(data_directory, output_file)
