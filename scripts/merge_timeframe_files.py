"""
merge_timeframe_files.py
------------------------
This script merges CSV files for each timeframe (e.g. '1min', '5min', etc.)
into a single Parquet file per timeframe.
It extracts the timeframe from filenames, concatenates data, and writes out the result.
"""

import os
from glob import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def merge_timeframe_files(data_directory, output_directory):
    """
    Merge CSV files for each timeframe into one Parquet file.

    Parameters:
        data_directory (str): Directory containing CSV files.
        output_directory (str): Destination directory for Parquet files.
    """
    os.makedirs(output_directory, exist_ok=True)
    # Find all tick data CSV files.
    all_files = glob(os.path.join(data_directory, 'BTCUSDT-tick-*.csv'))
    timeframes = set()
    # Extract unique timeframe values from filenames.
    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.split('-')
        timeframe = parts[2]
        timeframes.add(timeframe)
    for timeframe in sorted(timeframes):
        print(f"Processing timeframe: {timeframe}")
        # Get all files for the given timeframe.
        timeframe_files = sorted(glob(os.path.join(data_directory, f'BTCUSDT-tick-{timeframe}-*.csv')))
        df_list = []
        for file in tqdm(timeframe_files, desc=f"Reading files for {timeframe}"):
            try:
                df = pd.read_csv(file, parse_dates=['timestamp'],
                                 dtype={'open': 'float64', 'high': 'float64',
                                        'low': 'float64', 'close': 'float64',
                                        'volume': 'float64'})
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        if df_list:
            # Concatenate dataframes.
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
            combined_df.sort_values(by='timestamp', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            output_file = os.path.join(output_directory, f'BTCUSDT-tick-{timeframe}.parquet')
            try:
                table = pa.Table.from_pandas(combined_df)
                pq.write_table(table, output_file)
                print(f"Saved {output_file}")
            except Exception as e:
                print(f"Error saving {output_file}: {e}")
        else:
            print(f"No data found for timeframe {timeframe}")

if __name__ == '__main__':
    data_directory = 'data'
    output_directory = 'output_parquet'
    merge_timeframe_files(data_directory, output_directory)
