import os
from glob import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def merge_timeframe_files(data_directory, output_directory):
    """
    Merges CSV files for each timeframe into a single Parquet file.

    Parameters:
    - data_directory: str, the directory where the CSV files are located.
    - output_directory: str, the directory where the Parquet files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get list of all CSV files
    all_files = glob(os.path.join(data_directory, 'BTCUSDT-tick-*.csv'))

    # Extract timeframes from filenames
    timeframes = set()
    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.split('-')
        timeframe = parts[2]
        timeframes.add(timeframe)

    # Process each timeframe
    for timeframe in sorted(timeframes):
        print(f"Processing timeframe: {timeframe}")
        # Get all files for the timeframe, sorted by date
        timeframe_files = sorted(glob(os.path.join(data_directory, f'BTCUSDT-tick-{timeframe}-*.csv')))

        df_list = []
        for file in tqdm(timeframe_files, desc=f"Reading files for {timeframe}"):
            try:
                # Read CSV file
                df = pd.read_csv(
                    file,
                    parse_dates=['timestamp'],
                    dtype={
                        'open': 'float64',
                        'high': 'float64',
                        'low': 'float64',
                        'close': 'float64',
                        'volume': 'float64'
                    }
                )
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        if df_list:
            # Concatenate all dataframes
            combined_df = pd.concat(df_list, ignore_index=True)
            # Drop duplicates if any
            combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
            # Sort by timestamp
            combined_df.sort_values(by='timestamp', inplace=True)
            # Reset index
            combined_df.reset_index(drop=True, inplace=True)

            # Save to Parquet
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
    data_directory = 'data'  # Directory containing your CSV files
    output_directory = 'output_parquet'  # Directory to save the Parquet files
    merge_timeframe_files(data_directory, output_directory)
