import os
from glob import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def merge_funding_history(data_directory, output_file):
    """
    Merges funding history CSV files into a single Parquet file.

    Parameters:
    - data_directory: str, the directory where the CSV files are located.
    - output_file: str, the path to the output Parquet file.
    """
    # Ensure the data directory exists
    if not os.path.exists(data_directory):
        print(f"The directory {data_directory} does not exist.")
        return

    # Get list of all funding history CSV files matching the pattern
    all_files = glob(os.path.join(data_directory, 'Funding History 2024-11-*.csv'))

    if not all_files:
        print("No funding history files found matching the pattern.")
        return

    df_list = []

    print("Reading CSV files:")
    for file in tqdm(all_files):
        try:
            # Read CSV file
            df = pd.read_csv(
                file,
                parse_dates=['timestamp'],
                dtype={
                    'symbol': str,
                    'fundingInterval': str,
                    'fundingRate': float,
                    'fundingRateDaily': float
                }
            )
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if df_list:
        # Concatenate all DataFrames
        combined_df = pd.concat(df_list, ignore_index=True)
        # Drop duplicates based on the 'timestamp' column
        combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
        # Sort by 'timestamp'
        combined_df.sort_values(by='timestamp', inplace=True)
        # Reset index
        combined_df.reset_index(drop=True, inplace=True)

        # Save to Parquet
        try:
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, output_file)
            print(f"Saved combined data to {output_file}")
        except Exception as e:
            print(f"Error saving to Parquet file: {e}")
    else:
        print("No data was read from the CSV files.")


if __name__ == '__main__':
    data_directory = 'data/funding'  # Directory containing your CSV files
    output_file = 'output_parquet/funding_history.parquet'  # Output Parquet file
    merge_funding_history(data_directory, output_file)
