import glob
import os

import pandas as pd


def combine_csv_to_parquet(csv_folder: str, output_file: str):
    """
    Combines multiple CSV files in a specified folder into a single Parquet file.

    Parameters:
    - csv_folder (str): Path to the folder containing CSV files.
    - output_file (str): Path for the output Parquet file.

    Example:
        combine_csv_to_parquet('data/', 'combined_trades.parquet')
    """
    # Define the pattern to match your CSV files
    csv_pattern = os.path.join(csv_folder, 'BTCUSDT-trades-aggregated-*.csv')

    # Use glob to find all matching CSV files
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in folder '{csv_folder}' with pattern 'BTCUSDT-trades-aggregated-*.csv'")

    # Initialize a list to hold DataFrames
    df_list = []

    # Iterate over each CSV file and read it into a DataFrame
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            df_list.append(df)
            print(f"Loaded {file} with {len(df)} rows.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined DataFrame has {len(combined_df)} rows.")

    # Optional: Sort the DataFrame by timestamp
    combined_df.sort_values(by='timestamp', inplace=True)

    # Optional: Reset index after sorting
    combined_df.reset_index(drop=True, inplace=True)

    # Write the combined DataFrame to a Parquet file
    try:
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        print(f"Successfully wrote combined data to {output_file}")
    except Exception as e:
        print(f"Error writing to Parquet file: {e}")


if __name__ == "__main__":
    # Define the folder containing CSV files
    csv_directory = 'data/'  # Replace with your actual folder path

    # Define the output Parquet file path
    parquet_output = 'output_parquet/BTCUSDT-trades.parquet'  # Replace with your desired output path

    # Call the function to combine CSVs into a Parquet file
    combine_csv_to_parquet(csv_folder=csv_directory, output_file=parquet_output)
