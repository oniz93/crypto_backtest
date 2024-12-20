import glob
import os

import pandas as pd


def csv_to_parquet(csv_folder: str, output_directory: str):
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
    # Iterate over each CSV file and read it into a DataFrame
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])

            # Optional: Sort the DataFrame by timestamp
            df.sort_values(by='timestamp', inplace=True)

            # Optional: Reset index after sorting
            df.reset_index(drop=True, inplace=True)
            print(f"Loaded {file} with {len(df)} rows.")

            # Write the combined DataFrame to a Parquet file
            try:
                df.to_parquet(output_directory + file, engine='pyarrow', compression='snappy')
                print(f"Successfully wrote combined data to {output_directory + file}")
            except Exception as e:
                print(f"Error writing to Parquet file: {e}")
        except Exception as e:
            print(f"Error reading {file}: {e}")


if __name__ == "__main__":
    # Define the folder containing CSV files
    csv_directory = 'data/'  # Replace with your actual folder path

    # Define the output Parquet file path
    parquet_output = 'trades_history/'  # Replace with your desired output path

    # Call the function to combine CSVs into a Parquet file
    csv_to_parquet(csv_folder=csv_directory, output_directory=parquet_output)
