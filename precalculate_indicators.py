# precalculate_indicators.py

import json
import os
from functools import partial
from itertools import product
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

# Import DataLoader from your src module
from src.data_loader import DataLoader

# Global variable to hold the shared DataFrame
shared_tick_data = None


def init_worker(tick_data):
    """
    Initializes the worker by setting the global shared_tick_data.
    This function is called once per worker process.
    """
    global shared_tick_data
    shared_tick_data = tick_data


def load_indicator_config(config_file: str) -> dict:
    """
    Loads the indicators configuration from a JSON file.

    Parameters:
    - config_file (str): Path to the JSON configuration file.

    Returns:
    - dict: Indicators configuration.
    """
    with open(config_file, 'r') as f:
        indicators = json.load(f)
    return indicators


def has_float_params(params: dict) -> bool:
    """
    Checks if any parameter in the params dict is of float type.

    Parameters:
    - params (dict): Parameters dictionary.

    Returns:
    - bool: True if any parameter has float type, False otherwise.
    """
    for param_range in params.values():
        # Check if any element in the list is a float
        if isinstance(param_range, list):
            if any(isinstance(val, float) for val in param_range):
                return True
    return False


def generate_parameter_combinations(params: dict, step: int = 5) -> list:
    """
    Generates a list of parameter dictionaries based on the provided ranges.

    Parameters:
    - params (dict): Parameters with their ranges.
    - step (int, optional): Step size for integer parameters. Defaults to 5.

    Returns:
    - list: List of parameter dictionaries.
    """
    param_names = []
    param_values = []

    for param, range_vals in params.items():
        if isinstance(range_vals, list) and len(range_vals) == 2:
            low, high = range_vals
            if isinstance(low, int) and isinstance(high, int):
                values = list(range(low, high + 1, step))
            else:
                # Skip non-integer parameters
                values = []
        else:
            # Skip parameters that don't have a proper range
            values = []

        if values:
            param_names.append(param)
            param_values.append(values)

    # Generate all combinations
    combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]
    return combinations


def create_filename(indicator_name: str, timeframe: str, params: dict) -> str:
    """
    Creates a filename based on the indicator name, timeframe, and parameters.

    Parameters:
    - indicator_name (str): Name of the indicator.
    - timeframe (str): Timeframe string (e.g., '1T', '5T').
    - params (dict): Parameters dictionary.

    Returns:
    - str: Formatted filename.
    """
    if not params:
        filename = f"{indicator_name}-{timeframe}.feather"
    else:
        param_parts = [f"{key}-{value}" for key, value in params.items()]
        filename = f"{indicator_name}-{timeframe}-" + "-".join(param_parts) + ".feather"
    # Replace any spaces or special characters if necessary
    filename = filename.replace(" ", "_")
    return filename


def calculate_and_save_all_timeframes(indicator_info: tuple, output_dir: str, timeframes: list):
    """
    Calculates an indicator across all specified timeframes, reshapes the data to 1-minute intervals,
    and saves them as Parquet files.

    Parameters:
    - indicator_info (tuple): Tuple containing (indicator_name, params).
    - output_dir (str): Directory to save the Parquet files.
    - timeframes (list): List of timeframe strings.
    """
    indicator_name, params = indicator_info
    for timeframe in timeframes:
        try:
            # Access the shared tick_data
            data_loader = DataLoader()
            data_loader.tick_data = shared_tick_data  # Assign shared data
            data_loader.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']  # Ensure timeframes match

            # Calculate the indicator
            indicator_df = data_loader.calculate_indicator(indicator_name, params, timeframe)
            if indicator_df.empty:
                print(f"Empty DataFrame for {indicator_name} with params {params} on {timeframe}. Skipping.")
                continue

            # Shift the index backward by the timeframe duration
            shift_duration = pd.to_timedelta(timeframe)
            indicator_df_shifted = indicator_df.copy()
            indicator_df_shifted.index = indicator_df_shifted.index - shift_duration

            # Resample to 1-minute intervals and forward-fill the indicator values
            indicator_df_1T = indicator_df_shifted.resample('1min').ffill()

            # Drop any NaN values that may result from shifting
            indicator_df_1T.dropna(inplace=True)

            # Generate filename
            filename = create_filename(indicator_name, timeframe, params)
            filepath = os.path.join(output_dir, filename)

            # Save as Feather with Zstandard compression for efficiency
            indicator_df_1T.to_feather(filepath, compression='zstd')

            # If you prefer Parquet, uncomment the following line and comment out the Feather line:
            # indicator_df_1T.to_parquet(filepath, engine='pyarrow', compression='snappy')

        except Exception as e:
            print(f"Error processing {indicator_name} with params {params} on {timeframe}: {e}")


def main():
    # Configuration
    indicators_config_file = 'indicators_config.json'  # JSON file containing indicators and parameters
    output_dir = 'precalculated_indicators_parquet'
    os.makedirs(output_dir, exist_ok=True)

    # Load indicators configuration
    indicators = load_indicator_config(indicators_config_file)

    # Define timeframes
    timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']

    # Filter out indicators with float parameters
    filtered_indicators = {name: params for name, params in indicators.items() if not has_float_params(params)}

    if not filtered_indicators:
        print("No indicators to process after filtering out those with float parameters.")
        return

    # Initialize DataLoader and load data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Assign the loaded data to a global variable for sharing
    global shared_tick_data
    shared_tick_data = data_loader.tick_data

    # Generate all tasks: (indicator_name, params)
    tasks = []
    for indicator_name, params in filtered_indicators.items():
        param_combinations = generate_parameter_combinations(params, step=1)  # Adjust step as needed
        for param in param_combinations:
            tasks.append((indicator_name, param))

    if not tasks:
        print("No tasks generated. Check your indicators and parameter ranges.")
        return

    # Define the number of processes
    num_processes = 50  # Leave one CPU free
    print(f"Using {num_processes} processes for multiprocessing.")

    # Create a partial function fixing output_dir and timeframes
    worker = partial(calculate_and_save_all_timeframes, output_dir=output_dir, timeframes=timeframes)

    # Initialize the Pool with the worker initializer
    with Pool(processes=num_processes, initializer=init_worker, initargs=(shared_tick_data,)) as pool:
        # Use tqdm for progress bar
        list(tqdm(pool.imap_unordered(worker, tasks, chunksize=10), total=len(tasks)))

    print("Precalculation of indicators completed.")


if __name__ == "__main__":
    main()
