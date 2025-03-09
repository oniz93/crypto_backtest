"""
precalculate_indicators.py
--------------------------
This script precalculates technical indicators for different timeframes
and saves the results to Feather (or optionally Parquet) files.
It uses multiprocessing to parallelize the computation.
"""

import json
import os
from functools import partial
from itertools import product
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from src.data_loader import DataLoader

# Global variable for sharing tick data among worker processes.
shared_tick_data = None

def init_worker(tick_data):
    """
    Worker initializer for multiprocessing.

    Parameters:
        tick_data: Shared tick data (a dictionary of DataFrames).
    """
    global shared_tick_data
    shared_tick_data = tick_data

def load_indicator_config(config_file: str) -> dict:
    """
    Load indicator configuration from a JSON file.

    Parameters:
        config_file (str): Path to the JSON file.

    Returns:
        dict: Indicator configuration.
    """
    with open(config_file, 'r') as f:
        indicators = json.load(f)
    return indicators

def has_float_params(params: dict) -> bool:
    """
    Check if any parameter range in the indicator configuration contains float values.

    Parameters:
        params (dict): Indicator parameters.

    Returns:
        bool: True if any parameter is a float, else False.
    """
    for param_range in params.values():
        if isinstance(param_range, list):
            if any(isinstance(val, float) for val in param_range):
                return True
    return False

def generate_parameter_combinations(params: dict, step: int = 5) -> list:
    """
    Generate all combinations of indicator parameters given integer ranges.

    Parameters:
        params (dict): Dictionary of indicator parameter ranges.
        step (int): Step size for generating values.

    Returns:
        list: List of parameter dictionaries.
    """
    param_names = []
    param_values = []
    for param, range_vals in params.items():
        if isinstance(range_vals, list) and len(range_vals) == 2:
            low, high = range_vals
            if isinstance(low, int) and isinstance(high, int):
                values = list(range(low, high + 1, step))
            else:
                values = []
        else:
            values = []
        if values:
            param_names.append(param)
            param_values.append(values)
    # Create all possible combinations.
    combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]
    return combinations

def create_filename(indicator_name: str, timeframe: str, params: dict) -> str:
    """
    Create a filename for the precalculated indicator file.

    Parameters:
        indicator_name (str): Name of the indicator.
        timeframe (str): Timeframe (e.g., '1min').
        params (dict): Parameters used for calculation.

    Returns:
        str: A filename string.
    """
    if not params:
        filename = f"{indicator_name}-{timeframe}.feather"
    else:
        param_parts = [f"{key}-{value}" for key, value in params.items()]
        filename = f"{indicator_name}-{timeframe}-" + "-".join(param_parts) + ".feather"
    return filename.replace(" ", "_")

def calculate_and_save_all_timeframes(indicator_info: tuple, output_dir: str, timeframes: list):
    """
    Calculate an indicator for each timeframe and save the result to a file.

    Parameters:
        indicator_info (tuple): (indicator_name, params)
        output_dir (str): Directory to save the files.
        timeframes (list): List of timeframe strings.
    """
    indicator_name, params = indicator_info
    for timeframe in timeframes:
        try:
            # Initialize a new DataLoader and use the shared tick data.
            data_loader = DataLoader()
            data_loader.tick_data = shared_tick_data  # Use the global shared tick data.
            data_loader.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
            # Calculate the indicator.
            indicator_df = data_loader.calculate_indicator(indicator_name, params, timeframe)
            if indicator_df.empty:
                print(f"Empty DataFrame for {indicator_name} with params {params} on {timeframe}. Skipping.")
                continue
            # Shift the indicator data by the timeframe duration.
            shift_duration = pd.to_timedelta(timeframe)
            indicator_df_shifted = indicator_df.copy()
            indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
            # Resample to 1-minute intervals and forward-fill missing data.
            indicator_df_1T = indicator_df_shifted.resample('1min').ffill()
            indicator_df_1T.dropna(inplace=True)
            filename = create_filename(indicator_name, timeframe, params)
            filepath = os.path.join(output_dir, filename)
            # Save the indicator DataFrame to a Feather file.
            indicator_df_1T.to_feather(filepath, compression='zstd')
        except Exception as e:
            print(f"Error processing {indicator_name} with params {params} on {timeframe}: {e}")

def main():
    """
    Main function to precalculate technical indicators.
    """
    indicators_config_file = 'indicators_config.json'
    output_dir = 'precalculated_indicators_parquet'
    os.makedirs(output_dir, exist_ok=True)
    indicators = load_indicator_config(indicators_config_file)
    timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
    # Filter out indicators with float parameters.
    filtered_indicators = {name: params for name, params in indicators.items() if not has_float_params(params)}
    if not filtered_indicators:
        print("No indicators to process after filtering out those with float parameters.")
        return
    # Load tick data.
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()
    global shared_tick_data
    shared_tick_data = data_loader.tick_data
    tasks = []
    # Generate tasks for each indicator and each parameter combination.
    for indicator_name, params in filtered_indicators.items():
        param_combinations = generate_parameter_combinations(params, step=1)
        for param in param_combinations:
            tasks.append((indicator_name, param))
    if not tasks:
        print("No tasks generated. Check your indicators and parameter ranges.")
        return
    num_processes = 50
    print(f"Using {num_processes} processes for multiprocessing.")
    from functools import partial
    worker = partial(calculate_and_save_all_timeframes, output_dir=output_dir, timeframes=timeframes)
    with Pool(processes=num_processes, initializer=init_worker, initargs=(shared_tick_data,)) as pool:
        from tqdm import tqdm
        # Use a progress bar to track tasks.
        list(tqdm(pool.imap_unordered(worker, tasks, chunksize=10), total=len(tasks)))
    print("Precalculation of indicators completed.")

if __name__ == "__main__":
    main()
