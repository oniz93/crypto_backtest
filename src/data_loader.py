"""
data_loader.py
--------------
This module is responsible for loading and processing market data.
It loads tick data from Parquet files, resamples data to different timeframes,
calculates technical indicators, and filters data by date using the Strategy pattern.
"""

import os
from typing import Dict, Union
import numpy as np
import time
import logging
from datetime import datetime, date
import sys
import pandas as pd

from src.config_loader import Config
from src.dataframe_strategy import get_dataframe_strategy

# Set up module-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
pipe_path = '/tmp/genetic_optimizer_logpipe'
if not os.path.exists(pipe_path):
    os.mkfifo(pipe_path)
try:
    pipe = open(pipe_path, 'w')
    pipe_handler = logging.StreamHandler(pipe)
    pipe_handler.setLevel(logging.DEBUG)
    pipe_handler.setFormatter(formatter)
    logger.addHandler(pipe_handler)
except Exception as e:
    logger.error(f"Failed to open named pipe {pipe_path}: {e}")

class DataLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path=config_path)
        self.df_strategy = get_dataframe_strategy()
        self.timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D']
        self.base_timeframe = '1T'
        self.data_folder = 'output_parquet/'
        self.data_dir = 'data'
        self.tick_data: Dict[str, any] = {}
        self.support_resistance: Dict[str, any] = {}
        os.makedirs(self.data_dir, exist_ok=True)

    def import_ticks(self):
        """
        Import tick data from a Parquet file for the base timeframe.
        Converts the 'timestamp' column to datetime, sets it as the index,
        sorts the data, and filters it based on dates from the configuration.
        """
        tick_file = os.path.join(self.data_folder, 'BTCUSDT-tick-1min.parquet')
        tick_data_1m = self.df_strategy.load_parquet(tick_file)
        if 'timestamp' in tick_data_1m.columns:
            # Handle timestamp conversion based on the dataframe type
            try:
                import cudf
                if isinstance(tick_data_1m, cudf.DataFrame):
                    # cuDF has built-in datetime conversion when reading from parquet
                    pass
                else:
                    # Use pandas for non-cuDF dataframes
                    import pandas as pd_lib
                    tick_data_1m['timestamp'] = pd_lib.to_datetime(tick_data_1m['timestamp'])
            except ImportError:
                # Fall back to pandas if cuDF isn't available
                import pandas as pd_lib
                tick_data_1m['timestamp'] = pd_lib.to_datetime(tick_data_1m['timestamp'])
        else:
            raise ValueError("The 'timestamp' column is not present in the data.")
        tick_data_1m.set_index('timestamp', inplace=True)
        tick_data_1m.sort_index(inplace=True)
        tick_data_1m = self.filter_data_by_date(tick_data_1m, self.config.get('start_cutoff'), self.config.get('end_cutoff'))
        self.tick_data[self.base_timeframe] = tick_data_1m

    def resample_data(self):
        """Resample data to different timeframes from the base timeframe."""
        base_data = self.tick_data[self.base_timeframe]
        for tf in self.timeframes:
            if tf == self.base_timeframe:
                continue
            self.tick_data[tf] = self.df_strategy.resample(base_data, tf)

    def calculate_support_resistance(self):
        """Calculate support and resistance levels for each timeframe."""
        for tf in self.timeframes:
            data = self.tick_data[tf].copy()
            window = 5
            data['rolling_max'] = data['high'].rolling(window=window, center=True).max()
            data['rolling_min'] = data['low'].rolling(window=window, center=True).min()
            data['resistance'] = np.where(
                (data['high'] == data['rolling_max']) &
                (data['high'].shift(1) < data['high']) &
                (data['high'].shift(-1) < data['high']),
                data['high'], np.nan)
            data['support'] = np.where(
                (data['low'] == data['rolling_min']) &
                (data['low'].shift(1) > data['low']) &
                (data['low'].shift(-1) > data['low']),
                data['low'], np.nan)
            data.drop(columns=['rolling_max', 'rolling_min'], inplace=True)
            self.support_resistance[tf] = data[['support', 'resistance']]

    def calculate_indicator(self, indicator_name: str, params: dict, timeframe: str):
        """Calculate a technical indicator using the strategy."""
        timing_enabled = self.config.get("timing", False)
        t_start = time.time() if timing_enabled else None
        data = self.tick_data[timeframe].copy()
        logger.info(f"Calculating indicator: {indicator_name}, Timeframe: {timeframe}, DataFrame Type: {type(data)}")
        result = self.df_strategy.calculate_indicator(indicator_name, data, params, timeframe)
        if timing_enabled:
            elapsed = time.time() - t_start
            logger.info(f"Indicator {indicator_name} with params {params} on timeframe {timeframe} computed in {elapsed:.4f} seconds.")
        return result

    def filter_data_by_date(self, df, start_date: Union[str, datetime, date], end_date: Union[str, datetime, date]):
        """Filter a DataFrame by date range using the strategy."""
        if isinstance(start_date, (str, date)):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, (str, date)):
            end_date = pd.to_datetime(end_date)
        return self.df_strategy.filter_by_date(df, start_date, end_date)