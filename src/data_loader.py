# src/data_loader.py

import os
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

from src.config_loader import Config


class DataLoader:
    def __init__(self):
        self.tick_data = {}
        # Update timeframes to new strings:
        self.timeframes = [
            '1min', '5min', '15min', '30min', '1h', '4h', '1d'
        ]
        self.base_timeframe = '1min'  # Base timeframe is 1 minute
        self.data_folder = 'output_parquet/'
        self.cache = {}
        self.support_resistance = {}
        self.funding_rates = {}

    def import_ticks(self):
        """
        Imports tick data and sets it to the base timeframe.
        """
        tick_data_1m = pd.read_parquet(
            os.path.join(self.data_folder, 'BTCUSDT-tick-1min.parquet'),
        )
        if 'timestamp' in tick_data_1m.columns:
            tick_data_1m['timestamp'] = pd.to_datetime(tick_data_1m['timestamp'])
        else:
            raise ValueError("The 'timestamp' column is not present in the data.")

        tick_data_1m.set_index('timestamp', inplace=True)
        tick_data_1m.sort_index(inplace=True)

        config = Config()
        tick_data_1m = self.filter_data_by_date(tick_data_1m, config.get('start_cutoff'), config.get('end_cutoff'))
        self.tick_data[self.base_timeframe] = tick_data_1m

    def resample_data(self):
        """
        Resamples the base timeframe data to other specified timeframes.
        """
        base_data = self.tick_data[self.base_timeframe]
        for tf in self.timeframes:
            if tf == self.base_timeframe:
                continue
            # For example, tf = '5min', '15min', '1h', etc.
            resampled_data = base_data.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            self.tick_data[tf] = resampled_data

    def calculate_support_resistance(self):
        """
        Calculates support and resistance levels for each timeframe.
        """
        for tf in self.timeframes:
            data = self.tick_data[tf].copy()
            window = 5
            data['rolling_max'] = data['high'].rolling(window=window, center=True).max()
            data['rolling_min'] = data['low'].rolling(window=window, center=True).min()

            data['resistance'] = np.where(
                (data['high'] == data['rolling_max']) &
                (data['high'].shift(1) < data['high']) &
                (data['high'].shift(-1) < data['high']),
                data['high'],
                np.nan
            )

            data['support'] = np.where(
                (data['low'] == data['rolling_min']) &
                (data['low'].shift(1) > data['low']) &
                (data['low'].shift(-1) > data['low']),
                data['low'],
                np.nan
            )

            data.drop(columns=['rolling_max', 'rolling_min'], inplace=True)
            self.support_resistance[tf] = data[['support', 'resistance']]

    def calculate_indicator(self, indicator_name: str, params: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """
        Calculates a single indicator on the provided data.
        """
        data = self.tick_data[timeframe].copy()
        # Same logic for indicators as before
        # Example for sma:
        if indicator_name == 'sma':
            length = int(params['length'])
            data[f'SMA_{length}'] = ta.sma(data['close'], length=length).astype(float)
            result = data[[f'SMA_{length}']].dropna()
        elif indicator_name == 'ema':
            length = int(params['length'])
            data[f'EMA_{length}'] = ta.ema(data['close'], length=length).astype(float)
            result = data[[f'EMA_{length}']].dropna()
        elif indicator_name == 'rsi':
            length = int(params['length'])
            data[f'RSI_{length}'] = ta.rsi(data['close'], length=length).astype(float)
            result = data[[f'RSI_{length}']].dropna()
        elif indicator_name == 'macd':
            fast = int(params['fast'])
            slow = int(params['slow'])
            signal = int(params['signal'])
            macd_df = ta.macd(data['close'], fast=fast, slow=slow, signal=signal).astype(float)
            result = macd_df.dropna()
        elif indicator_name == 'atr':
            length = int(params['length'])
            data[f'ATR_{length}'] = ta.atr(data['high'], data['low'], data['close'], length=length).astype(float)
            result = data[[f'ATR_{length}']].dropna()
        elif indicator_name == 'stoch':
            k = int(params['k'])
            d = int(params['d'])
            stoch_df = ta.stoch(data['high'], data['low'], data['close'], k=k, d=d).astype(float)
            result = stoch_df.dropna()
        else:
            raise ValueError(f"Indicator {indicator_name} not implemented here.")

        return result

    def clear_variables(self):
        self.tick_data = {}
        self.support_resistance = {}
        self.funding_rates = {}
        self.cache = {}

    def filter_data_by_date(self, df: DataFrame, start_date: Union[str, pd.Timestamp], end_date: Union[str, pd.Timestamp]):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
            except Exception as e:
                print(f"Error converting index to DatetimeIndex: {e}")
                return df

        filtered_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        return filtered_df
