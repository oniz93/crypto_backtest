# src/data_loader.py

import pandas as pd
import os
import numpy as np
import pandas_ta as ta
from typing import Dict, Any


class DataLoader:
    def __init__(self):
        self.tick_data = {}
        self.timeframes = [
            '1T', '5T', '15T', '30T', '45T', '1H', '2H', '4H', '8H', '12H', '1D', '1W'
        ]
        self.base_timeframe = '1T'  # Base timeframe is 1 minute
        self.data_folder = 'output_parquet/'
        self.cache = {}
        self.support_resistance = {}  # New attribute to store support and resistance levels

    def import_ticks(self):
        """
        Imports tick data and resamples to base timeframe.
        """
        # Load 1-minute tick data
        tick_data_1m = pd.read_parquet(
            os.path.join(self.data_folder, 'BTCUSDT-tick-1min.parquet'),
        )
        # Store base timeframe data
        self.tick_data[self.base_timeframe] = tick_data_1m

    def resample_data(self):
        """
        Resamples the base timeframe data to other specified timeframes.
        """
        base_data = self.tick_data[self.base_timeframe]
        for tf in self.timeframes:
            if tf == self.base_timeframe:
                continue  # Skip the base timeframe
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
            # Use the rolling window method to find local minima and maxima
            window = 5  # You can adjust the window size as needed
            data['rolling_max'] = data['high'].rolling(window=window, center=True).max()
            data['rolling_min'] = data['low'].rolling(window=window, center=True).min()
            # Identify resistance levels
            data['resistance'] = np.where(
                (data['high'] == data['rolling_max']) &
                (data['high'].shift(1) < data['high']) &
                (data['high'].shift(-1) < data['high']),
                data['high'],
                np.nan
            )
            # Identify support levels
            data['support'] = np.where(
                (data['low'] == data['rolling_min']) &
                (data['low'].shift(1) > data['low']) &
                (data['low'].shift(-1) > data['low']),
                data['low'],
                np.nan
            )
            # Drop intermediate columns
            data.drop(columns=['rolling_max', 'rolling_min'], inplace=True)
            # Store the support and resistance levels
            self.support_resistance[tf] = data[['support', 'resistance']]
            # Optional: Save to CSV files
            data[['support', 'resistance']].to_csv(f"suppres/support_resistance_{tf}.csv")

    def calculate_indicator(self, indicator_name: str, params: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """
        Calculates a single indicator on the provided data.
        """
        data = self.tick_data[timeframe].copy()
        cache_key = (indicator_name, tuple(sorted(params.items())), timeframe)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if indicator_name == 'sma':
            length = int(params['length'])
            data[f'SMA_{length}'] = ta.sma(data['close'], length=length)
            result = data[[f'SMA_{length}']].dropna()
        elif indicator_name == 'ema':
            length = int(params['length'])
            data[f'EMA_{length}'] = ta.ema(data['close'], length=length)
            result = data[[f'EMA_{length}']].dropna()
        elif indicator_name == 'rsi':
            length = int(params['length'])
            data[f'RSI_{length}'] = ta.rsi(data['close'], length=length)
            result = data[[f'RSI_{length}']].dropna()
        elif indicator_name == 'macd':
            fast = int(params['fast'])
            slow = int(params['slow'])
            signal = int(params['signal'])
            macd_df = ta.macd(data['close'], fast=fast, slow=slow, signal=signal)
            result = macd_df.dropna()
        elif indicator_name == 'bbands':
            length = int(params['length'])
            std_dev = float(params['std_dev'])
            bbands_df = ta.bbands(data['close'], length=length, std=std_dev)
            result = bbands_df.dropna()
        elif indicator_name == 'atr':
            length = int(params['length'])
            data[f'ATR_{length}'] = ta.atr(data['high'], data['low'], data['close'], length=length)
            result = data[[f'ATR_{length}']].dropna()
        elif indicator_name == 'stoch':
            k = int(params['k'])
            d = int(params['d'])
            stoch_df = ta.stoch(data['high'], data['low'], data['close'], k=k, d=d)
            result = stoch_df.dropna()
        elif indicator_name == 'cci':
            length = int(params['length'])
            data[f'CCI_{length}'] = ta.cci(data['high'], data['low'], data['close'], length=length)
            result = data[[f'CCI_{length}']].dropna()
        elif indicator_name == 'adx':
            length = int(params['length'])
            data[f'ADX_{length}'] = ta.adx(data['high'], data['low'], data['close'], length=length)
            result = data[[f'ADX_{length}']].dropna()
        elif indicator_name == 'cmf':
            length = int(params['length'])
            data[f'CMF_{length}'] = ta.cmf(data['high'], data['low'], data['close'], data['volume'], length=length)
            result = data[[f'CMF_{length}']].dropna()
        elif indicator_name == 'mfi':
            length = int(params['length'])
            data[f'MFI_{length}'] = ta.mfi(data['high'], data['low'], data['close'], data['volume'], length=length)
            result = data[[f'MFI_{length}']].dropna()
        elif indicator_name == 'roc':
            length = int(params['length'])
            data[f'ROC_{length}'] = ta.roc(data['close'], length=length)
            result = data[[f'ROC_{length}']].dropna()
        elif indicator_name == 'willr':
            length = int(params['length'])
            data[f'WILLR_{length}'] = ta.willr(data['high'], data['low'], data['close'], length=length)
            result = data[[f'WILLR_{length}']].dropna()
        elif indicator_name == 'psar':
            acceleration = float(params['acceleration'])
            max_acceleration = float(params['max_acceleration'])
            data['PSAR'] = ta.psar(data['high'], data['low'], acceleration=acceleration, maximum=max_acceleration)
            result = data[['PSAR']].dropna()
        elif indicator_name == 'ichimoku':
            tenkan = int(params['tenkan'])
            kijun = int(params['kijun'])
            senkou = int(params['senkou'])
            ichimoku_df = ta.ichimoku(data['high'], data['low'], tenkan=tenkan, kijun=kijun, senkou=senkou)
            result = ichimoku_df.dropna()
        elif indicator_name == 'keltner':
            length = int(params['length'])
            multiplier = float(params['multiplier'])
            keltner_df = ta.kc(data['high'], data['low'], data['close'], length=length, scalar=multiplier)
            result = keltner_df.dropna()
        elif indicator_name == 'donchian':
            lower_length = int(params['lower_length'])
            upper_length = int(params['upper_length'])
            donchian_df = ta.donchian(data['high'], data['low'], lower_length=lower_length, upper_length=upper_length)
            result = donchian_df.dropna()
        elif indicator_name == 'emv':
            length = int(params['length'])
            emv_df = ta.eom(data['high'], data['low'], data['volume'], length=length)
            result = emv_df.dropna()
        elif indicator_name == 'force':
            length = int(params['length'])
            data[f'FORCE_{length}'] = ta.efi(data['close'], data['volume'], length=length)
            result = data[[f'FORCE_{length}']].dropna()
        elif indicator_name == 'uo':
            short = int(params['short'])
            medium = int(params['medium'])
            long = int(params['long'])
            data['UO'] = ta.uo(data['high'], data['low'], data['close'], fast=short, medium=medium, slow=long)
            result = data[['UO']].dropna()
        elif indicator_name == 'volatility':
            length = int(params['length'])
            data[f'STDDEV_{length}'] = ta.stdev(data['close'], length=length)
            result = data[[f'STDDEV_{length}']].dropna()
        elif indicator_name == 'dpo':
            length = int(params['length'])
            data['DPO'] = ta.dpo(data['close'], length=length)
            result = data[['DPO']].dropna()
        elif indicator_name == 'trix':
            length = int(params['length'])
            data['TRIX'] = ta.trix(data['close'], length=length)
            result = data[['TRIX']].dropna()
        elif indicator_name == 'chaikin_osc':
            fast = int(params['fast'])
            slow = int(params['slow'])
            data['Chaikin_Osc'] = ta.cmf(data['high'], data['low'], data['close'], data['volume'], fast=fast, slow=slow)
            result = data[['Chaikin_Osc']].dropna()
        else:
            raise ValueError(f"Indicator {indicator_name} is not implemented.")
        self.cache[cache_key] = result
        return result

    def clear_variables(self):
        self.tick_data = {}
        self.cache = {}
