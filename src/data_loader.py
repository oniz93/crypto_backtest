import os
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

from numba import njit

from src.config_loader import Config
from src.utils import normalize_price_vec, normalize_volume_vec, normalize_diff_vec, normalize_rsi_vec
import time
import logging

# Use the module-level logger
logger = logging.getLogger(__name__)

@njit
def compute_cumulative_volume_profile_numba(close, volume, bins, width):
    """
    Numba-optimized function to compute cumulative volume profile.
    """
    n = len(close)
    cumulative_volume = np.zeros((n, width))
    for i in range(n):
        # Assign bin
        bin_idx = np.searchsorted(bins, close[i], side='right') - 1
        if bin_idx == width:
            bin_idx = width - 1
        elif bin_idx < 0:
            bin_idx = 0
        # Update cumulative volume
        if i > 0:
            cumulative_volume[i] = cumulative_volume[i - 1]
        cumulative_volume[i, bin_idx] += volume[i]
    return cumulative_volume

def compute_cumulative_volume_profile_numba_wrapper(data, n_clusters=100):
    """
    Wrapper function to compute cumulative volume profile using Numba.
    """
    data = data.sort_index()
    close = data['close'].values
    volume = data['volume'].values
    min_price = min(close)
    max_price = max(close)
    bins = np.linspace(min_price, max_price, n_clusters + 1)
    cumulative_volume = compute_cumulative_volume_profile_numba(close, volume, bins, n_clusters)
    cluster_columns = [f'cluster_{i}' for i in range(n_clusters)]
    volume_profile_df = pd.DataFrame(cumulative_volume, columns=cluster_columns, index=data.index)
    volume_profile_df = volume_profile_df.round(2)
    return volume_profile_df

def incremental_vpvr_fixed_bins(df: pd.DataFrame, width: int = 100, n_rows: int = None, bins_array: np.ndarray = None) -> pd.DataFrame:
    """
    A single-pass O(N) approach to incremental volume-by-price.
    """
    df = df.sort_index()
    if n_rows is not None:
        df = df.iloc[:n_rows]
    closes = df['close'].astype(float).values
    volumes = df['volume'].astype(float).values
    n = len(df)
    if n == 0:
        return pd.DataFrame([], columns=[f"cluster_{c}" for c in range(width)])
    if n == 1:
        out = np.zeros((1, width), dtype=np.float64)
        out[0, 0] = volumes[0]
        return pd.DataFrame(out, index=df.index[:1], columns=[f"cluster_{c}" for c in range(width)])
    if bins_array is None:
        min_price = min(closes)
        max_price = max(closes)
        if min_price == max_price:
            out = np.zeros((n, width), dtype=np.float64)
            out[0, 0] = volumes[0]
            for i in range(1, n):
                out[i] = out[i - 1]
                out[i, 0] += volumes[i]
            cluster_cols = [f"cluster_{c}" for c in range(width)]
            return pd.DataFrame(out, columns=cluster_cols, index=df.index)
        bins_array = np.linspace(min_price, max_price, width + 1)
    width = len(bins_array) - 1
    cumulative_dist = np.zeros(width, dtype=np.float64)
    out = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        c_i = closes[i]
        v_i = volumes[i]
        bin_idx = np.searchsorted(bins_array, c_i, side='right') - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= width:
            bin_idx = width - 1
        cumulative_dist[bin_idx] += v_i
        out[i] = cumulative_dist
    cluster_cols = [f"cluster_{c}" for c in range(width)]
    result_df = pd.DataFrame(out, columns=cluster_cols, index=df.index)
    result_df = result_df.round(2)
    return result_df

class DataLoader:
    def __init__(self):
        self.tick_data = {}
        self.config = Config()  # Added so we can read config parameters (e.g. timing)
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
        self.base_timeframe = '1min'
        self.data_folder = 'output_parquet/'
        self.cache = {}
        self.support_resistance = {}
        self.funding_rates = {}

    def import_ticks(self):
        tick_data_1m = pd.read_parquet(os.path.join(self.data_folder, 'BTCUSDT-tick-1min.parquet'))
        if 'timestamp' in tick_data_1m.columns:
            tick_data_1m['timestamp'] = pd.to_datetime(tick_data_1m['timestamp'])
        else:
            raise ValueError("The 'timestamp' column is not present in the data.")
        tick_data_1m.set_index('timestamp', inplace=True)
        tick_data_1m.sort_index(inplace=True)
        tick_data_1m = self.filter_data_by_date(tick_data_1m, self.config.get('start_cutoff'), self.config.get('end_cutoff'))
        self.tick_data[self.base_timeframe] = tick_data_1m

    def resample_data(self):
        base_data = self.tick_data[self.base_timeframe]
        for tf in self.timeframes:
            if tf == self.base_timeframe:
                continue
            resampled_data = base_data.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            self.tick_data[tf] = resampled_data

    def calculate_support_resistance(self):
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

    def calculate_indicator(self, indicator_name: str, params: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """
        Calculates a single indicator on the provided data.
        Added timing logs if config["timing"] is True.
        """
        # Check whether timing is enabled via config
        timing_enabled = self.config.get("timing", False)
        if timing_enabled:
            t_start = time.time()

        data = self.tick_data[timeframe].copy()

        if indicator_name == 'sma':
            length = int(params['length'])
            data[f'SMA_{length}'] = ta.sma(data['close'], length=length).astype(float)
            data[f'SMA_{length}'] = normalize_price_vec(data[f'SMA_{length}'])
            result = data[[f'SMA_{length}']].dropna()
        elif indicator_name == 'ema':
            length = int(params['length'])
            data[f'EMA_{length}'] = ta.ema(data['close'], length=length).astype(float)
            data[f'EMA_{length}'] = normalize_price_vec(data[f'EMA_{length}'])
            result = data[[f'EMA_{length}']].dropna()
        elif indicator_name == 'rsi':
            length = int(params['length'])
            data[f'RSI_{length}'] = ta.rsi(data['close'], length=length).astype(float)
            data[f'RSI_{length}'] = normalize_rsi_vec(data[f'RSI_{length}'])
            result = data[[f'RSI_{length}']].dropna()
        elif indicator_name == 'macd':
            fast = int(params['fast'])
            slow = int(params['slow'])
            signal = int(params['signal'])
            macd_df = ta.macd(data['close'], fast=fast, slow=slow, signal=signal).astype(float)
            for col in macd_df.columns:
                macd_df[col] = normalize_diff_vec(macd_df[col])
            result = macd_df.dropna()
        elif indicator_name == 'bbands':
            length = int(params['length'])
            std_dev = float(params['std_dev'])
            bbands_df = ta.bbands(data['close'], length=length, std=std_dev).astype(float)
            result = bbands_df.dropna()
        elif indicator_name == 'atr':
            length = int(params['length'])
            data[f'ATR_{length}'] = ta.atr(data['high'], data['low'], data['close'], length=length).astype(float)
            data[f'ATR_{length}'] = normalize_diff_vec(data[f'ATR_{length}'])
            result = data[[f'ATR_{length}']].dropna()
        elif indicator_name == 'stoch':
            k = int(params['k'])
            d = int(params['d'])
            stoch_df = ta.stoch(data['high'], data['low'], data['close'], k=k, d=d).astype(float)
            result = stoch_df.dropna()
        elif indicator_name == 'cci':
            length = int(params['length'])
            data[f'CCI_{length}'] = ta.cci(data['high'], data['low'], data['close'], length=length).astype(float)
            result = data[[f'CCI_{length}']].dropna()
        elif indicator_name == 'adx':
            length = int(params['length'])
            adx_df = ta.adx(data['high'], data['low'], data['close'], length=length).astype(float)
            data[f'ADX_{length}'] = adx_df[f'ADX_{length}']
            result = data[[f'ADX_{length}']].dropna()
        elif indicator_name == 'cmf':
            length = int(params['length'])
            data[f'CMF_{length}'] = ta.cmf(data['high'], data['low'], data['close'], data['volume'], length=length).astype(float)
            result = data[[f'CMF_{length}']].dropna()
        elif indicator_name == 'mfi':
            length = int(params['length'])
            data[f'MFI_{length}'] = ta.mfi(data['high'], data['low'], data['close'], data['volume'], length=length).astype(float)
            result = data[[f'MFI_{length}']].dropna()
        elif indicator_name == 'roc':
            length = int(params['length'])
            data[f'ROC_{length}'] = ta.roc(data['close'], length=length).astype(float)
            result = data[[f'ROC_{length}']].dropna()
        elif indicator_name == 'willr':
            length = int(params['length'])
            data[f'WILLR_{length}'] = ta.willr(data['high'], data['low'], data['close'], length=length).astype(float)
            result = data[[f'WILLR_{length}']].dropna()
        elif indicator_name == 'psar':
            acceleration = float(params['acceleration'])
            max_acceleration = float(params['max_acceleration'])
            psar_df = ta.psar(data['high'], data['low'], close=data['close'], af=acceleration, max_af=max_acceleration).astype(float)
            data = data.join(psar_df)
            result = data.dropna()
        elif indicator_name == 'ichimoku':
            tenkan = int(params['tenkan'])
            kijun = int(params['kijun'])
            senkou = int(params['senkou'])
            ichimoku_df, _ = ta.ichimoku(data['high'], data['low'], close=data['close'], tenkan=tenkan, kijun=kijun, senkou=senkou)
            data = data.join(ichimoku_df.astype(float))
            result = data.dropna()
        elif indicator_name == 'keltner':
            length = int(params['length'])
            multiplier = float(params['multiplier'])
            keltner_df = ta.kc(data['high'], data['low'], data['close'], length=length, scalar=multiplier).astype(float)
            result = keltner_df.dropna()
        elif indicator_name == 'donchian':
            lower_length = int(params['lower_length'])
            upper_length = int(params['upper_length'])
            donchian_df = ta.donchian(data['high'], data['low'], lower_length=lower_length, upper_length=upper_length).astype(float)
            result = donchian_df.dropna()
        elif indicator_name == 'emv':
            length = int(params['length'])
            emv_df = ta.eom(data['high'], data['low'], close=data['close'], volume=data['volume'], length=length).astype(float)
            result = emv_df.dropna()
        elif indicator_name == 'force':
            length = int(params['length'])
            data[f'FORCE_{length}'] = ta.efi(data['close'], data['volume'], length=length).astype(float)
            result = data[[f'FORCE_{length}']].dropna()
        elif indicator_name == 'uo':
            short = int(params['short'])
            medium = int(params['medium'])
            long = int(params['long'])
            data['UO'] = ta.uo(data['high'], data['low'], data['close'], fast=short, medium=medium, slow=long).astype(float)
            result = data[['UO']].dropna()
        elif indicator_name == 'volatility':
            length = int(params['length'])
            data[f'STDDEV_{length}'] = ta.stdev(data['close'], length=length).astype(float)
            result = data[[f'STDDEV_{length}']].dropna()
        elif indicator_name == 'dpo':
            length = int(params['length'])
            data['DPO'] = ta.dpo(data['close'], length=length).astype(float)
            result = data[['DPO']].dropna()
        elif indicator_name == 'trix':
            length = int(params['length'])
            data['TRIX'] = ta.trix(data['close'], length=length).iloc[:, 0].astype(float)
            result = data[['TRIX']].dropna()
        elif indicator_name == 'chaikin_osc':
            fast = int(params['fast'])
            slow = int(params['slow'])
            data['Chaikin_Osc'] = ta.adosc(data['high'], data['low'], data['close'], data['volume'], fast=fast, slow=slow).astype(float)
            result = data[['Chaikin_Osc']].dropna()
        elif indicator_name == 'vwap':
            offset = int(params['offset'])
            data['VWAP'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'], offset=offset).astype(float)
            data['VWAP'] = normalize_diff_vec(data['VWAP'])
            result = data[['VWAP']].dropna()
            result = result.sub(data['close'], axis=0)
        elif indicator_name == 'vwma':
            length = int(params['length'])
            data['VWMA'] = ta.vwma(data['close'], data['volume'], length=length).astype(float)
            data['VWMA'] = normalize_diff_vec(data['VWMA'])
            result = data[['VWMA']].dropna()
            result = result.sub(data['close'], axis=0)
        elif indicator_name == 'vpvr':
            width = int(params['width'])
            n_clusters = 100
            cluster_columns = [f'cluster_{i}' for i in range(n_clusters)]
            clusters_df = pd.DataFrame(0.0, index=data.index, columns=cluster_columns)
            data = data.join(clusters_df)
            volume_profile_df = incremental_vpvr_fixed_bins(data, width=n_clusters)
            new_clusters = volume_profile_df.copy()
            for col in cluster_columns:
                new_clusters[col] = normalize_volume_vec(new_clusters[col].values)
            data = data.drop(columns=cluster_columns, errors='ignore').join(new_clusters)
            for col in cluster_columns:
                data[col] = normalize_volume_vec(data[col])
            result = data[cluster_columns].dropna()
        else:
            raise ValueError(f"Indicator {indicator_name} is not implemented.")

        if timing_enabled:
            elapsed = time.time() - t_start
            logger.info(f"Indicator {indicator_name} with params {params} on timeframe {timeframe} computed in {elapsed:.4f} seconds.")
        return result

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
