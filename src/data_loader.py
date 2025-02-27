"""
data_loader.py
--------------
This module is responsible for loading and processing market data.
It loads tick data from Parquet files, resamples data to different timeframes,
calculates technical indicators, and filters data by date.
It also provides functions to compute volume profiles using Numba for speed.
It now supports both cuDF (when CUDA is available) and standard pandas.
"""

import os
from typing import Dict, Any, Union
import numpy as np
import time
import logging
from numba import njit
from datetime import date

# -----------------------------------------------------------------------------
# CONDITIONAL DATAFRAME LIBRARY SELECTION:
# If CUDA is available, use cuDF for GPU-accelerated dataframe operations.
# Otherwise, fallback to standard pandas.
# -----------------------------------------------------------------------------
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        import cudf as pd
        USING_CUDF = True
        NUM_GPU = pd.cuda.get_device_count()
    else:
        import pandas as pd
        USING_CUDF = False
        NUM_GPU = 0
except Exception:
    import pandas as pd
    USING_CUDF = False

# -----------------------------------------------------------------------------
# Remove pandas_ta import and define custom indicator functions.
# These functions mimic common technical analysis indicators using
# native (cu)DF operations and built-in rolling/ewm functions.
# -----------------------------------------------------------------------------

def sma(series, length):
    """Simple Moving Average using rolling mean."""
    return series.rolling(window=length).mean()

def ema(series, length):
    """Exponential Moving Average using ewm mean."""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length):
    """Relative Strength Index calculation."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(window=length).mean()
    ma_down = down.rolling(window=length).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast, slow, signal):
    """MACD indicator calculation.
    
    Returns a DataFrame with three columns:
      - MACD line,
      - Signal line,
      - Histogram.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.concat([macd_line, signal_line, histogram], axis=1)

def bbands(series, length, std):
    """Bollinger Bands calculation.
    
    Returns a DataFrame with columns: Upper Band, Middle SMA, Lower Band.
    """
    sma_val = sma(series, length)
    rolling_std = series.rolling(window=length).std()
    upper = sma_val + std * rolling_std
    lower = sma_val - std * rolling_std
    return pd.concat([upper, sma_val, lower], axis=1)

def atr(high, low, close, length):
    """Average True Range calculation."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=length).mean()

def stoch(high, low, close, k, d):
    """Stochastic Oscillator calculation.
    
    Returns a DataFrame with %K and %D lines.
    """
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d).mean()
    return pd.concat([stoch_k, stoch_d], axis=1)

def cci(high, low, close, length):
    """Commodity Channel Index calculation."""
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, length)
    mean_deviation = typical_price.rolling(window=length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (typical_price - sma_tp) / (0.015 * mean_deviation)

def adx(high, low, close, length):
    """Average Directional Index calculation (simplified)."""
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = (-low.diff()).where(((-low.diff()) > up_move) & ((-low.diff()) > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window=length).mean()
    plus_di = 100 * (plus_dm.rolling(window=length).sum() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=length).sum() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=length).mean()

def cmf(high, low, close, volume, length):
    """Chaikin Money Flow calculation."""
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.rolling(window=length).sum() / volume.rolling(window=length).sum()

def mfi(high, low, close, volume, length):
    """Money Flow Index calculation."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    tp_diff = typical_price.diff()
    positive_flow = money_flow.where(tp_diff > 0, 0)
    negative_flow = money_flow.where(tp_diff < 0, 0)
    pos_mf = positive_flow.rolling(window=length).sum()
    neg_mf = negative_flow.rolling(window=length).sum().abs()
    mfr = pos_mf / neg_mf
    return 100 - (100 / (1 + mfr))

def roc(series, length):
    """Rate of Change calculation."""
    return ((series / series.shift(length)) - 1) * 100

def willr(high, low, close, length):
    """Williams %R calculation."""
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def psar(high, low, close, acceleration=0.02, max_acceleration=0.2):
    """Parabolic SAR calculation (simplified iterative algorithm)."""
    psar_series = close.copy()
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = acceleration
    ep = low.iloc[0]
    for i in range(1, len(close)):
        psar_series.iloc[i] = psar_series.iloc[i-1] + af * (ep - psar_series.iloc[i-1])
        if trend == 1:
            if low.iloc[i] < psar_series.iloc[i]:
                trend = -1
                psar_series.iloc[i] = ep
                af = acceleration
                ep = high.iloc[i]
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + acceleration, max_acceleration)
        else:
            if high.iloc[i] > psar_series.iloc[i]:
                trend = 1
                psar_series.iloc[i] = ep
                af = acceleration
                ep = low.iloc[i]
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + acceleration, max_acceleration)
    return psar_series.to_frame(name='PSAR')

def ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Ichimoku Cloud calculation (simplified)."""
    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2).shift(kijun)
    df = pd.concat([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b], axis=1)
    df.columns = ['tenkan', 'kijun', 'senkou_a', 'senkou_b']
    return df

def keltner(high, low, close, length, multiplier):
    """Keltner Channel calculation."""
    typical_price = (high + low + close) / 3
    ema_tp = ema(typical_price, length)
    atr_val = atr(high, low, close, length)
    upper = ema_tp + multiplier * atr_val
    lower = ema_tp - multiplier * atr_val
    return pd.concat([upper, ema_tp, lower], axis=1)

def donchian(high, low, lower_length, upper_length):
    """Donchian Channel calculation."""
    upper = high.rolling(window=upper_length).max()
    lower = low.rolling(window=lower_length).min()
    return pd.concat([upper, lower], axis=1)

def emv(high, low, close, volume, length):
    """Ease of Movement (EMV) calculation (simplified)."""
    bp = (high + low) / 2
    tr = high - low
    emv_raw = bp.diff() / tr.replace(0, np.nan)
    return emv_raw.rolling(window=length).mean()

def force(close, volume, length):
    """Force Index calculation."""
    return (close.diff(length) * volume).rolling(window=length).mean()

def uo(high, low, close, short, medium, long):
    """Ultimate Oscillator calculation (simplified)."""
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    avg_short = bp.rolling(window=short).sum() / tr.rolling(window=short).sum()
    avg_medium = bp.rolling(window=medium).sum() / tr.rolling(window=medium).sum()
    avg_long = bp.rolling(window=long).sum() / tr.rolling(window=long).sum()
    return 100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7

def volatility(series, length):
    """Volatility calculation as rolling standard deviation."""
    return series.rolling(window=length).std()

def dpo(series, length):
    """Detrended Price Oscillator calculation."""
    sma_val = sma(series, length)
    return series - sma_val.shift(int(length/2))

def trix(series, length):
    """TRIX indicator calculation."""
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100

def chaikin_osc(high, low, close, volume, fast, slow):
    """Chaikin Oscillator calculation."""
    ad = ((close - low) - (high - close)) / (high - low) * volume
    ema_fast = ema(ad, fast)
    ema_slow = ema(ad, slow)
    return ema_fast - ema_slow

def vwap(high, low, close, volume, offset=0):
    """Volume Weighted Average Price (VWAP) calculation."""
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical_price * volume).cumsum()
    vw = cum_tp_vol / cum_vol
    return vw.shift(offset)

def vwma(close, volume, length):
    """Volume Weighted Moving Average (VWMA) calculation."""
    tp_vol = close * volume
    sum_vol = volume.rolling(window=length).sum()
    return tp_vol.rolling(window=length).sum() / sum_vol

def obv(close, volume):
    """On-Balance Volume (OBV) calculation."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv_series = (volume * direction).cumsum()
    return obv_series

def hist_vol(close, length):
    """Historical Volatility calculation using log returns."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=length).std()

# -----------------------------------------------------------------------------
# END OF CUSTOM INDICATOR FUNCTIONS
# -----------------------------------------------------------------------------

from src.config_loader import Config
from src.utils import (normalize_price_vec, normalize_volume_vec,
                       normalize_diff_vec, normalize_rsi_vec)

# Set up module-level logging.
logger = logging.getLogger(__name__)

@njit
def compute_cumulative_volume_profile_numba(close, volume, bins, width):
    """
    Compute cumulative volume profile using Numba for speed.

    Parameters:
        close (np.ndarray): Array of closing prices.
        volume (np.ndarray): Array of volumes.
        bins (np.ndarray): Array of price bin edges.
        width (int): Number of bins.

    Returns:
        np.ndarray: 2D array where each row i contains cumulative volume per bin up to i.
    """
    n = len(close)
    cumulative_volume = np.zeros((n, width))
    for i in range(n):
        bin_idx = np.searchsorted(bins, close[i], side='right') - 1
        if bin_idx == width:
            bin_idx = width - 1
        elif bin_idx < 0:
            bin_idx = 0
        if i > 0:
            cumulative_volume[i] = cumulative_volume[i - 1]
        cumulative_volume[i, bin_idx] += volume[i]
    return cumulative_volume

def compute_cumulative_volume_profile_numba_wrapper(data, n_clusters=100):
    """
    Wrapper for computing cumulative volume profile.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'close' and 'volume' columns.
        n_clusters (int): Number of price bins/clusters.

    Returns:
        pd.DataFrame: DataFrame with cumulative volume per cluster.
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

def incremental_vpvr_fixed_bins(df: pd.DataFrame, width: int = 100,
                                n_rows: int = None, bins_array: np.ndarray = None) -> pd.DataFrame:
    """
    A one-pass algorithm to compute incremental volume-by-price.

    Parameters:
        df (pd.DataFrame): DataFrame with 'close' and 'volume' columns.
        width (int): Number of bins.
        n_rows (int, optional): Process only the first n_rows rows if provided.
        bins_array (np.ndarray, optional): Custom bin edges.

    Returns:
        pd.DataFrame: DataFrame with cumulative volume distribution per row.
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

from datetime import datetime, date
def filter_data_by_date(df: pd.DataFrame, start_date: Union[str, datetime, date],
                        end_date: Union[str, datetime, date]) -> pd.DataFrame:
    """
    Filter the DataFrame rows based on a date range.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        start_date, end_date: Date range (can be str, pd.Timestamp, or datetime.date).

    Returns:
        pd.DataFrame: DataFrame containing only rows within the specified range.
    """
    start_date = start_date.isoformat()
    end_date = end_date.isoformat()

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

class DataLoader:
    def __init__(self):
        """
        DataLoader class for importing and processing market tick data.

        Attributes:
            tick_data (dict): Dictionary to store data for different timeframes.
            config (Config): Configuration settings loaded from a YAML file.
            timeframes (list): List of timeframes for resampling.
            base_timeframe (str): The main timeframe.
            data_folder (str): Folder where data files are stored.
        """
        self.tick_data = {}
        self.config = Config()  # Load config parameters.
        # Differentiate timeframes based on the library used.
        self.timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D']
        self.base_timeframe = '1T'
        self.data_folder = 'output_parquet/'
        self.cache = {}
        self.support_resistance = {}
        self.funding_rates = {}

    def import_ticks(self):
        """
        Import tick data from a Parquet file for the base timeframe.
        Converts the 'timestamp' column to datetime, sets it as the index,
        sorts the data, and filters it based on dates from the configuration.
        """
        tick_file = os.path.join(self.data_folder, 'BTCUSDT-tick-1min.parquet')
        tick_data_1m = pd.read_parquet(tick_file)
        if 'timestamp' in tick_data_1m.columns:
            tick_data_1m['timestamp'] = pd.to_datetime(tick_data_1m['timestamp'])
        else:
            raise ValueError("The 'timestamp' column is not present in the data.")
        tick_data_1m.set_index('timestamp', inplace=True)
        tick_data_1m.sort_index(inplace=True)
        tick_data_1m = filter_data_by_date(tick_data_1m, self.config.get('start_cutoff'), self.config.get('end_cutoff'))
        self.tick_data[self.base_timeframe] = tick_data_1m

    def resample_data(self):
        """
        Resample the base tick data into different timeframes.
        For each timeframe (except the base timeframe), compute:
          - First open price.
          - Maximum high.
          - Minimum low.
          - Last close price.
          - Sum of volumes.
        """
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
        """
        Calculate support and resistance levels for each timeframe using a rolling window.
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
                data['high'], np.nan)
            data['support'] = np.where(
                (data['low'] == data['rolling_min']) &
                (data['low'].shift(1) > data['low']) &
                (data['low'].shift(-1) > data['low']),
                data['low'], np.nan)
            data.drop(columns=['rolling_max', 'rolling_min'], inplace=True)
            self.support_resistance[tf] = data[['support', 'resistance']]

    def calculate_indicator(self, indicator_name: str, params: dict, timeframe: str) -> pd.DataFrame:
        """
        Calculate a technical indicator on the tick data for a given timeframe.
        Timing can be enabled via config.

        Parameters:
            indicator_name (str): Name of the indicator (e.g., 'sma', 'ema').
            params (dict): Parameters for the indicator.
            timeframe (str): Timeframe on which to compute the indicator.

        Returns:
            pd.DataFrame: DataFrame containing the computed indicator.
        """
        timing_enabled = self.config.get("timing", False)
        if timing_enabled:
            t_start = time.time()
        data = self.tick_data[timeframe].copy()

        # Compute the indicator based on its name.
        if indicator_name == 'sma':
            length = int(params['length'])
            data[f'SMA_{length}'] = sma(data['close'], length=length).astype(float)
            data[f'SMA_{length}'] = normalize_price_vec(data[f'SMA_{length}'])
            result = data[[f'SMA_{length}']].dropna()

        elif indicator_name == 'ema':
            length = int(params['length'])
            data[f'EMA_{length}'] = ema(data['close'], length=length).astype(float)
            data[f'EMA_{length}'] = normalize_price_vec(data[f'EMA_{length}'])
            result = data[[f'EMA_{length}']].dropna()

        elif indicator_name == 'rsi':
            length = int(params['length'])
            data[f'RSI_{length}'] = rsi(data['close'], length=length).astype(float)
            data[f'RSI_{length}'] = normalize_rsi_vec(data[f'RSI_{length}'])
            result = data[[f'RSI_{length}']].dropna()

        elif indicator_name == 'macd':
            fast = int(params['fast'])
            slow = int(params['slow'])
            signal = int(params['signal'])
            macd_df = macd(data['close'], fast=fast, slow=slow, signal=signal).astype(float)
            for col in macd_df.columns:
                macd_df[col] = normalize_diff_vec(macd_df[col])
            result = macd_df.dropna()

        elif indicator_name == 'bbands':
            length = int(params['length'])
            std_dev = float(params['std_dev'])
            bbands_df = bbands(data['close'], length=length, std=std_dev).astype(float)
            for col in bbands_df.columns:
                bbands_df[col] = normalize_price_vec(bbands_df[col])
            result = bbands_df.dropna()

        elif indicator_name == 'atr':
            length = int(params['length'])
            data[f'ATR_{length}'] = atr(data['high'], data['low'], data['close'], length=length).astype(float)
            data[f'ATR_{length}'] = normalize_diff_vec(data[f'ATR_{length}'])
            result = data[[f'ATR_{length}']].dropna()

        elif indicator_name == 'stoch':
            k = int(params['k'])
            d = int(params['d'])
            stoch_df = stoch(data['high'], data['low'], data['close'], k=k, d=d).astype(float)
            result = stoch_df.dropna()

        elif indicator_name == 'cci':
            length = int(params['length'])
            data[f'CCI_{length}'] = cci(data['high'], data['low'], data['close'], length=length).astype(float)
            result = data[[f'CCI_{length}']].dropna()

        elif indicator_name == 'adx':
            length = int(params['length'])
            adx_series = adx(data['high'], data['low'], data['close'], length=length).astype(float)
            data[f'ADX_{length}'] = adx_series
            result = data[[f'ADX_{length}']].dropna()

        elif indicator_name == 'cmf':
            length = int(params['length'])
            data[f'CMF_{length}'] = cmf(data['high'], data['low'], data['close'], data['volume'], length=length).astype(float)
            result = data[[f'CMF_{length}']].dropna()

        elif indicator_name == 'mfi':
            length = int(params['length'])
            data[f'MFI_{length}'] = mfi(data['high'], data['low'], data['close'], data['volume'], length=length).astype(float)
            result = data[[f'MFI_{length}']].dropna()

        elif indicator_name == 'roc':
            length = int(params['length'])
            data[f'ROC_{length}'] = roc(data['close'], length=length).astype(float)
            data[f'ROC_{length}'] = normalize_diff_vec(data[f'ROC_{length}'])
            result = data[[f'ROC_{length}']].dropna()

        elif indicator_name == 'willr':
            length = int(params['length'])
            data[f'WILLR_{length}'] = willr(data['high'], data['low'], data['close'], length=length).astype(float)
            result = data[[f'WILLR_{length}']].dropna()

        elif indicator_name == 'psar':
            acceleration = float(params['acceleration'])
            max_acceleration = float(params['max_acceleration'])
            psar_df = psar(data['high'], data['low'], data['close'], acceleration=acceleration, max_acceleration=max_acceleration).astype(float)
            data = data.join(psar_df)
            result = data.dropna()

        elif indicator_name == 'ichimoku':
            tenkan = int(params['tenkan'])
            kijun = int(params['kijun'])
            senkou = int(params['senkou'])
            ichimoku_df = ichimoku(data['high'], data['low'], data['close'], tenkan=tenkan, kijun=kijun, senkou=senkou)
            data = data.join(ichimoku_df.astype(float))
            result = data.dropna()

        elif indicator_name == 'keltner':
            length = int(params['length'])
            multiplier = float(params['multiplier'])
            keltner_df = keltner(data['high'], data['low'], data['close'], length=length, multiplier=multiplier).astype(float)
            result = keltner_df.dropna()

        elif indicator_name == 'donchian':
            lower_length = int(params['lower_length'])
            upper_length = int(params['upper_length'])
            donchian_df = donchian(data['high'], data['low'], lower_length=lower_length, upper_length=upper_length).astype(float)
            result = donchian_df.dropna()

        elif indicator_name == 'emv':
            length = int(params['length'])
            emv_df = emv(data['high'], data['low'], data['close'], data['volume'], length=length).astype(float)
            result = emv_df.dropna()

        elif indicator_name == 'force':
            length = int(params['length'])
            data[f'FORCE_{length}'] = force(data['close'], data['volume'], length=length).astype(float)
            result = data[[f'FORCE_{length}']].dropna()

        elif indicator_name == 'uo':
            short = int(params['short'])
            medium = int(params['medium'])
            long = int(params['long'])
            data['UO'] = uo(data['high'], data['low'], data['close'], short=short, medium=medium, long=long).astype(float)
            result = data[['UO']].dropna()

        elif indicator_name == 'volatility':
            length = int(params['length'])
            data[f'STDDEV_{length}'] = volatility(data['close'], length=length).astype(float)
            result = data[[f'STDDEV_{length}']].dropna()

        elif indicator_name == 'dpo':
            length = int(params['length'])
            data['DPO'] = dpo(data['close'], length=length).astype(float)
            result = data[['DPO']].dropna()

        elif indicator_name == 'trix':
            length = int(params['length'])
            data['TRIX'] = trix(data['close'], length=length).iloc[:, 0].astype(float) if hasattr(trix(data['close'], length=length), 'iloc') else trix(data['close'], length=length).astype(float)
            result = data[['TRIX']].dropna()

        elif indicator_name == 'chaikin_osc':
            fast = int(params['fast'])
            slow = int(params['slow'])
            data['Chaikin_Osc'] = chaikin_osc(data['high'], data['low'], data['close'], data['volume'], fast=fast, slow=slow).astype(float)
            result = data[['Chaikin_Osc']].dropna()


        elif indicator_name == 'vwap':
            offset = int(params['offset'])
            data['VWAP'] = vwap(data['high'], data['low'], data['close'], data['volume'], offset=offset).astype(float)
            data['VWAP'] = normalize_diff_vec(data['VWAP'])
            result = pd.DataFrame(data['VWAP'] - data['close'], columns=['VWAP']).dropna()

        elif indicator_name == 'vwma':
            length = int(params['length'])
            data['VWMA'] = vwma(data['close'], data['volume'], length=length).astype(float)
            data['VWMA'] = normalize_diff_vec(data['VWMA'])
            result = pd.DataFrame(data['VWMA'] - data['close'], columns=['VWMA']).dropna()

        elif indicator_name == 'vpvr':
            width = int(params['width'])
            n_clusters = 100
            cluster_columns = [f'cluster_{i}' for i in range(n_clusters)]
            # Fix: Use a dictionary to initialize clusters_df
            data_dict = {col: np.zeros(len(data.index), dtype=np.float64) for col in cluster_columns}
            clusters_df = pd.DataFrame(data_dict, index=data.index)
            data = data.join(clusters_df)
            volume_profile_df = incremental_vpvr_fixed_bins(data, width=n_clusters)
            new_clusters = volume_profile_df.copy()
            for col in cluster_columns:
                new_clusters[col] = normalize_volume_vec(new_clusters[col].values)
            data = data.drop(columns=cluster_columns, errors='ignore').join(new_clusters)
            for col in cluster_columns:
                data[col] = normalize_volume_vec(data[col])
            result = data[cluster_columns].dropna()

        elif indicator_name == 'obv':
            data['OBV'] = obv(data['close'], data['volume']).astype(float)
            data['OBV'] = normalize_volume_vec(data['OBV'])
            result = data[['OBV']].dropna()

        elif indicator_name == 'hist_vol':
            length = int(params['length'])
            log_returns = np.log(data['close'] / data['close'].shift(1))
            data[f'Hist_Vol_{length}'] = log_returns.rolling(window=length).std().astype(float)
            data[f'Hist_Vol_{length}'] = normalize_diff_vec(data[f'Hist_Vol_{length}'])
            result = data[[f'Hist_Vol_{length}']].dropna()

        else:
            raise ValueError(f"Indicator {indicator_name} is not implemented.")

        if timing_enabled:
            elapsed = time.time() - t_start
            logger.info(f"Indicator {indicator_name} with params {params} on timeframe {timeframe} computed in {elapsed:.4f} seconds.")
        return result

    from datetime import datetime, date

    def filter_data_by_date(self, df: pd.DataFrame, start_date: Union[str, datetime, date],
                            end_date: Union[str, datetime, date]) -> pd.DataFrame:
        """
        Filter a DataFrame to include only rows between start_date and end_date.
        """
        return filter_data_by_date(df, start_date, end_date)
