"""
dataframe_strategy.py
---------------------
This module implements the Strategy pattern for dataframe operations,
allowing seamless switching between Pandas and cuDF based on GPU availability.

We have:
- BaseStrategy: an abstract base that implements most indicator calculations
  plus a dictionary-based dispatch to those methods. It *doesn't* implement
  _calculate_vpvr, because that differs for Pandas vs cuDF.
- PandasStrategy: extends BaseStrategy, overrides or adds the 'vpvr' key in
  indicator_funcs with a Pandas-based method.
- CuDFStrategy: extends BaseStrategy, likewise overrides 'vpvr' with a cuDF-based method.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from numba import njit

try:
    import cudf
    import dask_cudf
    import cupy as cp
    NUM_GPU = cp.cuda.runtime.getDeviceCount()
    USING_CUDF = NUM_GPU > 0
except ImportError:
    cudf = None
    dask_cudf = None
    cp = None
    USING_CUDF = False
    NUM_GPU = 0

from src.utils import normalize_price_vec, normalize_volume_vec, normalize_diff_vec, normalize_rsi_vec


# -------------- Numba / VPVR Helpers (old code) --------------

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
    Wrapper for computing cumulative volume profile using the numba function.

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


def incremental_vpvr_fixed_bins(df: pd.DataFrame,
                                width: int = 100,
                                n_rows: int = None,
                                bins_array: np.ndarray = None) -> pd.DataFrame:
    """
    A one-pass algorithm to compute incremental volume-by-price for Pandas data.

    Parameters:
        df (pd.DataFrame): DataFrame with 'close' and 'volume' columns.
        width (int): Number of bins.
        n_rows (int, optional): If you only want to process the first n rows.
        bins_array (np.ndarray, optional): Custom bin edges.

    Returns:
        pd.DataFrame: cumulative volume distribution per row.
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


def incremental_vpvr_fixed_bins_gpu(df: cudf.DataFrame,
                                     width: int = 100,
                                     n_rows: int = None,
                                     bins_array: np.ndarray = None) -> cudf.DataFrame:
    """
    A one-pass algorithm to compute incremental volume-by-price for cuDF data.

    Parameters:
        df (cudf.DataFrame): DataFrame with 'close' and 'volume' columns.
        width (int): Number of bins (default is 100).
        n_rows (int, optional): Number of rows to process; if None, process all rows.
        bins_array (np.ndarray, optional): Custom bin edges; if None, bins are auto-generated.

    Returns:
        cudf.DataFrame: Cumulative volume distribution per row, with columns 'cluster_0' to 'cluster_{width-1}'.
    """
    # Sort the DataFrame by index
    df = df.sort_index()

    # Slice the DataFrame if n_rows is specified
    if n_rows is not None:
        df = df.iloc[:n_rows]

    n = len(df)
    # Handle empty DataFrame case
    if n == 0:
        return cudf.DataFrame([], columns=[f"cluster_{c}" for c in range(width)])

    # Extract 'close' and 'volume' as float64 Series
    closes = df['close'].astype('float64')
    volumes = df['volume'].astype('float64')

    # Determine bin edges and indices
    if bins_array is None:
        min_price = closes.min()
        max_price = closes.max()
        if min_price == max_price:
            # All volumes go to the first bin if prices are identical
            bin_indices = cp.zeros(n, dtype=cp.int64)
        else:
            # Create bins based on min and max prices
            bins_array = np.linspace(min_price, max_price, width + 1)
            bins_series = cudf.Series(bins_array)
            bin_indices = (bins_series.searchsorted(closes, side='right') - 1).clip(0, width - 1)
    else:
        # Use provided bins_array and adjust width
        width = len(bins_array) - 1
        bins_series = cudf.Series(bins_array)
        bin_indices = (bins_series.searchsorted(closes, side='right') - 1).clip(0, width - 1)

    # Initialize output array and populate with volumes
    out = cp.zeros((n, width), dtype=cp.float64)
    rows = cp.arange(n)
    out[rows, bin_indices] = volumes.values

    # Compute cumulative sum along rows
    cumulative_dist = cp.cumsum(out, axis=0)

    # Create result DataFrame with original index and cluster column names
    result_df = cudf.DataFrame(
        cumulative_dist,
        index=df.index,
        columns=[f"cluster_{c}" for c in range(width)]
    )

    # Round to 2 decimal places
    result_df = result_df.round(2)

    return result_df


# -------------- Abstract & Base Classes --------------

class DataFrameStrategy(ABC):
    """Abstract base class defining the interface for dataframe operations."""

    @abstractmethod
    def load_parquet(self, file_path):
        """Load a Parquet file into a dataframe."""
        pass

    @abstractmethod
    def resample(self, data, timeframe):
        """Resample a dataframe to a specified timeframe."""
        pass

    @abstractmethod
    def filter_by_date(self, data, start, end):
        """Filter dataframe by date range."""
        pass

    @abstractmethod
    def calculate_indicator(self, indicator_name, data, params, timeframe):
        """Calculate a technical indicator on the dataframe, by name."""
        pass


class BaseStrategy(DataFrameStrategy):
    """
    Base class with common indicator implementations.
    We store them in a dictionary so that 'calculate_indicator' can dispatch easily.
    This base class does NOT implement `_calculate_vpvr` because that differs
    for Pandas vs cuDF. The children classes each add 'vpvr' to their dict.
    """

    def __init__(self):
        # Dictionary: map indicator_name -> bound method
        # The children classes (PandasStrategy, CuDFStrategy) will add 'vpvr'
        self.indicator_funcs = {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bbands': self._calculate_bbands,
            'atr': self._calculate_atr,
            'stoch': self._calculate_stoch,
            'cci': self._calculate_cci,
            'adx': self._calculate_adx,
            'cmf': self._calculate_cmf,
            'mfi': self._calculate_mfi,
            'roc': self._calculate_roc,
            'willr': self._calculate_willr,
            'psar': self._calculate_psar,
            'ichimoku': self._calculate_ichimoku,
            'keltner': self._calculate_keltner,
            'donchian': self._calculate_donchian,
            'emv': self._calculate_emv,
            'force': self._calculate_force,
            'uo': self._calculate_uo,
            'volatility': self._calculate_volatility,
            'dpo': self._calculate_dpo,
            'trix': self._calculate_trix,
            'chaikin_osc': self._calculate_chaikin_osc,
            'obv': self._calculate_obv,
            'hist_vol': self._calculate_hist_vol,
            'vwma': self._calculate_vwma,
            # 'vpvr': intentionally omitted here: each child has a special version
        }

    @abstractmethod
    def load_parquet(self, file_path):
        pass

    @abstractmethod
    def resample(self, data, timeframe):
        pass

    @abstractmethod
    def filter_by_date(self, data, start, end):
        pass

    def calculate_indicator(self, indicator_name, data, params, timeframe):
        """
        Dispatch to the relevant calculation method via the dictionary.
        'vpvr' is included in the child classes, not in the base dictionary.
        """
        func = self.indicator_funcs.get(indicator_name)
        if func is None:
            raise ValueError(f"Indicator '{indicator_name}' is not implemented in {type(self).__name__}.")
        return func(data, params)

    # ----------------------------------------------------------------
    # All your indicator methods from the old snippet (except vpvr):
    # ----------------------------------------------------------------

    def _calculate_sma(self, data, params):
        length = int(params['length'])
        sma_series = data['close'].rolling(window=length).mean().astype(float)
        col_name = f'SMA_{length}'
        df = pd.DataFrame({col_name: normalize_price_vec(sma_series)}, index=data.index)
        return df.dropna()

    def _calculate_ema(self, data, params):
        length = int(params['length'])
        ema_series = data['close'].ewm(span=length, adjust=False).mean().astype(float)
        col_name = f'EMA_{length}'
        df = pd.DataFrame({col_name: normalize_price_vec(ema_series)}, index=data.index)
        return df.dropna()

    def _calculate_rsi(self, data, params):
        length = int(params.get('length', 14))
        delta = data['close'].diff()
        # Handle different DataFrame types
        if hasattr(delta, 'clip'):
            # For Pandas
            up = delta.clip(lower=0)
            down = (-delta).clip(lower=0)
        else:
            # For cuDF
            up = delta.copy()
            up[up < 0] = 0
            down = -delta.copy()
            down[down < 0] = 0
            
        ma_up = up.rolling(window=length).mean()
        ma_down = down.rolling(window=length).mean()
        rs = ma_up / ma_down
        rsi_series = (100 - (100 / (1 + rs))).astype(float)
        col_name = f'RSI_{length}'
        
        # Handle different DataFrame types
        if hasattr(data, 'to_pandas'):
            # For cuDF
            df = cudf.DataFrame({col_name: normalize_rsi_vec(rsi_series)}, index=data.index)
        else:
            # For Pandas
            df = pd.DataFrame({col_name: normalize_rsi_vec(rsi_series)}, index=data.index)
            
        return df.dropna()

    def _calculate_macd(self, data, params):
        fast = int(params.get('fast', 12))
        slow = int(params.get('slow', 26))
        signal = int(params.get('signal', 9))
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).astype(float)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean().astype(float)
        histogram = (macd_line - signal_line).astype(float)
        
        # Handle cuDF vs pandas
        if hasattr(data, 'to_pandas'):
            # For cuDF
            macd_df = cudf.DataFrame({
                f'MACD_{fast}_{slow}_{signal}': macd_line,
                f'Signal_{fast}_{slow}_{signal}': signal_line,
                f'Histogram_{fast}_{slow}_{signal}': histogram
            }, index=data.index)
        else:
            # For pandas
            macd_df = pd.DataFrame({
                f'MACD_{fast}_{slow}_{signal}': macd_line,
                f'Signal_{fast}_{slow}_{signal}': signal_line,
                f'Histogram_{fast}_{slow}_{signal}': histogram
            }, index=data.index)
            
        for c in macd_df.columns:
            macd_df[c] = normalize_diff_vec(macd_df[c])
        return macd_df.dropna()

    def _calculate_bbands(self, data, params):
        length = int(params.get('length', 20))
        std_dev = float(params.get('std_dev', 2.0))
        sma_val = data['close'].rolling(window=length).mean()
        rolling_std = data['close'].rolling(window=length).std()
        upper = (sma_val + std_dev * rolling_std).astype(float)
        lower = (sma_val - std_dev * rolling_std).astype(float)
        mid = sma_val.astype(float)
        
        # Handle cuDF vs pandas
        if hasattr(data, 'to_pandas'):
            # For cuDF
            bb_df = cudf.DataFrame({
                f'BB_Upper_{length}_{std_dev}': upper,
                f'BB_Middle_{length}_{std_dev}': mid,
                f'BB_Lower_{length}_{std_dev}': lower
            }, index=data.index)
        else:
            # For pandas
            bb_df = pd.DataFrame({
                f'BB_Upper_{length}_{std_dev}': upper,
                f'BB_Middle_{length}_{std_dev}': mid,
                f'BB_Lower_{length}_{std_dev}': lower
            }, index=data.index)
            
        for c in bb_df.columns:
            bb_df[c] = normalize_price_vec(bb_df[c])
        return bb_df.dropna()

    def _calculate_atr(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        if hasattr(data, 'to_pandas'):
            true_range = cudf.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_series = true_range.rolling(window=length).mean().astype(float)
        atr_series = normalize_diff_vec(atr_series)
        col_name = f'ATR_{length}'
        data[col_name] = atr_series
        return data[[col_name]].dropna()

    def _calculate_stoch(self, data, params):
        k = int(params['k'])
        d = int(params['d'])
        high = data['high']
        low = data['low']
        close = data['close']

        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d).mean()
        df = cudf.DataFrame({
            f'Stoch_K_{k}_{d}': stoch_k,
            f'Stoch_D_{k}_{d}': stoch_d,
        }, index=data.index)
        return df.dropna()

    def _calculate_cci(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']

        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=length).mean()
        mean_deviation = typical_price.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))),
            raw=True
        )
        cci_series = (typical_price - sma_tp) / (0.015 * mean_deviation)
        cci_series = cci_series.astype(float)

        col_name = f'CCI_{length}'
        data[col_name] = cci_series
        return data[[col_name]].dropna()

    def _calculate_adx(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']

        up_move = high.diff()
        down_move = low.diff().abs()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = (-low.diff()).where(((-low.diff()) > up_move) & ((-low.diff()) > 0), 0)

        if hasattr(data, 'to_pandas'):
            tr = cudf.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
        else:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

        atr_val = tr.rolling(window=length).mean()
        plus_di = 100 * (plus_dm.rolling(window=length).sum() / atr_val)
        minus_di = 100 * (minus_dm.rolling(window=length).sum() / atr_val)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx_series = dx.rolling(window=length).mean().astype(float)

        col_name = f'ADX_{length}'
        data[col_name] = adx_series
        return data[[col_name]].dropna()

    def _calculate_cmf(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        cmf_series = (
            money_flow_volume.rolling(window=length).sum()
            / volume.rolling(window=length).sum()
        ).astype(float)

        col_name = f'CMF_{length}'
        data[col_name] = cmf_series
        return data[[col_name]].dropna()

    def _calculate_mfi(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        tp_diff = typical_price.diff()
        positive_flow = money_flow.where(tp_diff > 0, 0)
        negative_flow = money_flow.where(tp_diff < 0, 0)
        pos_mf = positive_flow.rolling(window=length).sum()
        neg_mf = negative_flow.rolling(window=length).sum().abs()
        mfr = pos_mf / neg_mf
        mfi_series = (100 - (100 / (1 + mfr))).astype(float)

        col_name = f'MFI_{length}'
        data[col_name] = mfi_series
        return data[[col_name]].dropna()

    def _calculate_roc(self, data, params):
        length = int(params['length'])
        close = data['close']
        roc_series = ((close / close.shift(length)) - 1) * 100
        roc_series = normalize_diff_vec(roc_series.astype(float))

        col_name = f'ROC_{length}'
        data[col_name] = roc_series
        return data[[col_name]].dropna()

    def _calculate_willr(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        close = data['close']

        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        willr_series = -100 * (highest_high - close) / (highest_high - lowest_low)
        col_name = f'WILLR_{length}'
        data[col_name] = willr_series.astype(float)
        return data[[col_name]].dropna()

    def _calculate_psar(self, data, params):
        acceleration = float(params['acceleration'])
        max_acceleration = float(params['max_acceleration'])

        high = data['high']
        low = data['low']
        close = data['close']

        psar_series = close.copy()
        trend = 1
        af = acceleration
        ep = low.iloc[0]

        for i in range(1, len(close)):
            psar_series.iloc[i] = psar_series.iloc[i - 1] + af * (ep - psar_series.iloc[i - 1])
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

        psar_df = psar_series.to_frame()
        col_name = f'PSAR_{acceleration}_{max_acceleration}'
        psar_df.columns = [col_name]
        data = data.join(psar_df.astype(float))
        return data[[col_name]].dropna()

    def _calculate_ichimoku(self, data, params):
        tenkan = int(params['tenkan'])
        kijun = int(params['kijun'])
        senkou = int(params['senkou'])

        high = data['high']
        low = data['low']

        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2).shift(kijun)

        if hasattr(data, 'to_pandas'):
            ichimoku_df = cudf.concat(
                [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b],
                axis=1
            ).astype(float)
        else:
            ichimoku_df = pd.concat(
                [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b],
                axis=1
            ).astype(float)
        ichimoku_df.columns = [
            f'Tenkan_{tenkan}',
            f'Kijun_{kijun}',
            f'Senkou_A_{senkou}',
            f'Senkou_B_{senkou}'
        ]
        data = data.join(ichimoku_df)
        return data[list(ichimoku_df.columns)].dropna()

    def _calculate_keltner(self, data, params):
        length = int(params['length'])
        multiplier = float(params['multiplier'])
        high = data['high']
        low = data['low']
        close = data['close']

        typical_price = (high + low + close) / 3
        ema_tp = typical_price.ewm(span=length, adjust=False).mean()
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        if hasattr(data, 'to_pandas'):
            true_range = cudf.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = true_range.rolling(window=length).mean()

        upper = ema_tp + multiplier * atr_val
        middle = ema_tp
        lower = ema_tp - multiplier * atr_val

        if hasattr(data, 'to_pandas'):
            keltner_df = cudf.concat([upper, middle, lower], axis=1).astype(float)
        else:
            keltner_df = pd.concat([upper, middle, lower], axis=1).astype(float)
        keltner_df.columns = [
            f'Keltner_Upper_{length}_{multiplier}',
            f'Keltner_Middle_{length}_{multiplier}',
            f'Keltner_Lower_{length}_{multiplier}'
        ]
        return keltner_df.dropna()

    def _calculate_donchian(self, data, params):
        lower_length = int(params['lower_length'])
        upper_length = int(params['upper_length'])
        high = data['high']
        low = data['low']
        upper = high.rolling(window=upper_length).max()
        lower = low.rolling(window=lower_length).min()

        if hasattr(data, 'to_pandas'):
            donchian_df = cudf.concat([upper, lower], axis=1).astype(float)
        else:
            donchian_df = pd.concat([upper, lower], axis=1).astype(float)
        donchian_df.columns = [
            f'Donchian_Upper_{upper_length}',
            f'Donchian_Lower_{lower_length}'
        ]
        return donchian_df.dropna()

    def _calculate_emv(self, data, params):
        length = int(params['length'])
        high = data['high']
        low = data['low']
        volume = data['volume']
        bp = (high + low) / 2
        tr = high - low
        emv_raw = bp.diff() / tr.replace(0, np.nan)
        emv_series = emv_raw.rolling(window=length).mean().astype(float)
        col_name = f'EMV_{length}'
        data[col_name] = emv_series
        return data[[col_name]].dropna()

    def _calculate_force(self, data, params):
        length = int(params['length'])
        close = data['close']
        volume = data['volume']
        force_series = (close.diff(length) * volume).rolling(window=length).mean().astype(float)
        col_name = f'FORCE_{length}'
        data[col_name] = force_series
        return data[[col_name]].dropna()

    def _calculate_uo(self, data, params):
        short = int(params['short'])
        medium = int(params['medium'])
        long_ = int(params['long'])
        high = data['high']
        low = data['low']
        close = data['close']

        if hasattr(data, 'to_pandas'):
            bp = close - cudf.concat([low, close.shift(1)], axis=1).min(axis=1)
            tr = cudf.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
        else:
            bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)

        avg_short = bp.rolling(window=short).sum() / tr.rolling(window=short).sum()
        avg_medium = bp.rolling(window=medium).sum() / tr.rolling(window=medium).sum()
        avg_long = bp.rolling(window=long_).sum() / tr.rolling(window=long_).sum()

        uo_series = (100 * (4 * avg_short + 2 * avg_medium + avg_long) / 7).astype(float)
        col_name = f'UO_{short}_{medium}_{long_}'
        data[col_name] = uo_series
        return data[[col_name]].dropna()

    def _calculate_volatility(self, data, params):
        length = int(params['length'])
        close = data['close']
        vol_series = close.rolling(window=length).std().astype(float)
        col_name = f'STDDEV_{length}'
        data[col_name] = vol_series
        return data[[col_name]].dropna()

    def _calculate_dpo(self, data, params):
        length = int(params['length'])
        close = data['close']
        sma_val = close.rolling(window=length).mean()
        dpo_series = (close - sma_val.shift(int(length / 2))).astype(float)
        col_name = f'DPO_{length}'
        data[col_name] = dpo_series
        return data[[col_name]].dropna()

    def _calculate_trix(self, data, params):
        length = int(params['length'])
        close = data['close']
        ema1 = close.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        trix_series = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
        col_name = f'TRIX_{length}'
        data[col_name] = trix_series.astype(float)
        return data[[col_name]].dropna()

    def _calculate_chaikin_osc(self, data, params):
        fast = int(params['fast'])
        slow = int(params['slow'])
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        ad = ((close - low) - (high - close)) / (high - low) * volume
        ema_fast = ad.ewm(span=fast, adjust=False).mean()
        ema_slow = ad.ewm(span=slow, adjust=False).mean()
        cho_series = (ema_fast - ema_slow).astype(float)
        col_name = f'Chaikin_Osc_{fast}_{slow}'
        data[col_name] = cho_series
        return data[[col_name]].dropna()

    def _calculate_obv(self, data, params):
        close = data['close']
        volume = data['volume']
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv_series = (volume * direction).cumsum().astype(float)
        obv_series = normalize_volume_vec(obv_series)
        data['OBV'] = obv_series
        return data[['OBV']].dropna()

    def _calculate_hist_vol(self, data, params):
        length = int(params['length'])
        close = data['close']
        log_returns = np.log(close / close.shift(1))
        hv_series = log_returns.rolling(window=length).std().astype(float)
        hv_series = normalize_diff_vec(hv_series)
        col_name = f'Hist_Vol_{length}'
        data[col_name] = hv_series
        return data[[col_name]].dropna()

    def _calculate_vwma(self, data, params):
        length = int(params['length'])
        close = data['close']
        volume = data['volume']
        tp_vol = close * volume
        sum_vol = volume.rolling(window=length).sum()
        raw_vwma = (tp_vol.rolling(window=length).sum() / sum_vol).astype(float)
        col_name = f'VWMA_{length}'
        data[col_name] = normalize_diff_vec(raw_vwma)
        final_series = (data[col_name] - close).dropna()
        # Return as a 1-col df named the same col_name or 'VWMA'?
        return pd.DataFrame(final_series, columns=[col_name]).dropna()

    # We'll skip _calculate_vpvr here. It's specialized per child class.

# -------------- Pandas vs cuDF --------------

class PandasStrategy(BaseStrategy):
    """Strategy implementation using Pandas."""

    def __init__(self):
        super().__init__()
        # Add 'vpvr' key for Pandas-based approach
        self.indicator_funcs['vpvr'] = self._calculate_vpvr
        # Add 'vwap' key for Pandas-based approach
        self.indicator_funcs['vwap'] = self._calculate_vwap
        # Add 'vwma' key for Pandas-based approach
        self.indicator_funcs['vwma'] = self._calculate_vwma
        
    def _calculate_vwma(self, data, params):
        """Calculate Volume Weighted Moving Average for Pandas."""
        length = int(params.get('length', 14))
        close = data['close']
        volume = data['volume']
        
        tp_vol = close * volume
        sum_vol = volume.rolling(window=length).sum()
        vwma = tp_vol.rolling(window=length).sum() / sum_vol
        vwma = vwma.astype(float)
        
        col_name = f'VWMA_{length}'
        df = pd.DataFrame({col_name: normalize_price_vec(vwma)}, index=data.index)
        return df.dropna()
        
    def _calculate_vwap(self, data, params):
        """Calculate Volume Weighted Average Price for Pandas."""
        offset = int(params.get('offset', 0))
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        if offset > 0:
            vwap = vwap.shift(offset)
            
        vwap = normalize_price_vec(vwap.astype(float))
        col_name = f'VWAP_{offset}'
        df = pd.DataFrame({col_name: vwap}, index=data.index)
        return df.dropna()

    def load_parquet(self, file_path):
        return pd.read_parquet(file_path)

    def resample(self, data, timeframe):
        return data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def filter_by_date(self, data, start, end):
        return data[(data.index >= start) & (data.index <= end)]

    def _calculate_vpvr(self, data, params):
        """
        The Pandas-based approach for vpvr, using incremental_vpvr_fixed_bins.
        Mimics your old logic with normalizing, rejoining, etc.
        """
        width = int(params['width'])
        n_clusters = 100
        cluster_columns = [f'cluster_{i}' for i in range(n_clusters)]
        # We can do a copy to avoid mutating the original
        data = data.copy()
        if data['close'].isnull().any():
            data.dropna(subset=['close'], inplace=True)

        volume_profile_df = incremental_vpvr_fixed_bins(data, width=n_clusters)

        # Normalize
        for col in cluster_columns:
            volume_profile_df[col] = normalize_volume_vec(volume_profile_df[col].values)

        # Provide a final joined result if you want (like your old code):
        data = data.reset_index(drop=False)  # Keep the original index as 'timestamp'
        new_clusters = volume_profile_df.reset_index(drop=False)

        # Possibly drop old cluster cols if they exist
        data = data.drop(columns=cluster_columns, errors='ignore').join(
            new_clusters, on='timestamp', lsuffix='_left', rsuffix='_right'
        )
        data = data.set_index('timestamp')
        for col in cluster_columns:
            data[col] = normalize_volume_vec(data[col])
        result = data[cluster_columns].dropna()
        return result


class CuDFStrategy(BaseStrategy):
    """Strategy implementation using cuDF."""

    def __init__(self):
        super().__init__()
        # Add 'vpvr' key for cuDF-based approach
        self.indicator_funcs['vpvr'] = self._calculate_vpvr
        # Add 'vwap' key for cuDF-based approach
        self.indicator_funcs['vwap'] = self._calculate_vwap
        # Add 'vwma' key for cuDF-based approach
        self.indicator_funcs['vwma'] = self._calculate_vwma
        
    def _calculate_vwma(self, data, params):
        """Calculate Volume Weighted Moving Average for cuDF."""
        length = int(params.get('length', 14))
        close = data['close']
        volume = data['volume']
        
        tp_vol = close * volume
        sum_vol = volume.rolling(window=length).sum()
        vwma = tp_vol.rolling(window=length).sum() / sum_vol
        vwma = vwma.astype(float)
        
        col_name = f'VWMA_{length}'
        df = cudf.DataFrame({col_name: normalize_price_vec(vwma)}, index=data.index)
        return df.dropna()
        
    def _calculate_vwap(self, data, params):
        """Calculate Volume Weighted Average Price for cuDF."""
        offset = int(params.get('offset', 0))
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        if offset > 0:
            vwap = vwap.shift(offset)
            
        vwap = normalize_price_vec(vwap.astype(float))
        col_name = f'VWAP_{offset}'
        df = cudf.DataFrame({col_name: vwap}, index=data.index)
        return df.dropna()

    def load_parquet(self, file_path):
        # if NUM_GPU >1:
        #     return dask_cudf.read_parquet(file_path, npartitions=NUM_GPU)
        return cudf.read_parquet(file_path)

    def resample(self, data, timeframe):
        # cuDF resample usage can differ from pandas, adapt if necessary.
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        # If your version of cuDF doesn't support .resample exactly like Pandas,
        # you may need a workaround. Shown here is a direct approach:
        return data.resample(timeframe).agg(agg_dict).dropna()

    def filter_by_date(self, data, start, end):
        return data[(data.index >= start) & (data.index <= end)]

    def _calculate_vpvr(self, data, params):
        """
        The cuDF-based approach for vpvr, using incremental_vpvr_fixed_bins_gpu.
        Mirrors your old code, but in a single method here.
        """
        width = int(params['width'])
        n_clusters = 100
        cluster_columns = [f'cluster_{i}' for i in range(n_clusters)]
        data = data.copy()
        if data['close'].isnull().any():
            data.dropna(subset=['close'], inplace=True)

        volume_profile_df = incremental_vpvr_fixed_bins_gpu(data, width=n_clusters)

        # Normalize
        for col in cluster_columns:
            volume_profile_df[col] = normalize_volume_vec(volume_profile_df[col].values)

        # Rejoin with original data if you want (like old code):
        data = data.reset_index(drop=False)
        new_clusters = volume_profile_df.reset_index(drop=False)

        # Change this line:
        data = data.drop(columns=cluster_columns, errors='ignore').merge(
            new_clusters, on='timestamp', suffixes=('_left', '_right')
        )
        data = data.set_index('timestamp')

        for col in cluster_columns:
            data[col] = normalize_volume_vec(data[col])
        return data[cluster_columns].dropna()


def get_dataframe_strategy():
    """Factory function to select the appropriate strategy based on GPU availability."""
    if cp is not None and cudf is not None:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return CuDFStrategy()
        except Exception:
            pass
    return PandasStrategy()
