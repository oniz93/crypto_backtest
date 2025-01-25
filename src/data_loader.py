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
        elif indicator_name == 'bbands':
            length = int(params['length'])
            std_dev = float(params['std_dev'])
            bbands_df = ta.bbands(data['close'], length=length, std=std_dev).astype(float)
            result = bbands_df.dropna()
        elif indicator_name == 'atr':
            length = int(params['length'])
            data[f'ATR_{length}'] = ta.atr(data['high'], data['low'], data['close'], length=length).astype(float)
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
            result = data[['VWAP']].dropna()
            result = result.sub(data['close'], axis=0)
        elif indicator_name == 'vwma':
            length = int(params['length'])
            data['VWMA'] = ta.vwma(data['close'], data['volume'], length=length).astype(float)
            result = data[['VWMA']].dropna()
            result = result.sub(data['close'], axis=0)
        elif indicator_name == 'vpvr':
            width = int(params['width'])
            # Initialize an empty temporary DataFrame with the same columns
            temp_df = pd.DataFrame(columns=data.columns)
            for idx, row in data.iterrows():
                # Append the current row to the temporary DataFrame
                temp_df = temp_df.append(row, ignore_index=True)
                vp_df = ta.vp(temp_df['close'], temp_df['volume'], width=width, sort_close=True).astype(float)
                binned = self.create_weighted_volume_clusters(vp_df)
            result = vp_df.dropna()
            result = result.sub(data['close'], axis=0)
        else:
            raise ValueError(f"Indicator {indicator_name} is not implemented.")
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

    def create_weighted_volume_clusters(self,
                                        vp_df: pd.DataFrame,
                                        n_clusters: int = 100,
                                        weight_column: str = 'mean_close',
                                        binning_method: str = 'equal_width') -> pd.DataFrame:
        """
        Creates clusters based on 'mean_close' and computes the weighted mean of 'total_volume' for each cluster.

        Args:
            vp_df (pd.DataFrame): DataFrame containing VPVR data with columns
                                  ['low_close', 'mean_close', 'high_close', 'pos_volume', 'neg_volume', 'total_volume'].
            n_clusters (int, optional): Number of clusters to create. Default is 100.
            weight_column (str, optional): Column to use as weights for the weighted mean. Default is 'mean_close'.
            binning_method (str, optional): Method to bin 'mean_close'.
                                            'equal_width' for equal-width bins,
                                            'quantile' for quantile-based bins. Default is 'equal_width'.

        Returns:
            pd.DataFrame: DataFrame with cluster information and weighted mean of 'total_volume'.
        """
        # Validate input DataFrame
        required_columns = {'low_close', 'mean_close', 'high_close', 'pos_volume', 'neg_volume', 'total_volume'}
        if not required_columns.issubset(vp_df.columns):
            missing = required_columns - set(vp_df.columns)
            raise ValueError(f"Input DataFrame is missing columns: {missing}")

        # Validate weight_column
        if weight_column not in vp_df.columns:
            raise ValueError(f"Weight column '{weight_column}' not found in DataFrame.")

        # Choose binning method
        if binning_method == 'equal_width':
            # Create equal-width bins
            vp_df['cluster'] = pd.cut(vp_df['mean_close'], bins=n_clusters, labels=False, include_lowest=True)
        elif binning_method == 'quantile':
            # Create quantile-based bins
            vp_df['cluster'] = pd.qcut(vp_df['mean_close'], q=n_clusters, labels=False, duplicates='drop')
            # Note: 'duplicates=drop' handles cases where there are not enough unique values to form all bins
        else:
            raise ValueError("Invalid binning_method. Choose 'equal_width' or 'quantile'.")

        # Handle possible NaN values after binning
        if vp_df['cluster'].isnull().any():
            print("Warning: Some rows could not be assigned to a cluster and will be excluded from aggregation.")

        # Drop rows with NaN in 'cluster'
        clustered_df = vp_df.dropna(subset=['cluster']).copy()

        # Convert 'cluster' to integer type
        clustered_df['cluster'] = clustered_df['cluster'].astype(int)

        # Group by 'cluster' and calculate weighted mean of 'total_volume'
        aggregated = clustered_df.groupby('cluster').apply(
            lambda x: pd.Series({
                'price_range_low': x['low_close'].min(),
                'price_range_high': x['high_close'].max(),
                'mean_close': (x['mean_close'] * x[weight_column]).sum() / x[weight_column].sum(),
                'weighted_total_volume': (x['total_volume'] * x[weight_column]).sum() / x[weight_column].sum(),
                'sum_total_volume': x['total_volume'].sum(),
                'count': x['total_volume'].count()
            })
        ).reset_index()

        # Optional: Sort by 'mean_close'
        aggregated.sort_values('mean_close', inplace=True)

        cluster_dict = {f'cluster_{i}': 0 for i in range(n_clusters)}

        # Populate the dictionary with actual weighted_total_volume values
        for _, row in aggregated.iterrows():
            cluster_name = f'cluster_{int(row["cluster"])}'
            cluster_dict[cluster_name] = row['weighted_total_volume']

        # Create a single-row DataFrame
        single_row_df = pd.DataFrame([cluster_dict])

        return single_row_df