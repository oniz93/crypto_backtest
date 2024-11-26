# src/trading_strategy.py

import pandas as pd
import numpy as np
import random
import time
import os
import csv
import json

from datetime import datetime, timedelta, timezone

from src.utils import (
    sat_to_usd, usd_to_sat, round_to_100, get_next_rebuy_pct,
    protected_div, timestamp_to_datetime
)
import numexpr as ne

# Import machine learning libraries
from sklearn.linear_model import LogisticRegression


# You can replace LogisticRegression with other models if desired

class TradingStrategy:
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        # Initialize attributes (from your existing code)
        self.last_support = 0
        self.last_resistance = 0
        self.last_order_price = 0
        self.real_last_order_price = 0
        self.last_order_qty = 0
        self.last_price = 0
        self.last_entry_price = 0
        self.treshold_price = config.get('treshold_price', 0.00001)
        self.positions = []
        self.order_qty = 0
        self.price_change_average_length = 10
        self.last_price_change = 0
        self.fees = 0.05
        self.current_leverage = 25
        self.unrealized_pnl_pct = 0
        self.initial_balance = config.get('initial_balance', 100000000)
        self.available_balance = self.initial_balance
        self.max_unrealized_pnl = 0
        self.min_unrealized_pnl = 0

        self.max_leverage = 20
        self.short_long = config.get('short_long', 'l')
        self.pct_new_level = config.get('pct_new_level', 100.5)
        self.current_order = {
            "quantity": 0,
            "entry_price": 0,
            "sat_qty": 0,
            "fee_paid": 0,
            "entry_time": None,
        }
        self.average = 0
        self.average_neg = 0
        self.old_min_alpha = 0
        self.old_min_sr = 0
        self.old_min_fr = 0

        self.trades = []
        self.funding_rates = self.data_loader.funding_rates
        self.id_trade = -1
        self.month = 1
        self.alpha = 0
        self.alphaneg = 0
        self.funding_rate = 0

        self.orders = []

        # Parameters (from your existing code)
        self.N_TICKS_DELTA = config.get('N_TICKS_DELTA', 8)
        self.N_TICKS_DELTA_NEG = config.get('N_TICKS_DELTA_NEG', 100)
        self.PCT_VOLUME_DISTANCE = config.get('PCT_VOLUME_DISTANCE', 0.4)
        self.PCT_VOLUME_DISTANCE_NEGATIVE = config.get('PCT_VOLUME_DISTANCE_NEGATIVE', 4)
        self.MULT_NEG_DELTA = config.get('MULT_NEG_DELTA', 32)
        self.pct_distance = 100
        self.N_REBUY = config.get('N_REBUY', 7)
        self.DEFAULT_PCT_STOP_LOSS = config.get('DEFAULT_PCT_STOP_LOSS', -15)
        self.PCT_QTY_ORDER = config.get('PCT_QTY_ORDER', 10)
        self.PCT_STOP_LOSS = config.get('PCT_STOP_LOSS', -3)

        self.step_rebuy = 2

        self.rand_number = random.randint(1, 1234567890)

        # Load data (from your existing code)
        self.calc_sr = self.data_loader.calc_sr
        self.calc_alpha = self.data_loader.calc_alpha
        self.ticks_1m = self.data_loader.ticks_1m
        self.ticks_5m = self.data_loader.ticks_5m

        # Internal variables (from your existing code)
        self.old_min_alpha = 0
        self.old_min_fr = 0
        self.old_min_sr = 0

        # Trades data (from your existing code)
        self.trades = {}
        self.trades['price'] = []
        self.trades['timestamp'] = []
        self.id_trade = -1
        self.month = 2
        self.current_time = 0
        self.last_price = 0

        # New attributes for the regression models
        self.model_buy = None
        self.model_sell = None
        self.threshold_buy = 0.6  # You can adjust the threshold
        self.threshold_sell = 0.6
        self.features = None
        self.labels_buy = None
        self.labels_sell = None
        self.X_train = None
        self.y_train_buy = None
        self.y_train_sell = None
        self.X_test = None
        self.y_test_buy = None
        self.y_test_sell = None

        # Prepare data and train models
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares the dataset by merging indicators and creating labels for training.
        """
        # Assuming self.data_loader.tick_data is a dictionary of DataFrames keyed by timeframe
        # For simplicity, we'll use the base timeframe data and merge indicators
        base_tf = self.data_loader.base_timeframe
        price_data = self.data_loader.tick_data[base_tf].copy()

        # Merge all indicators into one DataFrame
        indicators_df = price_data.copy()
        # Ensure self.data_loader.indicators exists
        if hasattr(self.data_loader, 'indicators'):
            for indicator_name, tf_dict in self.data_loader.indicators.items():
                for tf, df in tf_dict.items():
                    # Resample or reindex to base timeframe if necessary
                    df = df.reindex(indicators_df.index, method='ffill')
                    indicators_df = indicators_df.join(df, rsuffix=f'_{indicator_name}_{tf}')
        else:
            print("Indicators not found in data_loader.")
            # You might want to handle this case differently

        # Drop rows with NaN values
        indicators_df.dropna(inplace=True)

        # Create labels for buy (1) and hold/sell (0)
        indicators_df['future_return'] = indicators_df['close'].shift(-1) / indicators_df['close'] - 1
        indicators_df['buy_signal'] = np.where(indicators_df['future_return'] > 0, 1, 0)
        indicators_df['sell_signal'] = np.where(indicators_df['future_return'] < 0, 1, 0)

        # Split features and labels
        self.features = indicators_df.drop(columns=[
            'open', 'high', 'low', 'close', 'volume',
            'future_return', 'buy_signal', 'sell_signal'
        ])
        self.labels_buy = indicators_df['buy_signal']
        self.labels_sell = indicators_df['sell_signal']

        # Split into training and testing sets (e.g., 80% training, 20% testing)
        split_index = int(len(indicators_df) * 0.8)
        self.X_train = self.features.iloc[:split_index]
        self.y_train_buy = self.labels_buy.iloc[:split_index]
        self.y_train_sell = self.labels_sell.iloc[:split_index]
        self.X_test = self.features.iloc[split_index:]
        self.y_test_buy = self.labels_buy.iloc[split_index:]
        self.y_test_sell = self.labels_sell.iloc[split_index:]

        # Train the models
        self.train_models()

    def train_models(self):
        """
        Trains the regression models for buying and selling.
        """
        # For buying
        self.model_buy = LogisticRegression(max_iter=1000)
        self.model_buy.fit(self.X_train, self.y_train_buy)

        # For selling
        self.model_sell = LogisticRegression(max_iter=1000)
        self.model_sell.fit(self.X_train, self.y_train_sell)

        # Optionally, evaluate the models here and adjust thresholds

    def analyze_buy(self, indicator_values):
        """
        Predicts the probability of a price increase using the buy model.

        Parameters:
        - indicator_values: array-like, the indicator values at the current time step.

        Returns:
        - float: probability between 0 and 1.
        """
        prob = self.model_buy.predict_proba([indicator_values])[0][1]  # Probability of class 1
        return prob

    def analyze_sell(self, indicator_values):
        """
        Predicts the probability of a price decrease using the sell model.

        Parameters:
        - indicator_values: array-like, the indicator values at the current time step.

        Returns:
        - float: probability between 0 and 1.
        """
        prob = self.model_sell.predict_proba([indicator_values])[0][1]  # Probability of class 1
        return prob

    def calculate_profit(self):
        """
        Main method to calculate profit by running the trading simulation.
        """
        try:
            start_time = time.time()
            # Load initial trades
            self.read_next_trade()
            while self.last_price >= 0:
                if self.current_time % 3600000 == 0:
                    # Print status every hour
                    pass

                self.read_next_trade()
                if self.last_price == -1:
                    self.data_loader.clear_variables()
                    return self.write_orders()
                try:
                    self.get_open_positions()
                    # Prepare current indicator values
                    current_time_dt = timestamp_to_datetime(self.current_time)
                    if current_time_dt in self.features.index:
                        indicator_values = self.features.loc[current_time_dt].values
                        # Use analyze_buy and analyze_sell methods
                        buy_prob = self.analyze_buy(indicator_values)
                        sell_prob = self.analyze_sell(indicator_values)

                        # Decision logic based on probabilities
                        if self.current_order['quantity'] == 0 and buy_prob > self.threshold_buy:
                            self.get_available_balance()
                            self.place_order_buy(self.order_qty)
                        elif self.current_order['quantity'] != 0 and sell_prob > self.threshold_sell:
                            self.place_order_sell(abs(self.current_order['quantity']))
                        else:
                            # Existing logic when not using models
                            self.last_support_resistance()
                            if self.last_support > 0 or self.last_resistance > 0:
                                self.fetch_price()
                    else:
                        # If current time not in features index, proceed with existing logic
                        self.last_support_resistance()
                        if self.last_support > 0 or self.last_resistance > 0:
                            self.fetch_price()
                except Exception as e:
                    pass
            self.data_loader.clear_variables()
            return self.write_orders()
        except Exception as e:
            self.data_loader.clear_variables()
            return 0

    # The rest of your existing methods remain unchanged
    # ...
    # Include all methods from your existing code

    def read_next_trade(self):
        self.id_trade += 1
        if self.id_trade == 0:
            month_str = str(self.month).zfill(2)
            trades_file = f"data/BTCUSDT-trades-slim-2022-{month_str}.csv"
            if os.path.exists(trades_file):
                columns = ['id', 'price', 'timestamp']
                trades_df = pd.read_csv(trades_file, names=columns, index_col='id')
                self.trades['price'] = trades_df['price'].tolist()
                self.trades['timestamp'] = trades_df['timestamp'].tolist()
            else:
                self.last_price = -1
                return
        try:
            self.last_price = self.trades['price'][self.id_trade]
            self.current_time = self.trades['timestamp'][self.id_trade]
        except IndexError:
            # End of trades for current month, try next month
            self.month += 1
            month_str = str(self.month).zfill(2)
            trades_file = f"data/BTCUSDT-trades-slim-2022-{month_str}.csv"
            if os.path.exists(trades_file):
                columns = ['id', 'price', 'timestamp']
                trades_df = pd.read_csv(trades_file, names=columns, index_col='id')
                self.trades['price'] = trades_df['price'].tolist()
                self.trades['timestamp'] = trades_df['timestamp'].tolist()
                self.id_trade = 0
                self.last_price = self.trades['price'][self.id_trade]
                self.current_time = self.trades['timestamp'][self.id_trade]
            else:
                self.last_price = -1

    def write_orders(self):
        orders_df = pd.DataFrame(self.orders)
        orders_filename = f"orders/{self.short_long}_{self.rand_number}.csv"
        orders_df.to_csv(orders_filename, header=False, index=False)
        # Additional processing and writing of stats can be added here
        try:
            perc_profit = round((orders_df.iloc[-1][1] - self.initial_balance) / self.initial_balance * 100, 4)
        except Exception:
            perc_profit = 0
        return perc_profit

    def get_available_balance(self):
        sat_balance = self.available_balance
        usd_balance = sat_to_usd(sat_balance, self.last_price)
        self.order_qty = usd_balance / 100 * self.PCT_QTY_ORDER
        self.order_qty = round_to_100(self.order_qty)

    def get_open_positions(self):
        if self.current_order['quantity'] != 0:
            self.unrealized_pnl_pct = round(((self.last_price / self.current_order['entry_price']) * 100) - 100, 2)
            if self.short_long == 's':
                self.unrealized_pnl_pct = -self.unrealized_pnl_pct
            if self.max_unrealized_pnl == 0:
                self.max_unrealized_pnl = self.unrealized_pnl_pct

    def last_support_resistance(self):
        cur_min = int(self.current_time / 1000 / (60 * 5))
        if cur_min == self.old_min_sr:
            return
        self.old_min_sr = cur_min
        row = self.calc_sr.loc[self.calc_sr['close_time'] <= self.current_time].tail(1)
        if not row.empty:
            self.last_support = row['support'].values[0]
            self.last_resistance = row['resistance'].values[0]

    def fetch_price(self):
        self.get_available_balance()
        if (self.last_price < self.last_support - (self.last_support * self.treshold_price) and
                self.last_support > 0 and self.short_long == 's' and
                self.last_support != self.last_order_price):
            # Enter a short position
            self.place_order_sell(self.order_qty)
            self.last_order_price = self.last_support
        elif (self.last_price > self.last_resistance + (self.last_resistance * self.treshold_price) and
              self.last_resistance > 0 and self.short_long == 'l' and
              self.last_resistance != self.last_order_price):
            # Enter a long position
            self.place_order_buy(self.order_qty)
            self.last_order_price = self.last_resistance

    def place_order_buy(self, qty):
        fees = self.calc_fees(qty)
        if self.current_order['quantity'] == 0:
            self.current_order['quantity'] = qty
            self.current_order['sat_qty'] = usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = self.last_price
            self.current_order['fee_paid'] = fees
            self.current_order['entry_time'] = str(timestamp_to_datetime(self.current_time))
        elif self.current_order['quantity'] < 0:
            # Close short position and record profit/loss
            self.current_order['fee_paid'] += fees
            gain = usd_to_sat(abs(qty), self.last_price) - abs(self.current_order['sat_qty'])
            self.available_balance += gain - self.current_order['fee_paid']
            self.orders.append((
                gain - self.current_order['fee_paid'],
                self.available_balance,
                self.current_order['entry_time'],
                str(timestamp_to_datetime(self.current_time)),
                str(self.current_order['quantity']),
                str(self.current_order['entry_price']),
                str(self.last_price),
                str(self.step_rebuy),
                str(self.unrealized_pnl_pct),
                str(self.min_unrealized_pnl)
            ))
            self.write_orders()
            self.current_order = {
                "quantity": 0,
                "entry_price": 0,
                "sat_qty": 0,
                "fee_paid": 0,
                "entry_time": None,
            }
        elif self.current_order['quantity'] > 0:
            entry_price = self.calc_entry_price(qty)
            self.current_order['quantity'] += qty
            self.current_order['sat_qty'] += usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = entry_price
            self.current_order['fee_paid'] += fees
        self.get_open_positions()

    def place_order_sell(self, qty):
        qty = -qty
        fees = self.calc_fees(abs(qty))
        if self.current_order['quantity'] == 0:
            self.current_order['quantity'] = qty
            self.current_order['sat_qty'] = usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = self.last_price
            self.current_order['fee_paid'] = fees
            self.current_order['entry_time'] = str(timestamp_to_datetime(self.current_time))
        elif self.current_order['quantity'] > 0:
            # Close long position and record profit/loss
            self.current_order['fee_paid'] += fees
            gain = abs(self.current_order['sat_qty']) - usd_to_sat(abs(qty), self.last_price)
            self.available_balance += gain - self.current_order['fee_paid']
            self.orders.append((
                gain - self.current_order['fee_paid'],
                self.available_balance,
                self.current_order['entry_time'],
                str(timestamp_to_datetime(self.current_time)),
                str(self.current_order['quantity']),
                str(self.current_order['entry_price']),
                str(self.last_price),
                str(self.step_rebuy),
                str(self.unrealized_pnl_pct),
                str(self.min_unrealized_pnl),
                str(self.current_order['sat_qty'])
            ))
            self.write_orders()
            self.current_order = {
                "quantity": 0,
                "entry_price": 0,
                "sat_qty": 0,
                "fee_paid": 0,
                "entry_time": None,
            }
        elif self.current_order['quantity'] < 0:
            entry_price = self.calc_entry_price(qty)
            self.current_order['quantity'] += qty
            self.current_order['sat_qty'] += usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = entry_price
            self.current_order['fee_paid'] += fees
        self.get_open_positions()

    def fetch_position_to_close_pct(self):
        self.get_open_positions()
        self.get_available_balance()

        if self.last_order_qty != abs(int(self.current_order['quantity'])):
            self.max_unrealized_pnl = 0
            self.last_order_qty = abs(int(self.current_order['quantity']))

        if self.max_unrealized_pnl < self.unrealized_pnl_pct:
            self.max_unrealized_pnl = self.unrealized_pnl_pct

        if self.min_unrealized_pnl > self.unrealized_pnl_pct:
            self.min_unrealized_pnl = self.unrealized_pnl_pct

        can_close = False
        if self.unrealized_pnl_pct > (self.fees * 4):
            can_close = True

        self.fetch_alpha()
        gamma_stop = -self.alphaneg * self.PCT_VOLUME_DISTANCE_NEGATIVE * (self.step_rebuy / 2)
        gamma_take = self.alpha * self.PCT_VOLUME_DISTANCE

        next_step_rebuy = get_next_rebuy_pct(self.step_rebuy, self.N_REBUY, self.PCT_STOP_LOSS)
        mult_order_qty = self.order_qty * abs(next_step_rebuy) * 10 / 2
        mult_order_qty = round_to_100(mult_order_qty)
        double_rebuy = round_to_100(abs(int(self.current_order['quantity'])) * 1.3)

        # Implement position closing logic based on PNL and strategy

    def calc_entry_price(self, qty):
        entry_price = (abs(self.current_order['quantity']) * self.current_order['entry_price'] +
                       abs(qty) * self.last_price) / (abs(self.current_order['quantity']) + abs(qty))
        return round(entry_price, 2)

    def calc_fees(self, qty):
        fees = self.fees  # Replace with dynamic fee calculation if needed
        return usd_to_sat(qty, self.last_price) * fees / 100

    def fetch_alpha(self):
        cur_min = int(self.current_time / 1000 / 60)
        if cur_min != self.old_min_alpha:
            alphavalues = self.calc_alpha['close_time'].values
            row = self.calc_alpha[ne.evaluate('(alphavalues <= self.current_time)')].tail(1)
            if not row.empty:
                self.average = row.iloc[0, 1]
                self.average_neg = row.iloc[0, 2]
            self.old_min_alpha = cur_min

        if self.last_price > self.average:
            delta = (self.last_price / self.average * 100) - 100
        else:
            delta = (self.average / self.last_price * 100) - 100
        self.alpha = round(delta, 2)

        if self.last_price > self.average_neg:
            delta = (self.last_price / self.average_neg * 100) - 100
        else:
            delta = (self.average_neg / self.last_price * 100) - 100
        self.alphaneg = round(delta, 2)

        self.PCT_STOP_LOSS = -abs(self.alphaneg * self.MULT_NEG_DELTA)
        if self.PCT_STOP_LOSS > self.DEFAULT_PCT_STOP_LOSS:
            self.PCT_STOP_LOSS = self.DEFAULT_PCT_STOP_LOSS

    # Include any additional methods from your existing code
