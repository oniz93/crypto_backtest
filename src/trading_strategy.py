"""
trading_strategy.py
-------------------
This module implements a TradingStrategy class that simulates trading operations.
It uses technical indicators to determine buy, sell, or hold signals,
and simulates the trading process by interacting with market data.
"""

import os
import random
import time
import numexpr as ne
import pandas as pd
from src.config_loader import Config
from src.data_loader import DataLoader
from src.utils import (sat_to_usd, usd_to_sat, round_to_100, get_next_rebuy_pct,
                       timestamp_to_datetime)

class TradingStrategy:
    def __init__(self, data_loader: DataLoader, config, model_buy, model_sell):
        """
        Initialize the TradingStrategy.

        Parameters:
            data_loader (DataLoader): Loads market data.
            config (Config): Configuration settings.
            model_buy, model_sell: Models (or arrays) for buy/sell decision making.
        """
        self.data_loader = data_loader
        self.config = config
        self.cl = Config()  # Optionally create a separate config instance
        self.model_buy = model_buy
        self.model_sell = model_sell
        self.threshold_buy = config.get('threshold_buy', 0.6)
        self.threshold_sell = config.get('threshold_sell', 0.6)
        # These variables hold aggregated prediction signals.
        self.aggregated_buy = None
        self.aggregated_sell = None
        self.current_step = 0
        # Various trading parameters:
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
        # Current order information.
        self.current_order = {"quantity": 0, "entry_price": 0, "sat_qty": 0, "fee_paid": 0, "entry_time": None}
        self.average = 0
        self.average_neg = 0
        self.old_min_alpha = 0
        self.old_min_sr = 0
        self.old_min_fr = 0
        # Trade history variables.
        self.trades = {'price': [], 'timestamp': []}
        self.id_trade = -1
        self.month = 1
        self.funding_rate = 0
        self.orders = []
        self.step_rebuy = 2
        self.rand_number = random.randint(1, 1234567890)
        self.current_time = 0
        self.current_timestamp = 0
        self.last_price = 0
        # Prepare the features DataFrame by combining price data and indicators.
        self.features = self.prepare_features()

    def prepare_features(self):
        """
        Prepare the feature DataFrame by joining base price data with indicator data.

        Returns:
            pd.DataFrame: Features used for trading decisions.
        """
        base_tf = self.data_loader.base_timeframe
        # Get a copy of the base timeframe price data.
        price_data = self.data_loader.tick_data[base_tf].copy()
        if hasattr(self.data_loader, 'indicators'):
            indicators_df = price_data.copy()
            # Loop over each indicator and merge its 1min DataFrame.
            for indicator_name, tf_dict in self.data_loader.indicators.items():
                if '1min' in tf_dict:
                    df = tf_dict['1min']
                    # Align indicator DataFrame with the price data index.
                    df = df.reindex(indicators_df.index, method='ffill')
                    # Add a suffix so we know which indicator and timeframe the column comes from.
                    df = df.add_suffix(f'_{indicator_name}')
                    # Join the indicator data to the base price data.
                    indicators_df = indicators_df.join(df)
            # Remove rows with missing values.
            indicators_df.dropna(inplace=True)
            # Remove basic price columns to keep only indicators as features.
            features = indicators_df.drop(columns=['open', 'high', 'low', 'close', 'volume',
                                                    'future_return', 'buy_signal', 'sell_signal'], errors='ignore')
            return features
        else:
            return pd.DataFrame()

    def set_aggregated_predictions(self, aggregated_buy, aggregated_sell):
        """
        Set aggregated buy and sell signals.

        Parameters:
            aggregated_buy: Aggregated signal for buying.
            aggregated_sell: Aggregated signal for selling.
        """
        self.aggregated_buy = aggregated_buy
        self.aggregated_sell = aggregated_sell

    def analyze_buy(self, indicator_values):
        """
        Determine whether to buy based on aggregated predictions.

        Parameters:
            indicator_values: Values of indicators at current time.

        Returns:
            float: Buy probability (1.0 if signal is active, else 0.0).
        """
        return 1.0 if self.aggregated_buy[self.current_step] == 1 else 0.0

    def analyze_sell(self, indicator_values):
        """
        Determine whether to sell based on aggregated predictions.

        Parameters:
            indicator_values: Values of indicators at current time.

        Returns:
            float: Sell probability.
        """
        return 1.0 if self.aggregated_sell[self.current_step] == 1 else 0.0

    def calculate_profit(self):
        """
        Run the trading simulation by iterating over trades, applying aggregated predictions,
        and executing orders. Returns the overall profit percentage.

        Returns:
            float: Profit percentage.
        """
        try:
            start_time = time.time()  # Start time for simulation
            self.read_next_trade()  # Load the first trade
            # Loop until there are no more trades (last_price becomes -1)
            while self.last_price >= 0:
                self.read_next_trade()  # Get next trade
                if self.last_price == -1:
                    # No more data; clear resources and write orders.
                    self.data_loader.clear_variables()
                    return self.write_orders()
                try:
                    self.get_open_positions()  # Update current position details
                    if self.current_timestamp in self.features.index:
                        # Map current timestamp to feature index.
                        step_idx = self.features.index.get_loc(self.current_timestamp)
                        self.current_step = step_idx
                        # Get the indicator values for current time.
                        indicator_values = self.features.loc[self.current_timestamp].values
                        # Determine buy and sell probabilities.
                        buy_prob = self.analyze_buy(indicator_values)
                        sell_prob = self.analyze_sell(indicator_values)
                        # If no open position and buy probability is high, execute a buy order.
                        if self.current_order['quantity'] == 0 and buy_prob > self.threshold_buy:
                            self.get_available_balance()
                            self.place_order_buy(self.order_qty)
                        # If a position exists and sell probability is high, execute a sell order.
                        elif self.current_order['quantity'] != 0 and sell_prob > self.threshold_sell:
                            self.place_order_sell(abs(self.current_order['quantity']))
                except Exception as e:
                    # If an error occurs during the trading logic, ignore and continue.
                    pass
            # Clear data and write final orders.
            self.data_loader.clear_variables()
            return self.write_orders()
        except Exception as e:
            self.data_loader.clear_variables()
            return 0

    def read_next_trade(self):
        """
        Read the next trade from stored trade history.
        Updates the current trade index and loads new trade data if needed.
        """
        self.id_trade += 1  # Move to next trade
        if self.id_trade == 0:
            # First trade: try to load a Parquet file for the current month.
            month_str = str(self.month).zfill(2)
            trades_file = f"trades_history/BTCUSDT-trades-aggregated-2023-{month_str}.parquet"
            if os.path.exists(trades_file):
                trades_df = pd.read_parquet(trades_file)
                # Convert columns to lists for faster access.
                self.trades['price'] = trades_df['price'].tolist()
                self.trades['timestamp'] = trades_df['timestamp'].tolist()
            else:
                self.last_price = -1  # Signal end of data
                return
        try:
            # Get the price and timestamp of the current trade.
            self.last_price = self.trades['price'][self.id_trade]
            self.current_timestamp = self.trades['timestamp'][self.id_trade]
            # Convert the timestamp to an integer (epoch seconds).
            self.current_time = int(self.trades['timestamp'][self.id_trade].timestamp())
        except IndexError:
            # If there are no more trades, try loading data from the next month.
            self.month += 1
            month_str = str(self.month).zfill(2)
            trades_file = f"data/BTCUSDT-trades-aggregated-2023-{month_str}.csv"
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
        """
        Write executed orders to a CSV file and compute the profit percentage.

        Returns:
            float: Profit percentage.
        """
        orders_df = pd.DataFrame(self.orders)
        orders_filename = f"orders/{self.short_long}_{self.rand_number}.csv"
        orders_df.to_csv(orders_filename, header=False, index=False)
        try:
            # Calculate profit percentage using the final balance.
            perc_profit = round((orders_df.iloc[-1][1] - self.initial_balance) / self.initial_balance * 100, 4)
        except Exception:
            perc_profit = 0
        return perc_profit

    def get_available_balance(self):
        """
        Calculate the available balance and determine the order quantity based on the current price.
        """
        sat_balance = self.available_balance
        usd_balance = sat_to_usd(sat_balance, self.last_price)
        # Use a percentage from the config to decide how much to order.
        PCT_QTY_ORDER = self.config.get('PCT_QTY_ORDER', 10)
        self.order_qty = usd_balance / 100 * PCT_QTY_ORDER
        # Round the order quantity to the nearest 100.
        self.order_qty = round_to_100(self.order_qty)

    def get_open_positions(self):
        """
        Update the unrealized profit/loss percentage for any open positions.
        """
        if self.current_order['quantity'] != 0:
            # Calculate profit percentage.
            self.unrealized_pnl_pct = round(((self.last_price / self.current_order['entry_price']) * 100) - 100, 2)
            if self.short_long == 's':
                self.unrealized_pnl_pct = -self.unrealized_pnl_pct
            if self.max_unrealized_pnl == 0:
                self.max_unrealized_pnl = self.unrealized_pnl_pct

    def last_support_resistance(self):
        """
        Update support and resistance levels based on the current time.
        """
        # Compute a time index value (e.g. every 5 minutes).
        cur_min = int(self.current_time / 1000 / (60 * 5))
        if cur_min == self.old_min_sr:
            return  # No update needed if we're still in the same window.
        self.old_min_sr = cur_min
        # Get the last row of support/resistance data up to current time.
        row = self.calc_sr.loc[self.calc_sr['close_time'] <= self.current_time].tail(1)
        if not row.empty:
            self.last_support = row['support'].values[0]
            self.last_resistance = row['resistance'].values[0]

    def fetch_price(self):
        """
        Fetch a new price from the current trade and trigger a buy or sell order if conditions are met.
        """
        self.get_available_balance()
        # If price is lower than support by a threshold in short mode, sell.
        if (self.last_price < self.last_support - (self.last_support * self.treshold_price) and
                self.last_support > 0 and self.short_long == 's' and
                self.last_support != self.last_order_price):
            self.place_order_sell(self.order_qty)
            self.last_order_price = self.last_support
        # If price is higher than resistance by a threshold in long mode, buy.
        elif (self.last_price > self.last_resistance + (self.last_resistance * self.treshold_price) and
              self.last_resistance > 0 and self.short_long == 'l' and
              self.last_resistance != self.last_order_price):
            self.place_order_buy(self.order_qty)
            self.last_order_price = self.last_resistance

    def place_order_buy(self, qty):
        """
        Execute a buy order with the specified quantity.

        Parameters:
            qty (float): Order quantity.
        """
        fees = self.calc_fees(qty)
        # If no current position, start a new long position.
        if self.current_order['quantity'] == 0:
            self.current_order['quantity'] = qty
            self.current_order['sat_qty'] = usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = self.last_price
            self.current_order['fee_paid'] = fees
            self.current_order['entry_time'] = str(timestamp_to_datetime(self.current_time))
        elif self.current_order['quantity'] < 0:
            # If currently in a short position, close it.
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
            # Reset the current order after closing.
            self.current_order = {"quantity": 0, "entry_price": 0, "sat_qty": 0, "fee_paid": 0, "entry_time": None}
        elif self.current_order['quantity'] > 0:
            # If already in a long position, adjust the weighted average entry price.
            entry_price = self.calc_entry_price(qty)
            self.current_order['quantity'] += qty
            self.current_order['sat_qty'] += usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = entry_price
            self.current_order['fee_paid'] += fees
        self.get_open_positions()

    def place_order_sell(self, qty):
        """
        Execute a sell order with the specified quantity.

        Parameters:
            qty (float): Order quantity.
        """
        qty = -qty  # Sell orders are represented with negative quantity.
        fees = self.calc_fees(abs(qty))
        if self.current_order['quantity'] == 0:
            # If no current position, start a new short position.
            self.current_order['quantity'] = qty
            self.current_order['sat_qty'] = usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = self.last_price
            self.current_order['fee_paid'] = fees
            self.current_order['entry_time'] = str(timestamp_to_datetime(self.current_time))
        elif self.current_order['quantity'] > 0:
            # If in a long position, close it by selling.
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
            self.current_order = {"quantity": 0, "entry_price": 0, "sat_qty": 0, "fee_paid": 0, "entry_time": None}
        elif self.current_order['quantity'] < 0:
            # If already in a short position, update the average entry price.
            entry_price = self.calc_entry_price(qty)
            self.current_order['quantity'] += qty
            self.current_order['sat_qty'] += usd_to_sat(qty, self.last_price)
            self.current_order['entry_price'] = entry_price
            self.current_order['fee_paid'] += fees
        self.get_open_positions()

    def fetch_position_to_close_pct(self):
        """
        (Placeholder) Determine the percentage of a position to close based on profit/loss.
        """
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
        # Additional closing logic would be implemented here.

    def calc_entry_price(self, qty):
        """
        Calculate a new weighted average entry price when adding to a position.

        Parameters:
            qty (float): Additional quantity.

        Returns:
            float: New average entry price.
        """
        entry_price = (abs(self.current_order['quantity']) * self.current_order['entry_price'] +
                       abs(qty) * self.last_price) / (abs(self.current_order['quantity']) + abs(qty))
        return round(entry_price, 2)

    def calc_fees(self, qty):
        """
        Calculate trading fees based on order quantity and current price.

        Parameters:
            qty (float): Order quantity.

        Returns:
            float: Calculated fee.
        """
        fees = self.fees  # Could be modified for dynamic fee calculation.
        return usd_to_sat(qty, self.last_price) * fees / 100

    def fetch_alpha(self):
        """
        Compute an 'alpha' metric based on historical price changes.
        This function is a placeholder and may require further implementation.
        """
        cur_min = int(self.current_time / 1000 / 60)
        if cur_min != self.old_min_alpha:
            alphavalues = self.calc_alpha['close_time'].values
            row = self.calc_alpha[ne.evaluate('(alphavalues <= self.current_time)')].tail(1)
            if not row.empty:
                self.average = row.iloc[0, 1]
                self.average_neg = row.iloc[0, 2]
            self.old_min_alpha = cur_min
        # Calculate a percentage change as the alpha value.
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
