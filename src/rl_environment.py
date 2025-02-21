"""
rl_environment.py
-----------------
This module defines the TradingEnvironment class which simulates trading.
It uses price data and technical indicators to simulate trading actions (buy, sell, hold)
and computes rewards based on changes in portfolio value.
It is designed to be used by the RL agent.
"""

import logging
import numpy as np

# Use the GeneticOptimizer logger for consistency.
logger = logging.getLogger('GeneticOptimizer')

class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000,
                 transaction_cost=0.005, max_steps=500000):
        """
        Initialize the trading environment.

        Parameters:
            price_data (pd.DataFrame): DataFrame containing price data.
            indicators (pd.DataFrame): DataFrame containing technical indicators.
            mode (str): Trading mode ('long' or 'short').
            initial_capital (float): Starting capital.
            transaction_cost (float): Fractional transaction fee.
            max_steps (int): Maximum number of timesteps to simulate.
        """
        self.price_data = price_data
        self.indicators = indicators
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.mode = mode

        # Merge price data and indicator data on their timestamps.
        self.data = price_data.join(indicators, how='inner').dropna()
        self.data = self.data.sort_index()

        # Convert the DataFrame to a NumPy array for faster indexing.
        self.data_values = self.data.values
        self.columns = self.data.columns
        # Find the column index for the closing price.
        self.close_index = self.data.columns.get_loc('close')
        self.n_steps = len(self.data_values)
        # Store the timestamps as a list.
        self.timestamps_list = self.data.index.tolist()

        # Define the dimension of the state vector:
        # all columns from data plus 5 extra features.
        self.state_dim = self.data.shape[1] + 5
        self.action_dim = 3  # Actions: 0 = hold, 1 = buy, 2 = sell

        # Initialize state variables
        self.entry_price = 0.0  # Still needed for position tracking in step
        self.gain_loss = 0.0
        self.total_fees = 0.0# New attribute to track gain/loss
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: The initial state vector.
        """
        self.entry_price = 0.0
        self.gain_loss = 0.0
        self.total_fees = 0.0  # Reset fees
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        self.last_price = self.data_values[self.current_step][self.close_index]
        return self._get_state()

    def _get_state(self, step=None):
        """
        Build the state vector for the given timestep.

        The state vector includes:
          - Normalized market features (e.g., price, volume).
          - Extra features: normalized adjusted buy price, normalized adjusted sell price,
            current inventory, cash ratio, and normalized gain/loss.

        Parameters:
            step (int): Timestep index for which to generate the state.

        Returns:
            np.array: The state vector.
        """
        from src.utils import normalize_price, normalize_volume, normalize_diff
        if step is None:
            step = self.current_step
        if step < 0 or step >= self.n_steps:
            logger.error(f"Attempted to access step {step}, which is out of bounds.")
            raise IndexError("Step is out of bounds in _get_state.")
        # Get the row corresponding to the current timestep.
        row = self.data_values[step]
        norm_features = []
        # Normalize each feature based on its type.
        for col, val in zip(self.data.columns, row):
            if col in ['open', 'high', 'low', 'close']:
                norm_features.append(normalize_price(val))
            elif col == 'volume':
                norm_features.append(normalize_volume(val))
            elif col.startswith('VWAP') or col.startswith('VWMA'):
                norm_features.append(normalize_diff(val))
            elif col.startswith('cluster_'):
                norm_features.append(normalize_volume(val))
            else:
                norm_features.append(val)
        # Get the raw close price for extra calculations.
        close_price = row[self.close_index]
        # Calculate adjusted buy and sell prices (including transaction cost).
        adjusted_buy_price = close_price * (1 + self.transaction_cost)
        adjusted_sell_price = close_price / (1 + self.transaction_cost)
        norm_adjusted_buy = normalize_price(adjusted_buy_price)
        norm_adjusted_sell = normalize_price(adjusted_sell_price)
        # Normalize the gain/loss value (default to 0 if no position).
        norm_gain_loss = normalize_diff(self.gain_loss, 10000)
        # Extra features include these calculated values.
        extra_features = [norm_adjusted_buy, norm_adjusted_sell, self.inventory,
                          self.cash / self.initial_capital, norm_gain_loss]
        # Return the concatenated state vector.
        return np.concatenate([np.array(norm_features), np.array(extra_features)])

    def step(self, action):
        current_price = self.data_values[self.current_step][self.close_index]
        portfolio_before = self.cash + self.inventory * current_price
        current_timestamp = self.timestamps_list[self.current_step].strftime('%Y-%m-%d %H:%M:%S')
        penalty = 0.0
        trade_fraction = 0.1

        if action == 1:  # Buy
            if self.inventory < 0:
                penalty = 0.02 * portfolio_before
            else:
                buy_usd = trade_fraction * self.cash
                if buy_usd < 100:
                    penalty = 0.01 * portfolio_before
                else:
                    transaction_fee = buy_usd * self.transaction_cost
                    self.total_fees += transaction_fee
                    total_cost = buy_usd + transaction_fee
                    if total_cost > self.cash:
                        penalty = 0.01 * portfolio_before
                    else:
                        qty = buy_usd / current_price
                        if self.inventory > 0:
                            old_value = self.inventory * self.entry_price
                            new_value = qty * current_price
                            total_qty = self.inventory + qty
                            self.entry_price = (old_value + new_value) / total_qty
                            self.inventory = total_qty
                        else:
                            self.entry_price = current_price
                            self.inventory = qty
                        self.cash -= total_cost
                        logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Buy, Spent={total_cost:.2f}, "
                                     f"Qty={qty:.6f}, Fee={transaction_fee:.2f}, New Cash={self.cash:.2f}")

        elif action == 2:  # Sell
            if self.inventory <= 0:
                penalty = 0.02 * portfolio_before
            else:
                proceeds = self.inventory * current_price
                transaction_fee = proceeds * self.transaction_cost
                self.total_fees += transaction_fee
                proceeds_after_fee = proceeds - transaction_fee
                cost_basis = self.inventory * self.entry_price
                realized_gain_loss = proceeds_after_fee - cost_basis
                self.cash += proceeds_after_fee
                self.inventory = 0.0
                self.entry_price = 0.0
                logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Sell, Proceeds={proceeds:.2f}, "
                             f"Fee={transaction_fee:.2f}, Net Proceeds={proceeds_after_fee:.2f}, "
                             f"Gain/Loss={realized_gain_loss:.2f}, New Cash={self.cash:.2f}")

        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps
        new_price = self.data_values[next_step][self.close_index] if not done else current_price
        portfolio_after = self.cash + self.inventory * new_price

        # Reward: Change in portfolio value minus penalty
        reward = portfolio_after - portfolio_before - penalty
        self.gain_loss = self.inventory * (new_price - self.entry_price) - self.total_fees if self.inventory > 0 else 0.0

        if done:
            net_profit = portfolio_after - self.initial_capital
            reward = net_profit
            if net_profit > 0:
                reward += 5000
            elif portfolio_after < 0.75 * self.initial_capital:
                reward -= 10000

        next_state = self._get_state(next_step) if not done else np.zeros(self.state_dim)
        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.info(
                f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after:.2f} - Done: {done}")

        return next_state, reward, done, {"n_step": next_step}