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

        self.entry_price = 0.0
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: The initial state vector.
        """
        self.entry_price = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        # Set the current price from the first row.
        self.last_price = self.data_values[self.current_step][self.close_index]
        return self._get_state()

    def _get_state(self, step=None):
        """
        Build the state vector for the given timestep.

        The state vector includes:
          - Normalized market features (e.g., price, volume).
          - Extra features: normalized adjusted buy price, normalized adjusted sell price,
            current inventory, cash ratio, and the normalized difference between current price and entry price.

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
        # Calculate the difference from the entry price.
        if self.entry_price == 0.0:
            norm_entry_diff = 0.0
        else:
            entry_diff = self.entry_price - close_price
            norm_entry_diff = normalize_diff(entry_diff)
        # Extra features include these calculated values.
        extra_features = [norm_adjusted_buy, norm_adjusted_sell, self.inventory,
                          self.cash / self.initial_capital, norm_entry_diff]
        # Return the concatenated state vector.
        return np.concatenate([np.array(norm_features), np.array(extra_features)])

    def step(self, action):
        """
        Execute an action (buy, sell, or hold) and advance one timestep.

        Parameters:
            action (int): 0 (hold), 1 (buy), or 2 (sell).

        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: The state after the action.
                - reward: The reward obtained.
                - done (bool): Whether the simulation has finished.
                - info (dict): Additional info such as the current step.
        """
        # Get the current price and compute the portfolio value before the action
        current_price = self.data_values[self.current_step][self.close_index]
        portfolio_before = self.cash + self.inventory * current_price
        current_timestamp = self.timestamps_list[self.current_step].strftime('%Y-%m-%d %H:%M:%S')
        penalty = 0.0  # Initialize a penalty variable
        trade_fraction = 0.1  # Fraction of cash to use per trade

        # Log invalid actions
        if (self.inventory == 0 and action == 2) or (self.inventory != 0 and action == 1):
            logger.debug(f"Step {self.current_step}: Action={action}, Inventory={self.inventory}, Cash={self.cash}")

        if action == 1:  # Buy action
            if self.inventory < 0:
                # Cannot buy if in a short position; apply penalty
                penalty = 0.02 * portfolio_before
            else:
                buy_usd = trade_fraction * self.cash
                if buy_usd < 100:
                    penalty = 0.01 * portfolio_before  # Not enough cash; small penalty
                else:
                    # Total cost includes transaction fee (e.g., 0.05% of buy_usd)
                    transaction_fee = buy_usd * self.transaction_cost
                    total_cost = buy_usd + transaction_fee  # What we actually spend
                    if total_cost > self.cash:
                        penalty = 0.01 * portfolio_before  # Insufficient funds after fees
                    else:
                        # Quantity bought is based on the base amount before fees
                        qty = buy_usd / current_price
                        if self.inventory > 0:
                            # Update the weighted average entry price
                            old_value = self.inventory * self.entry_price
                            new_value = qty * current_price
                            total_qty = self.inventory + qty
                            self.entry_price = (old_value + new_value) / total_qty
                            self.inventory = total_qty
                        else:
                            self.entry_price = current_price
                            self.inventory = qty
                        self.cash -= total_cost  # Deduct total cost including fees
                        # logger.debug(f"Step {self.current_step}: Action=Buy, Spent={total_cost:.2f}, "
                        #             f"Qty={qty:.6f}, Fee={transaction_fee:.2f}, New Cash={self.cash:.2f}")

        elif action == 2:  # Sell action (to close a long position)
            if self.inventory <= 0:
                penalty = 0.02 * portfolio_before
            else:
                # Proceeds before fees
                proceeds = self.inventory * current_price
                # Transaction fee is 0.05% of the proceeds
                transaction_fee = proceeds * self.transaction_cost
                proceeds_after_fee = proceeds - transaction_fee  # What we actually receive
                # Calculate realized gain/loss including fees
                cost_basis = self.inventory * self.entry_price  # Original cost of inventory
                realized_gain_loss = proceeds_after_fee - cost_basis
                self.cash += proceeds_after_fee
                self.inventory = 0.0
                self.entry_price = 0.0
                logger.debug(f"Step {self.current_step}: Action=Sell, Proceeds={proceeds:.2f}, "
                            f"Fee={transaction_fee:.2f}, Net Proceeds={proceeds_after_fee:.2f}, "
                            f"Gain/Loss={realized_gain_loss:.2f}, New Cash={self.cash:.2f}")

        # Advance to the next timestep
        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps
        if not done:
            try:
                new_price = self.data_values[next_step][self.close_index]
            except IndexError:
                logger.error(f"Attempted to access step {next_step}, which is out of bounds.")
                new_price = current_price
                done = True
        else:
            new_price = current_price

        portfolio_after = self.cash + self.inventory * new_price
        # Reward is the change in portfolio value minus any penalty
        raw_reward = (portfolio_after - portfolio_before) - penalty
        normalized_reward = raw_reward / self.initial_capital

        # Get the next state
        if not done:
            try:
                next_state = self._get_state(next_step)
            except IndexError:
                logger.error(f"Failed to get next state for step {next_step}. Setting state to zeros.")
                next_state = np.zeros(self.state_dim)
        else:
            next_state = np.zeros(self.state_dim)

        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.info(f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after:.2f} - Done: {done}")

        # If portfolio value falls below 75% of initial capital, end the simulation
        if portfolio_after < 0.75 * self.initial_capital:
            logger.warning(f"Portfolio value below 75% initial: {portfolio_after:.2f}")
            done = True
            raw_reward -= 0.05 * portfolio_before
            normalized_reward = raw_reward / self.initial_capital

        reward = np.clip(normalized_reward, -1, 1)
        if done and portfolio_after > self.initial_capital:
            reward = min(reward * 10, 1)

        return next_state, reward, done, {"n_step": next_step}
