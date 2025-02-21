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
        from src.utils import normalize_price_vec, normalize_volume_vec, normalize_diff_vec
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.mode = mode
        self.data = price_data.join(indicators, how='inner').dropna().sort_index()
        self.data_values = self.data.values
        self.columns = self.data.columns
        self.close_index = self.columns.get_loc('close')
        self.n_steps = len(self.data_values)
        self.timestamps_list = self.data.index.tolist()

        # Precompute normalized features using vectorized functions
        self.norm_features = np.zeros_like(self.data_values, dtype=np.float32)
        for i, col in enumerate(self.columns):
            if col in ['open', 'high', 'low', 'close']:
                self.norm_features[:, i] = normalize_price_vec(self.data_values[:, i])
            elif col == 'volume':
                self.norm_features[:, i] = normalize_volume_vec(self.data_values[:, i])
            elif col.startswith('VWAP') or col.startswith('VWMA'):
                self.norm_features[:, i] = normalize_diff_vec(self.data_values[:, i])
            elif col.startswith('cluster_'):
                self.norm_features[:, i] = normalize_volume_vec(self.data_values[:, i])
            else:
                self.norm_features[:, i] = self.data_values[:, i]

        self.state_dim = self.data.shape[1] + 5
        self.action_dim = 3
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
        from src.utils import normalize_price_vec, normalize_diff_vec
        step = self.current_step if step is None else step
        row = self.norm_features[step]
        close_price = self.data_values[step, self.close_index]
        norm_adjusted_buy = normalize_price_vec(np.array([close_price * (1 + self.transaction_cost)]))[0]
        norm_adjusted_sell = normalize_price_vec(np.array([close_price / (1 + self.transaction_cost)]))[0]
        norm_gain_loss = normalize_diff_vec(np.array([self.gain_loss], dtype=np.float32), max_diff=10000)[0]
        extra_features = np.array([norm_adjusted_buy, norm_adjusted_sell, self.inventory,
                                   self.cash / self.initial_capital, norm_gain_loss], dtype=np.float32)
        return np.concatenate([row, extra_features])

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