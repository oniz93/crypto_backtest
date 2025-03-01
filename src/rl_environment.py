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

logger = logging.getLogger('GeneticOptimizer')

class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000,
                 transaction_cost=0.005, max_steps=500000):
        from src.utils import normalize_price_vec, normalize_volume_vec, normalize_diff_vec
        self.portfolio_history = [] # Track portfolio value history
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

        # Precompute normalized features
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

        self.state_dim = self.data.shape[1] + 4
        self.action_dim = 3
        self.reset()

    def reset(self):
        self.entry_price = 0.0
        self.buy_fee_per_share = 0.0  # New variable to track buy fees per share
        self.gain_loss = 0.0
        self.total_fees = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        self.last_price = self.data_values[self.current_step][self.close_index]
        return self._get_state()

    def _get_state(self, step=None):
        from src.utils import normalize_price_vec, normalize_diff_vec
        step = self.current_step if step is None else step
        row = self.norm_features[step]
        close_price = self.data_values[step, self.close_index]
        norm_adjusted_buy = normalize_price_vec(np.array([close_price * (1 + self.transaction_cost)]))[0]
        norm_adjusted_sell = normalize_price_vec(np.array([close_price / (1 + self.transaction_cost)]))[0]
        extra_features = np.array([norm_adjusted_buy, norm_adjusted_sell, self.inventory,
                                   self.cash / self.initial_capital], dtype=np.float32)  # Removed norm_gain_loss
        return np.concatenate([row, extra_features])

    def step(self, action):
        current_price = self.data_values[self.current_step][self.close_index]
        portfolio_before = self.cash + self.inventory * current_price
        current_timestamp = self.timestamps_list[self.current_step].strftime('%Y-%m-%d %H:%M:%S')
        penalty = 0.0
        trade_fraction = 0.1
        realized_gain_loss = 0.0

        if self.mode == "long":
            if action == 1:  # Buy (long)
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
                            buy_fee_per_share = transaction_fee / qty
                            if self.inventory > 0:
                                old_value = self.inventory * self.entry_price
                                old_fees = self.inventory * self.buy_fee_per_share
                                new_value = qty * current_price
                                new_fees = transaction_fee
                                total_qty = self.inventory + qty
                                self.entry_price = (old_value + new_value) / total_qty
                                self.buy_fee_per_share = (old_fees + new_fees) / total_qty
                            else:
                                self.entry_price = current_price
                                self.buy_fee_per_share = buy_fee_per_share
                            self.inventory = self.inventory + qty if self.inventory > 0 else qty
                            self.cash -= total_cost
                            logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Buy, Spent={total_cost:.2f}, "
                                        f"Qty={qty:.6f}, Fee={transaction_fee:.2f}, Buy Fee/Share={self.buy_fee_per_share:.6f}, "
                                        f"New Cash={self.cash:.2f}")

            elif action == 2:  # Sell (long)
                if self.inventory <= 0:
                    penalty = 0.02 * portfolio_before
                else:
                    proceeds = self.inventory * current_price
                    transaction_fee = proceeds * self.transaction_cost
                    self.total_fees += transaction_fee
                    proceeds_after_fee = proceeds - transaction_fee
                    cost_basis = self.inventory * (self.entry_price + self.buy_fee_per_share)
                    realized_gain_loss = proceeds_after_fee - cost_basis
                    self.cash += proceeds_after_fee
                    self.inventory = 0.0
                    self.entry_price = 0.0
                    self.buy_fee_per_share = 0.0
                    logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Sell, Proceeds={proceeds:.2f}, "
                                f"Fee={transaction_fee:.2f}, Net Proceeds={proceeds_after_fee:.2f}, "
                                f"Gain/Loss={realized_gain_loss:.2f}, New Cash={self.cash:.2f}")

        elif self.mode == "short":
            if action == 1:  # Sell (open short)
                if self.inventory > 0:
                    penalty = 0.02 * portfolio_before
                else:
                    sell_usd = trade_fraction * portfolio_before  # Use portfolio value since shorting borrows
                    if sell_usd < 100:
                        penalty = 0.01 * portfolio_before
                    else:
                        qty = sell_usd / current_price  # Quantity to short
                        transaction_fee = sell_usd * self.transaction_cost
                        self.total_fees += transaction_fee
                        proceeds = sell_usd - transaction_fee
                        if self.inventory < 0:  # Already short
                            old_value = abs(self.inventory) * self.entry_price
                            old_fees = abs(self.inventory) * self.buy_fee_per_share
                            new_value = qty * current_price
                            new_fees = transaction_fee
                            total_qty = abs(self.inventory) + qty
                            self.entry_price = (old_value + new_value) / total_qty
                            self.buy_fee_per_share = (old_fees + new_fees) / total_qty
                            self.inventory -= qty
                        else:
                            self.entry_price = current_price
                            self.buy_fee_per_share = transaction_fee / qty
                            self.inventory = -qty  # Negative inventory for short
                        self.cash += proceeds  # Receive proceeds from short sale
                        logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Short Sell, Proceeds={proceeds:.2f}, "
                                    f"Qty={qty:.6f}, Fee={transaction_fee:.2f}, Short Fee/Share={self.buy_fee_per_share:.6f}, "
                                    f"New Cash={self.cash:.2f}")

            elif action == 2:  # Buy (close short)
                if self.inventory >= 0:
                    penalty = 0.02 * portfolio_before
                else:
                    qty_to_cover = abs(self.inventory)  # Cover entire short position
                    cost = qty_to_cover * current_price
                    transaction_fee = cost * self.transaction_cost
                    self.total_fees += transaction_fee
                    total_cost = cost + transaction_fee
                    proceeds_from_short = qty_to_cover * self.entry_price  # Original short sale proceeds adjusted
                    realized_gain_loss = proceeds_from_short - total_cost  # Profit if entry > current
                    self.cash -= total_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0
                    self.buy_fee_per_share = 0.0
                    logger.debug(f"[{current_timestamp}] Step {self.current_step}: Action=Buy to Cover, Cost={total_cost:.2f}, "
                                f"Fee={transaction_fee:.2f}, Gain/Loss={realized_gain_loss:.2f}, New Cash={self.cash:.2f}")

        # Rest of the code remains largely the same
        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps
        new_price = self.data_values[next_step][self.close_index] if not done else current_price
        portfolio_after = self.cash + self.inventory * new_price

        # Base reward
        reward = portfolio_after - portfolio_before - penalty

        # Update gain_loss for both modes
        if self.inventory > 0:  # Long position
            self.gain_loss = self.inventory * (new_price - self.entry_price) - self.inventory * self.buy_fee_per_share - self.total_fees
        elif self.inventory < 0:  # Short position
            self.gain_loss = abs(self.inventory) * (self.entry_price - new_price) - abs(self.inventory) * self.buy_fee_per_share - self.total_fees
        else:
            self.gain_loss = 0.0

        # Penalize for excessive trading
        if action in [1, 2]:
            reward -= 0.001 * portfolio_before

        # Bonus for sustained profitability
        if self.current_step > 100:
            recent_portfolio_values = self.portfolio_history[-100:]
            if all(val > self.initial_capital for val in recent_portfolio_values):
                reward += 0.01 * portfolio_before

        # Risk-adjusted reward
        if portfolio_after < 0.9 * self.initial_capital:
            reward -= 0.05 * portfolio_before


        is_sell = 0
        # Enhanced reward shaping for profitable trades
        if action == 2 and self.inventory == 0.0:
            if realized_gain_loss > 0:
                scale = 1000.0
                bonus = min(np.exp(realized_gain_loss / scale) - 1, 1000.0) * 100
                reward += bonus
                is_sell = 1
                logger.debug(f"[{current_timestamp}] ---PROFITABLE TRADE--- Gain={realized_gain_loss:.2f}, Bonus={bonus:.2f}")
            elif realized_gain_loss < 0:
                penalty = realized_gain_loss * 1.5
                reward += penalty
                is_sell = -1
                logger.debug(f"[{current_timestamp}] Unprofitable trade. Gain={realized_gain_loss:.2f}, Penalty={penalty:.2f}")

        # Balance stop condition
        balance_threshold = 0.5 * self.initial_capital
        if portfolio_after < balance_threshold and not done:
            remaining_steps = self.n_steps - next_step
            total_steps = self.n_steps
            penalty_factor = remaining_steps / total_steps
            reward *= (1 + penalty_factor)
            done = True
            logger.info(f"[{current_timestamp}] Stopped at Step {next_step}: Balance {portfolio_after:.2f} < {balance_threshold:.2f}, "
                        f"Adjusted reward: {reward:.2f} (factor: {penalty_factor:.2f})")

        if done and next_step >= self.n_steps:
            net_profit = portfolio_after - self.initial_capital
            reward = net_profit
            if net_profit > 0:
                reward += 5000
            elif portfolio_after < 0.75 * self.initial_capital:
                reward -= 10000

        next_state = self._get_state(next_step) if not done else np.zeros(self.state_dim)
        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.info(f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after:.2f} - Done: {done}")

        self.portfolio_history.append(portfolio_after)

        return next_state, reward, done, {"n_step": next_step, "is_sell": is_sell, "gain_loss": realized_gain_loss}