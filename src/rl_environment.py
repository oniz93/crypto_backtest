# src/rl_environment.py

import logging
import numpy as np

logger = logging.getLogger('GeneticOptimizer')


class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000,
                 transaction_cost=0.005, max_steps=500000):
        self.price_data = price_data
        self.indicators = indicators
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.mode = mode

        # Merge price_data and indicators
        self.data = price_data.join(indicators, how='inner').dropna()
        self.data = self.data.sort_index()
        self.timestamps = self.data.index
        self.n_steps = len(self.data)

        # We want to maximize total reward from the RL perspective.
        # The GA might invert sign if it wants to minimize.
        self.state_dim = self.data.shape[1] + 5  # indicators + adjusted_buy_price + adjusted_sell_price + inventory + cash_ratio + entry_price
        self.action_dim = 3  # 0=hold, 1=buy, 2=sell

        self.entry_price = 0.0
        self.reset()

    def reset(self):
        self.entry_price = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        self.last_price = self.data['close'].iloc[self.current_step]
        return self._get_state()

    def _get_state(self, step=None):
        if step is None:
            step = self.current_step
        if step < 0 or step >= self.n_steps:
            logger.error(f"Attempted to access step {step}, which is out of bounds.")
            raise IndexError("Step is out of bounds in _get_state.")

        row = self.data.iloc[step].values
        close_price = self.data['close'].iloc[step]

        # Adjusted prices if we decide to buy or sell
        adjusted_buy_price = close_price * (1 + self.transaction_cost)
        adjusted_sell_price = close_price / (1 + self.transaction_cost)

        return np.concatenate([
            row,
            [adjusted_buy_price, adjusted_sell_price, self.inventory, self.cash / self.initial_capital, self.entry_price]
        ])

    def step(self, action):
        """
        We'll keep the logic for multi-buys. The environment yields a reward
        equal to portfolio_after - portfolio_before, which an RL agent tries
        to maximize. The GA will handle sign if needed.
        """
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_before = self.cash + self.inventory * current_price

        current_timestamp = self.timestamps[self.current_step].strftime('%Y-%m-%d %H:%M:%S')

        reward = 0.0

        if self.mode == "long":
            if action == 1:  # buy
                if self.inventory < 0:
                    reward -= 1000.0
                else:
                    buy_usd = 0.1 * self.cash
                    if buy_usd < 100:
                        reward -= 500.0
                    else:
                        qty = (buy_usd / current_price) * (1 - self.transaction_cost)
                        if self.inventory > 0:
                            old_value = self.inventory * self.entry_price
                            new_value = qty * current_price
                            total_qty = self.inventory + qty
                            new_entry_price = (old_value + new_value) / total_qty
                            self.entry_price = new_entry_price
                            self.inventory = total_qty
                        else:
                            self.entry_price = current_price
                            self.inventory += qty
                        self.cash -= buy_usd

            elif action == 2:  # sell (close all)
                if self.inventory <= 0:
                    reward -= 500.0
                else:
                    proceeds = self.inventory * current_price
                    proceeds_after_cost = proceeds * (1 - self.transaction_cost)
                    self.cash += proceeds_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0

        elif self.mode == "short":
            if action == 1:  # open or add short
                if self.inventory > 0:
                    reward -= 1000.0
                else:
                    sell_usd = 0.1 * self.cash
                    if sell_usd < 100:
                        reward -= 500.0
                    else:
                        qty = (sell_usd / current_price) * (1 - self.transaction_cost)
                        if self.inventory < 0:
                            old_value = abs(self.inventory) * self.entry_price
                            new_value = qty * current_price
                            total_qty = abs(self.inventory) + qty
                            new_entry_price = (old_value + new_value) / total_qty
                            self.entry_price = new_entry_price
                            self.inventory -= qty
                        else:
                            self.entry_price = current_price
                            self.inventory -= qty
                        self.cash -= sell_usd

            elif action == 2:  # buy to cover short fully
                if self.inventory >= 0:
                    reward -= 500.0
                else:
                    cost_to_cover = abs(self.inventory) * current_price
                    cost_after_cost = cost_to_cover * (1 + self.transaction_cost)
                    if cost_after_cost > self.cash:
                        reward -= 1000.0
                    else:
                        self.cash -= cost_after_cost
                        self.inventory = 0.0
                        self.entry_price = 0.0

        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps

        # portfolio delta
        if not done:
            try:
                new_price = self.data['close'].iloc[next_step]
            except IndexError:
                logger.error(f"Attempted to access step {next_step}, which is out of bounds.")
                done = True
                new_price = self.data['close'].iloc[self.current_step]
        else:
            new_price = current_price

        portfolio_after = self.cash + self.inventory * new_price
        reward += (portfolio_after - portfolio_before)

        if not done:
            try:
                next_state = self._get_state(next_step)
            except IndexError:
                logger.error(f"Failed to get next state for step {next_step}. Setting to zeros.")
                next_state = np.zeros(self.state_dim)
        else:
            next_state = np.zeros(self.state_dim)

        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.debug(f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after} - Done: {done}")

        if portfolio_after < self.initial_capital * 0.5:
            logger.warning(f"Portfolio value below 50% initial: {portfolio_after}")
        if portfolio_after < self.initial_capital * 0.3:
            done = True

        return next_state, reward, done, {}
