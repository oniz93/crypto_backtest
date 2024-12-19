# src/rl_environment.py

import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000, transaction_cost=0.001, max_steps=100000000):
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

        # Define action_dim and state_dim
        self.action_dim = 3  # 0: hold, 1: buy, 2: sell
        self.state_dim = self.data.shape[1] + 3  # indicators + inventory + cash_ratio + entry_price

        self.entry_price = 0.0  # Track price at which current position was initiated
        self.reset()

    def reset(self):
        self.entry_price = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        self.last_price = self.data['close'].iloc[self.current_step]
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step].values
        # State: indicators + [inventory, cash_ratio, entry_price]
        return np.concatenate([row, [self.inventory, self.cash / self.initial_capital, self.entry_price]])

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_before = self.cash + self.inventory * current_price

        # Mode-specific actions
        if self.mode == "long":
            if action == 1:  # Buy
                if self.inventory == 0:
                    qty = self.cash / current_price
                    qty_after_cost = qty * (1 - self.transaction_cost)
                    self.inventory = qty_after_cost
                    self.cash = 0.0
                    self.entry_price = current_price
            elif action == 2:  # Sell (close long)
                if self.inventory > 0:
                    proceeds = self.inventory * current_price
                    proceeds_after_cost = proceeds * (1 - self.transaction_cost)
                    self.cash += proceeds_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0

        elif self.mode == "short":
            if action == 1:  # Sell to go short
                if self.inventory == 0:
                    qty = self.cash / current_price
                    qty_after_cost = qty * (1 - self.transaction_cost)
                    self.inventory = -qty_after_cost  # Negative inventory indicates a short position
                    self.entry_price = current_price
            elif action == 2:  # Buy to cover short
                if self.inventory < 0:
                    cost_to_cover = abs(self.inventory) * current_price
                    cost_after_cost = cost_to_cover * (1 + self.transaction_cost)
                    self.cash -= cost_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0

        self.current_step += 1
        done = self.current_step >= self.n_steps or self.current_step >= self.max_steps
        new_price = self.data['close'].iloc[self.current_step] if not done else current_price
        portfolio_after = self.cash + self.inventory * new_price
        reward = portfolio_after - portfolio_before

        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        return next_state, reward, done, {}
