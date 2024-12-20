# src/rl_environment.py

import logging

import numpy as np

# Configure logger (ensure this matches the logger in genetic_optimizer.py)
logger = logging.getLogger('GeneticOptimizer')


class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000, transaction_cost=0.005, max_steps=1000000):
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
        logger.debug(f"Environment reset. Starting at step {self.current_step}.")
        return self._get_state()

    def _get_state(self, step=None):
        if step is None:
            step = self.current_step
        if step < 0 or step >= self.n_steps:
            logger.error(f"Attempted to access step {step}, which is out of bounds.")
            raise IndexError("Step is out of bounds in _get_state.")
        row = self.data.iloc[step].values
        # State: indicators + [inventory, cash_ratio, entry_price]
        return np.concatenate([row, [self.inventory, self.cash / self.initial_capital, self.entry_price]])

    def step(self, action):
        # Access current price
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_before = self.cash + self.inventory * current_price

        # Mode-specific actions
        if self.mode == "long":
            if action == 1:  # Buy
                if self.inventory == 0 and self.cash > 0:
                    qty = self.cash / current_price
                    qty_after_cost = qty * (1 - self.transaction_cost)
                    self.inventory = qty_after_cost
                    self.cash = 0.0
                    self.entry_price = current_price
                    logger.debug(f"Action Buy executed. Inventory: {self.inventory}, Entry Price: {self.entry_price}")
            elif action == 2:  # Sell (close long)
                if self.inventory > 0:
                    proceeds = self.inventory * current_price
                    proceeds_after_cost = proceeds * (1 - self.transaction_cost)
                    self.cash += proceeds_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0
                    logger.debug(f"Action Sell executed. Cash: {self.cash}, Close Price: {current_price}")
        elif self.mode == "short":
            if action == 1:  # Sell to go short
                if self.inventory == 0 and self.cash > 0:
                    qty = self.cash / current_price
                    qty_after_cost = qty * (1 - self.transaction_cost)
                    self.inventory = -qty_after_cost  # Negative inventory indicates a short position
                    self.entry_price = current_price
                    logger.debug(f"Action Short Sell executed. Inventory: {self.inventory}, Entry Price: {self.entry_price}")
            elif action == 2:  # Buy to cover short
                if self.inventory < 0:
                    cost_to_cover = abs(self.inventory) * current_price
                    cost_after_cost = cost_to_cover * (1 + self.transaction_cost)
                    self.cash -= cost_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0
                    logger.debug(f"Action Cover Short executed. Cash: {self.cash}, Close Price: {current_price}")

        # Determine the next step
        next_step = self.current_step + 1

        # Check if the next step is out-of-bounds or if balance and inventory are zero
        done = next_step >= self.n_steps or next_step >= self.max_steps or (self.cash == 0 and self.inventory == 0)

        if not done:
            # Access new price if not done
            try:
                new_price = self.data['close'].iloc[next_step]
            except IndexError:
                logger.error(f"Attempted to access step {next_step}, which is out of bounds.")
                done = True
                new_price = self.data['close'].iloc[self.current_step]
            portfolio_after = self.cash + self.inventory * new_price
            reward = portfolio_after - portfolio_before
            try:
                next_state = self._get_state(next_step)
            except IndexError:
                logger.error(f"Failed to get next state for step {next_step}. Setting to zeros.")
                next_state = np.zeros(self.state_dim)
        else:
            if self.cash == 0 and self.inventory == 0:
                logger.info("Balance is 0 and inventory is 0. Stopping training.")
            # If done, use the last valid price and set next_state to zeros
            new_price = self.data['close'].iloc[self.current_step]
            portfolio_after = self.cash + self.inventory * new_price
            reward = portfolio_after - portfolio_before
            next_state = np.zeros(self.state_dim)
            if next_step >= self.n_steps or next_step >= self.max_steps:
                logger.debug("Reached end of data. Setting next_state to zeros.")
            elif self.cash == 0 and self.inventory == 0:
                logger.debug("No more operations possible. Setting next_state to zeros.")

        # Update the current step
        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.debug(f"Step: {self.current_step} - Balance: {portfolio_after} - Done: {done}")

        return next_state, reward, done, {}
