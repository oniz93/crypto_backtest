# src/rl_environment.py

import logging

import numpy as np

# Configure logger (ensure this matches the logger in genetic_optimizer.py)
logger = logging.getLogger('GeneticOptimizer')


class TradingEnvironment:
    def __init__(self, price_data, indicators, mode="long", initial_capital=100000, transaction_cost=0.005, max_steps=500000):
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
        # Now we have original data features + adjusted_buy_price + adjusted_sell_price + inventory + cash_ratio + entry_price
        # Originally it was data.shape[1] + 3, now we add 2 more features, so +5
        # But let's just keep data.shape[1] the same and add 5.
        # Actually, we must reflect this in state_dim:
        self.state_dim = self.data.shape[1] + 5  # indicators + adjusted_buy_price + adjusted_sell_price + inventory + cash_ratio + entry_price
        self.action_dim = 3  # 0: hold, 1: buy, 2: sell

        self.entry_price = 0.0  # Track price at which current position was initiated
        self.reset()

    def reset(self):
        self.entry_price = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        self.last_price = self.data['close'].iloc[self.current_step]
        # logger.debug(f"Environment reset. Starting at step {self.current_step}.")
        return self._get_state()

    def _get_state(self, step=None):
        if step is None:
            step = self.current_step
        if step < 0 or step >= self.n_steps:
            logger.error(f"Attempted to access step {step}, which is out of bounds.")
            raise IndexError("Step is out of bounds in _get_state.")
        
        row = self.data.iloc[step].values
        close_price = self.data['close'].iloc[step]

        # Calculate adjusted buy and sell prices
        adjusted_buy_price = close_price * (1 + self.transaction_cost)
        adjusted_sell_price = close_price / (1 + self.transaction_cost)

        # State: indicators (row) + adjusted_buy_price + adjusted_sell_price + inventory + cash_ratio + entry_price
        return np.concatenate([row, [adjusted_buy_price, adjusted_sell_price, self.inventory, self.cash / self.initial_capital, self.entry_price]])

    def step(self, action):
        """
        Extended step function with multiple-buys logic, minimum 100 USD rule,
        weighted-average entry price, and a penalty for multiple sells.
        """
        # Access current price
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_before = self.cash + self.inventory * current_price

        # Retrieve and format current timestamp
        current_timestamp = self.timestamps[self.current_step].strftime('%Y-%m-%d %H:%M:%S')

        # Default reward is the portfolio change; we might adjust further for penalties
        reward = 0.0

        # === Mode: LONG ===
        if self.mode == "long":
            if action == 1:  # Buy
                # If we already have a short position or zero inventory, this might be disallowed.
                # But we want to permit multiple buys if inventory >= 0.
                if self.inventory < 0:
                    # Trying to buy while in a short position is invalid => negative reward
                    reward -= 1000.0  # example penalty
                else:
                    # 10% of current cash in USD
                    buy_usd = 0.1 * self.cash
                    if buy_usd < 100:
                        # Not enough to meet minimum 100 USD => penalty
                        reward -= 500.0  # example small penalty
                    else:
                        # Convert buy_usd into quantity of coin
                        qty = (buy_usd / current_price) * (1 - self.transaction_cost)
                        # Weighted-average entry price if we already have a positive inventory
                        if self.inventory > 0:
                            # old_value = self.inventory * self.entry_price
                            # new_value = qty * current_price
                            # combined_inventory = self.inventory + qty
                            old_value = self.inventory * self.entry_price
                            new_value = qty * current_price
                            total_qty = self.inventory + qty
                            new_entry_price = (old_value + new_value) / total_qty
                            self.entry_price = new_entry_price
                            self.inventory = total_qty
                        else:
                            # First time buying or inventory == 0
                            self.entry_price = current_price
                            self.inventory += qty

                        self.cash -= buy_usd  # spent full 10% in USD
                        # logger.debug(f"[{current_timestamp}] Action Buy executed. Inventory: {self.inventory}, Entry Price: {self.entry_price}")

            elif action == 2:  # Sell (close all)
                if self.inventory <= 0:
                    # Trying to sell but no inventory => negative reward
                    reward -= 500.0
                else:
                    proceeds = self.inventory * current_price
                    proceeds_after_cost = proceeds * (1 - self.transaction_cost)
                    self.cash += proceeds_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0
                    # logger.debug(f"[{current_timestamp}] Action Sell executed. Cash: {self.cash}, Close Price: {current_price}")

            # action == 0 => hold => do nothing special

        # === Mode: SHORT ===
        elif self.mode == "short":
            if action == 1:  # "Sell" to open or add short
                # If we already have a long position or zero inventory, is that allowed?
                # We want multiple short sells only if inventory <= 0.
                if self.inventory > 0:
                    # Negative reward => can't open new short if you hold a positive inventory
                    reward -= 1000.0
                else:
                    # 10% of current cash (in USD)
                    sell_usd = 0.1 * self.cash
                    if sell_usd < 100:
                        reward -= 500.0
                    else:
                        qty = (sell_usd / current_price) * (1 - self.transaction_cost)
                        # Weighted-average short entry if we already hold a negative inventory
                        if self.inventory < 0:
                            old_value = abs(self.inventory) * self.entry_price
                            new_value = qty * current_price
                            total_qty = abs(self.inventory) + qty
                            new_entry_price = (old_value + new_value) / total_qty
                            self.entry_price = new_entry_price
                            self.inventory -= qty  # inventory is negative
                        else:
                            # first short
                            self.entry_price = current_price
                            self.inventory -= qty
                        self.cash -= sell_usd

            elif action == 2:  # "Buy" to cover entire short
                if self.inventory >= 0:
                    reward -= 500.0
                else:
                    cost_to_cover = abs(self.inventory) * current_price
                    cost_after_cost = cost_to_cover * (1 + self.transaction_cost)
                    # Attempt to close entire short
                    if cost_after_cost > self.cash:
                        # Not enough cash => partial coverage or penalty
                        reward -= 1000.0
                    else:
                        self.cash -= cost_after_cost
                        self.inventory = 0.0
                        self.entry_price = 0.0

            # action == 0 => hold => do nothing

        # ===== After Action Logic =====
        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps

        # Compute new portfolio value
        new_price = current_price
        if not done:
            try:
                new_price = self.data['close'].iloc[next_step]
            except IndexError:
                logger.error(f"Attempted to access step {next_step}, which is out of bounds.")
                done = True
                new_price = self.data['close'].iloc[self.current_step]

        portfolio_after = self.cash + self.inventory * new_price

        # The primary reward is the change in portfolio value
        reward += (portfolio_after - portfolio_before)

        # Next state or final zeros
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
            logger.warning(f"Portfolio value is below 50% of initial capital: {portfolio_after}")

        # Optional: if portfolio falls below 30% of initial, end episode early
        if portfolio_after < self.initial_capital * 0.3:
            # logger.debug(f"Portfolio < 30% initial: {portfolio_after}")
            done = True

        # logger.debug(f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after} - Done: {done}")
        return next_state, reward, done, {}

