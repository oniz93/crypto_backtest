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

        # Pre-convert the DataFrame for fast indexing:
        #  - self.data_values: a NumPy array of all row values
        #  - self.close_index: the index of the 'close' column in self.data
        #  - self.timestamps_list: a list of timestamps for fast lookup
        self.data_values = self.data.values
        self.columns = self.data.columns
        self.close_index = self.data.columns.get_loc('close')
        self.n_steps = len(self.data_values)
        self.timestamps_list = self.data.index.tolist()

        # Calculate state dimension: original features + 5 extra fields
        # (adjusted_buy_price, adjusted_sell_price, inventory, cash_ratio, entry_price)
        self.state_dim = self.data.shape[1] + 5
        self.action_dim = 3  # 0=hold, 1=buy, 2=sell

        self.entry_price = 0.0
        self.reset()

    def reset(self):
        self.entry_price = 0.0
        self.current_step = 0
        self.cash = self.initial_capital
        self.inventory = 0
        # Retrieve current close price directly from the NumPy array
        self.last_price = self.data_values[self.current_step][self.close_index]
        return self._get_state()

    def _get_state(self, step=None):
        if step is None:
            step = self.current_step
        if step < 0 or step >= self.n_steps:
            logger.error(f"Attempted to access step {step}, which is out of bounds.")
            raise IndexError("Step is out of bounds in _get_state.")

        # Retrieve the row quickly from the NumPy array
        row = self.data_values[step]
        close_price = row[self.close_index]

        # Compute adjusted prices using the transaction cost
        adjusted_buy_price = close_price * (1 + self.transaction_cost)
        adjusted_sell_price = close_price / (1 + self.transaction_cost)

        return np.concatenate([
            row,
            [adjusted_buy_price, adjusted_sell_price, self.inventory, self.cash / self.initial_capital, self.entry_price]
        ])

    def step(self, action):
        # Get the current price and portfolio value before any action.
        current_price = self.data_values[self.current_step][self.close_index]
        portfolio_before = self.cash + self.inventory * current_price

        # For logging purposes.
        current_timestamp = self.timestamps_list[self.current_step].strftime('%Y-%m-%d %H:%M:%S')

        # Initialize a penalty variable.
        penalty = 0.0
        # Define a fraction of cash to use in each trade.
        trade_fraction = 0.1

        # --- Process Actions Based on Mode ---
        if self.mode == "long":
            if action == 1:  # BUY
                # If already short, buying is an invalid move.
                if self.inventory < 0:
                    penalty = 0.02 * portfolio_before  # 2% penalty of current portfolio.
                else:
                    buy_usd = trade_fraction * self.cash
                    # If the available cash is too little to trade, assign a small penalty.
                    if buy_usd < 100:
                        penalty = 0.01 * portfolio_before
                    else:
                        qty = (buy_usd / current_price) * (1 - self.transaction_cost)
                        # If already holding a long position, update the weighted average entry price.
                        if self.inventory > 0:
                            old_value = self.inventory * self.entry_price
                            new_value = qty * current_price
                            total_qty = self.inventory + qty
                            self.entry_price = (old_value + new_value) / total_qty
                            self.inventory = total_qty
                        else:
                            self.entry_price = current_price
                            self.inventory += qty
                        self.cash -= buy_usd

            elif action == 2:  # SELL (Close Long Position)
                if self.inventory <= 0:
                    penalty = 0.02 * portfolio_before
                else:
                    proceeds = self.inventory * current_price
                    proceeds_after_cost = proceeds * (1 - self.transaction_cost)
                    self.cash += proceeds_after_cost
                    self.inventory = 0.0
                    self.entry_price = 0.0
            # Action 0 (hold) does nothing.

        elif self.mode == "short":
            if action == 1:  # Open/Add Short Position
                if self.inventory > 0:
                    penalty = 0.02 * portfolio_before
                else:
                    sell_usd = trade_fraction * self.cash
                    if sell_usd < 100:
                        penalty = 0.01 * portfolio_before
                    else:
                        qty = (sell_usd / current_price) * (1 - self.transaction_cost)
                        if self.inventory < 0:
                            old_value = abs(self.inventory) * self.entry_price
                            new_value = qty * current_price
                            total_qty = abs(self.inventory) + qty
                            self.entry_price = (old_value + new_value) / total_qty
                            self.inventory -= qty
                        else:
                            self.entry_price = current_price
                            self.inventory -= qty
                        self.cash -= sell_usd

            elif action == 2:  # Buy to Cover Short Position
                if self.inventory >= 0:
                    penalty = 0.02 * portfolio_before
                else:
                    cost_to_cover = abs(self.inventory) * current_price
                    cost_after_cost = cost_to_cover * (1 + self.transaction_cost)
                    if cost_after_cost > self.cash:
                        penalty = 0.02 * portfolio_before
                    else:
                        self.cash -= cost_after_cost
                        self.inventory = 0.0
                        self.entry_price = 0.0
            # Hold does nothing.

        # --- Step to the Next Time Period ---
        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps

        # Try to obtain the new price (if we're not at the end).
        if not done:
            try:
                new_price = self.data_values[next_step][self.close_index]
            except IndexError:
                logger.error(f"Attempted to access step {next_step}, which is out of bounds.")
                new_price = current_price
                done = True
        else:
            new_price = current_price

        # Compute the portfolio value after the action.
        portfolio_after = self.cash + self.inventory * new_price
        # The reward is the net change in portfolio value minus any penalties.
        reward = (portfolio_after - portfolio_before) - penalty
        if reward > 0:
            reward *= 3

        # Build the next state.
        if not done:
            try:
                next_state = self._get_state(next_step)
            except IndexError:
                logger.error(f"Failed to get next state for step {next_step}. Setting to zeros.")
                next_state = np.zeros(self.state_dim)
        else:
            next_state = np.zeros(self.state_dim)

        # Update the current step.
        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.debug(f"[{current_timestamp}] Step: {self.current_step} - Balance: {portfolio_after} - Done: {done}")

        # Terminal condition: if the portfolio falls below 50% of the initial capital.
        if portfolio_after < 0.5 * self.initial_capital:
            logger.warning(f"Portfolio value below 50% initial: {portfolio_after}")
            done = True
            reward -= 0.05 * portfolio_before  # additional terminal penalty

        return next_state, reward, done, {}

