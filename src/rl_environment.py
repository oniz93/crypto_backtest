# src/rl_environment.py
import logging
import numpy as np

logger = logging.getLogger('GeneticOptimizer')

class TradingEnvironment:
    def __init__(self, price_data, indicators_df, mode="long", initial_capital=100000, transaction_cost=0.005, max_steps=1000000):
        self.price_data = price_data          # second-level trades DataFrame
        self.indicators_df = indicators_df    # minute-level indicators DataFrame
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.mode = mode

        self.data = self.price_data
        self.data = self.data.sort_index()
        self.timestamps = self.data.index
        self.n_steps = len(self.data)

        # indicators_df: indexed by minute
        # For state_dim: data.shape[1] + 5 as before
        self.state_dim = self.data.shape[1] + 5
        self.action_dim = 3  # hold, buy, sell

        self.entry_price = 0.0
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
            logger.error(f"Step {step} out of bounds.")
            raise IndexError("Step out of bounds")

        row = self.data.iloc[step].values  # second-level price data row, just 'close'
        close_price = self.data['close'].iloc[step]
        adjusted_buy_price = close_price * (1 + self.transaction_cost)
        adjusted_sell_price = close_price / (1 + self.transaction_cost)

        # Find corresponding indicators by flooring timestamp to minute
        trade_ts = self.timestamps[step]
        minute_ts = trade_ts.floor('T')  # floor to nearest minute
        # Fetch indicators row
        if self.indicators_df is not None and not self.indicators_df.empty:
            try:
                indicators_row = self.indicators_df.loc[minute_ts].values
            except KeyError:
                # If no exact minute match, handle gracefully (e.g. use the previous minute)
                # or return zeros if missing
                # Here we assume it's always available, else:
                indicators_row = np.zeros(self.indicators_df.shape[1])
        else:
            indicators_row = np.zeros(0)  # no indicators

        # Concatenate: indicators + adjusted_buy_price + adjusted_sell_price + inventory + cash_ratio + entry_price
        # row currently only has 'close', we don't need it since we have close_price already.
        # We can skip row since close is known:
        # Actually, if 'row' only contains close, we don't need it separate. Just use indicators_row.
        state = np.concatenate([
            indicators_row,
            [adjusted_buy_price, adjusted_sell_price, self.inventory, self.cash / self.initial_capital, self.entry_price]
        ])

        return state

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_before = self.cash + self.inventory * current_price

        # Actions: same as before (buy/sell logic)
        # ... unchanged trading logic

        # After performing action:
        next_step = self.current_step + 1
        done = next_step >= self.n_steps or next_step >= self.max_steps

        if not done:
            new_price = self.data['close'].iloc[next_step]
            portfolio_after = self.cash + self.inventory * new_price
            reward = portfolio_after - portfolio_before
            try:
                next_state = self._get_state(next_step)
            except IndexError:
                logger.error(f"Failed to get state for step {next_step}, zeros used.")
                next_state = np.zeros(self.state_dim)
        else:
            new_price = self.data['close'].iloc[self.current_step]
            portfolio_after = self.cash + self.inventory * new_price
            reward = portfolio_after - portfolio_before
            next_state = np.zeros(self.state_dim)

        self.current_step = next_step
        if self.current_step % 1000 == 0:
            logger.debug(f"Step: {self.current_step} - Balance: {portfolio_after} - Done: {done}")

        if portfolio_after < self.initial_capital * 0.2:
            logger.warning(f"Portfolio value below 20% of initial capital: {portfolio_after}")
            done = True

        return next_state, reward, done, {}
