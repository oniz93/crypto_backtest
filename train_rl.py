# train_rl.py

from src.config_loader import Config
from src.data_loader import DataLoader
from src.rl_environment import TradingEnvironment
from src.rl_agent import DQNAgent
import numpy as np

# Load config
config = Config()  # ensure config.yaml has start_cutoff,end_cutoff, etc.

data_loader = DataLoader()
data_loader.import_ticks()
data_loader.resample_data()

# Prepare indicators if needed
# In RL, you might still calculate indicators upfront using data_loader
# or you can do it inside environment creation.

# Example: no indicators for simplicity
indicators = data_loader.tick_data[data_loader.base_timeframe][['close']] # minimal indicators
# or compute your indicators and join them here.

price_data = data_loader.tick_data[data_loader.base_timeframe]

env = TradingEnvironment(price_data=price_data, indicators=indicators)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space, lr=1e-3)

N_EPISODES = 100
for episode in range(N_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update_policy()
        state = next_state
        total_reward += reward
    print(f"Episode {episode}, total_profit: {total_reward}")

# Save the trained model
agent.save('dqn_model.pth')
