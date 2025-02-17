"""
train_rl.py
-----------
This script trains the reinforcement learning agent.
It loads market data, sets up the trading environment, creates the RL agent,
and runs multiple training episodes.
"""

from src.config_loader import Config
from src.data_loader import DataLoader
from src.rl_agent import DQNAgent
from src.rl_environment import TradingEnvironment

# Load configuration settings.
config = Config()  # Ensure config.yaml includes parameters like start_cutoff and end_cutoff.

# Load and preprocess market data.
data_loader = DataLoader()
data_loader.import_ticks()
data_loader.resample_data()

# For this example, use only the 'close' price as the indicator.
indicators = data_loader.tick_data[data_loader.base_timeframe][['close']]
price_data = data_loader.tick_data[data_loader.base_timeframe]

# Create a trading environment.
env = TradingEnvironment(price_data=price_data, indicators=indicators)
# Instantiate the RL agent.
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3)

N_EPISODES = 100
# Run training episodes.
for episode in range(N_EPISODES):
    state = env.reset()  # Reset the environment to start a new episode.
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state)  # Choose an action.
        next_state, reward, done, info = env.step(action)  # Execute action.
        agent.store_transition(state, action, reward, next_state, done)  # Store transition.
        agent.update_policy()  # Update the network from a batch of experiences.
        state = next_state  # Move to the next state.
        total_reward += reward  # Accumulate reward.
    print(f"Episode {episode}, total_profit: {total_reward}")

# Save the trained model weights.
agent.save('dqn_model.pth')
