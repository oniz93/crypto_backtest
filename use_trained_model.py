# use_trained_model.py

import numpy as np
import torch

from src.data_loader import DataLoader
from src.rl_agent import QNetwork

# Suppose we know state_dim and action_dim from training.
state_dim = 10  # This should match what was used during training
action_dim = 3  # hold, buy, sell

# Initialize the same network architecture
model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load('dqn_model.pth', map_location=torch.device('cpu')))
model.eval()


# No epsilon-greedy needed now. Just greedy action selection.
def select_greedy_action(model, state):
    with torch.no_grad():
        q_values = model(torch.FloatTensor(state).unsqueeze(0))
    return q_values.argmax().item()


# Now get current market data and compute indicators
# In a live setting, replace this with live data feed logic
data_loader = DataLoader()
data_loader.import_ticks()
data_loader.resample_data()

# For simplicity, assume state is last row of some indicators DataFrame
base_tf = data_loader.base_timeframe
price_data = data_loader.tick_data[base_tf].copy()

# Compute indicators used during training (must match training!)
# Example: just take 'close' price and some precomputed indicator columns
if 'close' not in price_data.columns:
    raise ValueError("Close price not found in price_data")

# Create a dummy state vector
# In real usage, replicate the same feature extraction steps from training
indicator_values = price_data.iloc[-1].values  # last row
# Suppose indicator_values length = state_dim - 2 (because we had position info during training)
# We'll assume inventory=0 and cash ratio=1 for this example
inventory = 0
cash_ratio = 1.0
state = np.concatenate([indicator_values, [inventory, cash_ratio]])

action = select_greedy_action(model, state)

if action == 0:
    print("Action: HOLD")
elif action == 1:
    print("Action: BUY")
elif action == 2:
    print("Action: SELL")
