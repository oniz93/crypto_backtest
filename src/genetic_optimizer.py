# src/genetic_optimizer.py

import multiprocessing
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import os
import json
import pandas as pd
import logging
import sys

from src.config_loader import Config
from src.rl_environment import TradingEnvironment
from src.rl_agent import DQNAgent

# Configure logging
logger = logging.getLogger('GeneticOptimizer')
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Adjust as needed

# Named Pipe Handler
pipe_path = '/tmp/genetic_optimizer_logpipe'  # Path to your named pipe
if not os.path.exists(pipe_path):
    os.mkfifo(pipe_path)

try:
    pipe = open(pipe_path, 'w')
    pipe_handler = logging.StreamHandler(pipe)
    pipe_handler.setLevel(logging.DEBUG)  # Capture detailed logs
except Exception as e:
    logger.error(f"Failed to open named pipe {pipe_path}: {e}")
    pipe_handler = logging.NullHandler()

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
pipe_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(pipe_handler)

class GeneticOptimizer:
    def __init__(self, data_loader, indicators_dir='precalculated_indicators_parquet', checkpoint_dir='checkpoints', checkpoint_file=None):
        self.data_loader = data_loader
        self.indicator_cache = {}
        self.indicators = self.define_indicators()
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']

        # Define GA parameters and their bounds
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9)
        }
        self.parameter_indices = self.create_parameter_indices()

        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.prepare_data()
        self.config = Config()

        # Initialize variables to hold loaded checkpoint data
        self.loaded_best_config = None
        self.loaded_best_fitness = None
        self.loaded_best_agent_path = None

        # Load checkpoint if provided
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def define_indicators(self):
        # Define indicators and ranges for the GA
        return {
            'sma': {'length': (5, 200)},
            'ema': {'length': (5, 200)},
            'rsi': {'length': (5, 30)},
            'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
            'atr': {'length': (5, 50)},
            'stoch': {'k': (5, 20), 'd': (3, 10)},
        }

    def create_parameter_indices(self):
        parameter_indices = {}
        idx = 0
        for indicator_name, params in self.indicators.items():
            for param_name in params.keys():
                for timeframe in self.timeframes:
                    key = ('indicator', indicator_name, param_name, timeframe)
                    parameter_indices[key] = idx
                    idx += 1
        # If needed, we can also evolve RL hyperparameters, but let's skip for now
        for param_name in self.model_params.keys():
            key = ('model', param_name)
            parameter_indices[key] = idx
            idx += 1
        return parameter_indices

    def get_total_parameters(self):
        return len(self.parameter_indices)

    def get_varbound_and_vartype(self):
        total_params = self.get_total_parameters()
        varbound = np.zeros((total_params, 2))
        vartype = []
        for key, idx in self.parameter_indices.items():
            if key[0] == 'indicator':
                _, indicator_name, param_name, timeframe = key
                param_range = self.indicators[indicator_name][param_name]
            elif key[0] == 'model':
                _, param_name = key
                param_range = self.model_params[param_name]
            else:
                continue
            varbound[idx, 0] = param_range[0]
            varbound[idx, 1] = param_range[1]
            if isinstance(param_range[0], int):
                vartype.append('int')
            else:
                vartype.append('real')
        vartype = np.array(vartype).reshape(-1, 1)
        return varbound, vartype

    def extract_config_from_individual(self, individual):
        config = {}
        indicator_params = {}
        model_params = {}
        for key, idx in self.parameter_indices.items():
            value = individual[idx]
            if key[0] == 'indicator':
                _, indicator_name, param_name, timeframe = key
                if indicator_name not in indicator_params:
                    indicator_params[indicator_name] = {}
                if timeframe not in indicator_params[indicator_name]:
                    indicator_params[indicator_name][timeframe] = {}
                indicator_params[indicator_name][timeframe][param_name] = value
            elif key[0] == 'model':
                _, param_name = key
                model_params[param_name] = value
        config['indicator_params'] = indicator_params
        config['model_params'] = model_params
        return config

    def prepare_data(self):
        base_tf = self.data_loader.base_timeframe
        price_data = self.data_loader.tick_data[base_tf].copy()
        # We don't need buy/sell signals for RL
        # RL learns directly from reward
        self.base_price_data = price_data

    def load_indicators(self, config):
        indicators = {}
        indicator_params = config['indicator_params']
        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}
            for timeframe, params in timeframes_params.items():
                key = (indicator_name, tuple(params.items()), timeframe)
                if key in self.indicator_cache:
                    indicator_df = self.indicator_cache[key]
                else:
                    indicator_df = self.data_loader.calculate_indicator(indicator_name, params, timeframe)
                    # Reshape to 1min if timeframe != '1min'
                    if timeframe != '1min':
                        shift_duration = pd.to_timedelta(timeframe)
                        indicator_df_shifted = indicator_df.copy()
                        indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                        indicator_df_1min = indicator_df_shifted.resample('1min').ffill()
                        indicator_df_1min.dropna(inplace=True)
                        indicator_df = indicator_df_1min
                    self.indicator_cache[key] = indicator_df
                indicators[indicator_name]['1min'] = indicator_df
        return indicators

    def prepare_features(self, indicators):
        # Merge indicators into a single DataFrame
        features_df = self.base_price_data.copy()
        for indicator_name, tf_dict in indicators.items():
            if '1min' in tf_dict:
                df = tf_dict['1min']
                df = df.reindex(features_df.index, method='ffill').add_suffix(f'_{indicator_name}')
                features_df = features_df.join(df)
        features_df.dropna(inplace=True)
        # In RL, features_df will form the state. The environment will use these features.
        return features_df

    def run_rl_training(self, env, episodes=10):
        """
        Run RL training for a given environment and return the trained agent and average final profit.
        """
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3)
        total_rewards = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update_policy()
                state = next_state
                ep_reward += reward
            total_rewards.append(ep_reward)
            logger.info(f"RL Training - Episode {ep}/{episodes} - Episode Reward: {ep_reward}")
        avg_reward = np.mean(total_rewards)
        logger.info(f"RL Training - Average Reward over {episodes} episodes: {avg_reward}")
        return agent, avg_reward

    def evaluate(self, individual, n_evaluations=1, rl_episodes=10):
        """
        Evaluate an individual by:
        1. Extracting indicator params
        2. Loading indicators and preparing features
        3. Creating RL environment
        4. Running RL training and returning final avg profit as fitness
        """
        import multiprocessing
        process_id = multiprocessing.current_process().pid
        config = self.extract_config_from_individual(individual)
        indicators = self.load_indicators(config)
        features_df = self.prepare_features(indicators)

        if len(features_df) < 100:
            logger.warning(f"Process {process_id}: Not enough data to run RL.")
            return 0

        if 'close' not in features_df.columns:
            logger.error(f"Process {process_id}: 'close' column missing in features_df.")
            return 0

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')

        env = self.create_environment(price_data, indicators_only)

        # Log individual parameters
        logger.debug(f"Process {process_id}: Evaluating Individual Parameters: {config}")

        # Run RL training
        agent, avg_profit = self.run_rl_training(env, episodes=rl_episodes)

        # Log fitness
        logger.debug(f"Process {process_id}: Individual Fitness (Negative Avg Profit): {-avg_profit}")

        # Since GA's evaluate function expects a fitness value, return -avg_profit
        # Negative because GA minimizes, but we want to maximize profit
        return -avg_profit

    def create_environment(self, price_data, indicators):
        # Create TradingEnvironment instance
        initial_capital = 100000
        transaction_cost = 0.001
        mode = self.config.get('training_mode')  # Should be 'long' or 'short'
        env = TradingEnvironment(price_data, indicators, initial_capital=initial_capital, transaction_cost=transaction_cost, mode=mode)
        return env

    def run(self):
        varbound, vartype = self.get_varbound_and_vartype()
        algorithm_param = {
            'max_num_iteration': 100,  # Set to desired number of generations
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'two_point',
            'max_iteration_without_improv': 1000,
            'multiprocessing_ncpus': 5,  # Use 3 CPUs
            'multiprocessing_engine': 'process',  # Use multiprocessing
            # 'callback_generation': self.generation_callback  # Uncomment if supported
        }

        logger.info("Starting Genetic Algorithm Optimization")

        model = ga(
            function=self.evaluate,
            dimension=self.get_total_parameters(),
            variable_type_mixed=vartype,
            variable_boundaries=varbound,
            function_timeout=1000000,
            algorithm_parameters=algorithm_param
        )

        # Run the GA
        model.run()

        logger.info("Genetic Algorithm Optimization Completed")

        # After GA run, get the best individual
        best_individual = model.output_dict['variable']
        best_fitness = model.output_dict['function']

        logger.info(f"Best Individual Fitness: {-best_fitness}")
        logger.debug(f"Best Individual Parameters: {self.extract_config_from_individual(best_individual)}")

        # Extract config
        best_config = self.extract_config_from_individual(best_individual)

        # Load indicators and prepare features
        indicators = self.load_indicators(best_config)
        features_df = self.prepare_features(indicators)

        if len(features_df) < 100:
            logger.warning("Not enough data to run RL for the best individual.")
            return

        if 'close' not in features_df.columns:
            logger.error("'close' column missing in features_df.")
            return

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')

        # Create environment
        env = self.create_environment(price_data, indicators_only)

        # Run RL training for the best individual
        agent, avg_profit = self.run_rl_training(env, episodes=10)

        logger.info(f"Average Profit from RL Training: {avg_profit}")

        # Save the RL agent's weights
        current_iteration = model.current_iteration if hasattr(model, 'current_iteration') else 'final'
        best_agent_path = os.path.join(self.checkpoint_dir, f'best_agent_gen_{current_iteration}.pth')
        agent.save(best_agent_path)
        logger.info(f"Saved RL agent's weights to {best_agent_path}")

        # Save checkpoint
        self.save_checkpoint(
            generation=current_iteration,
            best_config=best_config,
            best_fitness=best_fitness,
            best_agent_path=best_agent_path
        )
        logger.info(f"Saved checkpoint to {os.path.join(self.checkpoint_dir, 'ga_checkpoint.json')}")

    def test_individual(self, individual_params=None):
        if individual_params is None:
            total_params = self.get_total_parameters()
            varbound, vartypes = self.get_varbound_and_vartype()
            individual = []
            for idx in range(total_params):
                low, high = varbound[idx]
                vt = vartypes[idx][0]
                if vt == 'int':
                    val = int(np.random.randint(low, high + 1))
                else:
                    val = float(np.random.uniform(low, high))
                individual.append(val)
        else:
            individual = individual_params
        fitness = self.evaluate(individual)
        logger.info(f"Test Individual Fitness: {fitness}")
        return fitness

    def save_checkpoint(self, generation, best_config, best_fitness, best_agent_path):
        checkpoint_data = {
            'generation': generation,
            'best_individual_params': best_config,
            'best_individual_fitness': best_fitness,
            'best_agent_path': best_agent_path
        }
        checkpoint_file = os.path.join(self.checkpoint_dir, 'ga_checkpoint.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved to {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        generation = checkpoint_data.get('generation', 0)
        best_individual_params = checkpoint_data.get('best_individual_params')
        best_fitness = checkpoint_data.get('best_individual_fitness')
        best_agent_path = checkpoint_data.get('best_agent_path')

        logger.info(f"Loaded checkpoint from generation {generation}")
        logger.info(f"Best individual fitness: {-best_fitness}")
        logger.info(f"Best agent weights path: {best_agent_path}")

        # Optionally, load the RL agent's weights
        if best_agent_path and os.path.exists(best_agent_path):
            # Initialize a new DQNAgent with the correct dimensions
            state_dim = self.data_loader.tick_data['1min'].shape[1] + 3  # Adjust as per your state_dim
            action_dim = 3  # As defined in TradingEnvironment
            agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
            agent.load(best_agent_path)
            self.loaded_best_agent = agent
            logger.info("Loaded best RL agent's weights.")
        else:
            logger.error(f"Best agent weights file {best_agent_path} not found.")

        # Note: Restoring the GA's population and state is not supported by the 'geneticalgorithm' package.
        # If you need to resume the GA, you may need to implement a custom GA loop or use a different library.
