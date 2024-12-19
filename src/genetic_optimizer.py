# src/genetic_optimizer.py
import multiprocessing
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import os
import json
import pandas as pd

from src.config_loader import Config

from src.rl_environment import TradingEnvironment
from src.rl_agent import DQNAgent

class GeneticOptimizer:
    def __init__(self, data_loader, indicators_dir='precalculated_indicators_parquet', checkpoint_dir='checkpoints'):
        self.data_loader = data_loader
        self.indicator_cache = {}
        self.indicators = self.define_indicators()
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
        # model_params could remain or be simplified since RL doesn't need them as before
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
        Run RL training for a given environment and return average final profit.
        """
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3)
        total_rewards = []
        for ep in range(episodes):
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
        avg_reward = np.mean(total_rewards)
        return avg_reward

    def evaluate(self, individual, n_evaluations=1, rl_episodes=10):
        config = self.extract_config_from_individual(individual)
        indicators = self.load_indicators(config)
        features_df = self.prepare_features(indicators)

        if len(features_df) < 100:
            print("Not enough data to run RL.")
            return 0

        if 'close' not in features_df.columns:
            print("'close' column missing in features_df.")
            return 0

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')

        env = self.create_environment(price_data, indicators_only)

        print(f"Environment state_dim: {env.state_dim}, action_dim: {env.action_dim}")

        # Run RL training
        avg_profit = self.run_rl_training(env, episodes=rl_episodes)
        print(f"Average Profit: {avg_profit}")

        return -avg_profit  # GA minimizes, we return negative

    def create_environment(self, price_data, indicators):
        initial_capital = 100000
        transaction_cost = 0.001
        mode = self.config.get('training_mode')  # Should be 'long' or 'short'
        env = TradingEnvironment(price_data, indicators, initial_capital=initial_capital, transaction_cost=transaction_cost, mode=mode)
        return env

    def run(self):
        varbound, vartype = self.get_varbound_and_vartype()
        algorithm_param = {
            'max_num_iteration': None,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'two_point',
            'max_iteration_without_improv': None,
            'multiprocessing_ncpus': multiprocessing.cpu_count(),
            'multiprocessing_engine': None
        }

        model = ga(
            function=self.evaluate,
            dimension=self.get_total_parameters(),
            variable_type_mixed=vartype,
            variable_boundaries=varbound,
            function_timeout=1000000,
            algorithm_parameters=algorithm_param
        )
        model.run()

    def test_individual(self, individual_params=None):
        if individual_params is None:
            total_params = self.get_total_parameters()
            varbound, vartypes = self.get_varbound_and_vartype()
            individual = np.random.uniform(varbound[:, 0], varbound[:, 1]).tolist()
            # Alternatively, handle integer and real types accordingly
        else:
            individual = individual_params
        fitness = self.evaluate(individual)
        print(f"Fitness of the test individual: {fitness}")
        return fitness

    def save_checkpoint(self, generation, population, model_path='model_checkpoint.joblib', config_path='config_checkpoint.json'):
        checkpoint_data = {
            'generation': generation,
            'population': population
        }
        with open(os.path.join(self.checkpoint_dir, 'ga_checkpoint.json'), 'w') as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, 'ga_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            generation = checkpoint_data['generation']
            population = checkpoint_data['population']
            return generation, population
        return None, None
