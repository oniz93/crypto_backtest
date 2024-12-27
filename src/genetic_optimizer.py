# src/genetic_optimizer.py

import datetime
import gc
import json
import logging
import multiprocessing
import os
import random
import sys

import numpy as np
import pandas as pd
from deap import base, creator, tools
from dask.distributed import Client

from src.config_loader import Config
from src.rl_agent import DQNAgent
from src.rl_environment import TradingEnvironment

# Configure logging
logger = logging.getLogger('GeneticOptimizer')
logger.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Named Pipe Handler
pipe_path = '/tmp/genetic_optimizer_logpipe'
if not os.path.exists(pipe_path):
    os.mkfifo(pipe_path)

try:
    pipe = open(pipe_path, 'w')
    pipe_handler = logging.StreamHandler(pipe)
    pipe_handler.setLevel(logging.DEBUG)
    pipe_handler.setFormatter(formatter)
    logger.addHandler(pipe_handler)
except Exception as e:
    logger.error(f"Failed to open named pipe {pipe_path}: {e}")

# Global variables for multiprocessing
global_optimizer = None
global_indicators = None
global_model_params = None
global_parameter_indices = None


def init_worker(optimizer, indicators, model_params, parameter_indices):
    """
    Initializes each worker process with references.
    """
    global global_optimizer, global_indicators, global_model_params, global_parameter_indices
    global_optimizer = optimizer
    global_indicators = indicators
    global_model_params = model_params
    global_parameter_indices = parameter_indices


def eval_wrapper(individual):
    """
    Evaluates an individual using the global_optimizer instance (to be captured in closure).
    """
    # This placeholder is overwritten by a closure inside GeneticOptimizer.setup_deap()
    raise RuntimeError("eval_wrapper called before being replaced by dask_map_fn closure.")


def init_ind(icls):
    """
    Top-level function to initialize an individual.
    Uses global_indicators, global_model_params, and global_parameter_indices.
    """
    keys = list(global_parameter_indices.keys())
    ind = []
    for k in keys:
        if k[0] == 'model':
            low, high = global_model_params[k[1]]
        else:
            # indicator param
            indicator_name, param_name, timeframe = k
            low, high = global_indicators[indicator_name][param_name]
        if isinstance(low, int):
            val = random.randint(low, high)
        else:
            val = random.uniform(low, high)
        ind.append(val)
    return icls(ind)


class GeneticOptimizer:
    def __init__(self, data_loader, indicators_dir='precalculated_indicators_parquet', checkpoint_dir='checkpoints', checkpoint_file=None):
        self.data_loader = data_loader
        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.config = Config()

        # Define parameter ranges from your indicators and model_params
        self.indicators = self.define_indicators()
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9)
        }
        self.parameter_indices = self.create_parameter_indices()

        # Prepare data
        self.prepare_data()

        # Setup DEAP
        self.setup_deap()

        # Load checkpoint if needed
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def define_indicators(self):
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
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
        for indicator_name, params in self.indicators.items():
            for param_name in params.keys():
                for timeframe in self.timeframes:
                    parameter_indices[(indicator_name, param_name, timeframe)] = idx
                    idx += 1
        for param_name in self.model_params.keys():
            parameter_indices[('model', param_name)] = idx
            idx += 1
        return parameter_indices

    def get_total_parameters(self):
        return len(self.parameter_indices)

    def prepare_data(self):
        base_tf = self.data_loader.base_timeframe
        price_data = self.data_loader.tick_data[base_tf].copy()
        self.base_price_data = price_data

    def load_indicators(self, config):
        # Similar code as before
        indicator_params = config['indicator_params']
        indicators = {}
        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}
            for timeframe, params in timeframes_params.items():
                indicator_df = self.data_loader.calculate_indicator(indicator_name, params, timeframe)
                if timeframe != '1min':
                    shift_duration = pd.to_timedelta(timeframe)
                    indicator_df_shifted = indicator_df.copy()
                    indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                    indicator_df_1min = indicator_df_shifted.resample('1min').ffill()
                    indicator_df_1min.dropna(inplace=True)
                    indicator_df = indicator_df_1min
                indicators[indicator_name]['1min'] = indicator_df
        return indicators

    def prepare_features(self, indicators):
        features_df = self.base_price_data.copy()
        for indicator_name, tf_dict in indicators.items():
            if '1min' in tf_dict:
                df = tf_dict['1min']
                df = df.reindex(features_df.index, method='ffill').add_suffix(f'_{indicator_name}')
                features_df = features_df.join(df)
        features_df.dropna(inplace=True)

        return self.data_loader.filter_data_by_date(features_df, self.config.get('start_simulation'), self.config.get('end_simulation'))

    def extract_config_from_individual(self, individual):
        config = {}
        indicator_params = {}
        model_params = {}

        keys = list(self.parameter_indices.keys())
        for i, val in enumerate(individual):
            key = keys[i]
            if key[0] == 'model':
                model_params[key[1]] = val
            else:
                indicator_name, param_name, timeframe = key
                if indicator_name not in indicator_params:
                    indicator_params[indicator_name] = {}
                if timeframe not in indicator_params[indicator_name]:
                    indicator_params[indicator_name][timeframe] = {}
                indicator_params[indicator_name][timeframe][param_name] = val

        config['indicator_params'] = indicator_params
        config['model_params'] = model_params
        return config

    def evaluate_individual(self, individual):
        config = self.extract_config_from_individual(individual)
        indicators = self.load_indicators(config)
        features_df = self.prepare_features(indicators)

        if len(features_df) < 100:
            logger.warning("Not enough data to run RL.")
            return (0.0,), 0.0, None

        if 'close' not in features_df.columns:
            logger.error("'close' column missing in features_df.")
            return (0.0,), 0.0, None

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')

        env = self.create_environment(price_data, indicators_only)

        agent, avg_profit = self.run_rl_training(env, episodes=50)

        return (-avg_profit,), avg_profit, agent

    def create_environment(self, price_data, indicators):
        initial_capital = 100000
        transaction_cost = 0.005
        mode = self.config.get('training_mode')
        env = TradingEnvironment(price_data, indicators, initial_capital=initial_capital, transaction_cost=transaction_cost, mode=mode)
        return env

    def run_rl_training(self, env, episodes=10):
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
            logger.info(f"One training done. Reward: {ep_reward}")

        avg_reward = np.mean(total_rewards)
        return agent, avg_reward

    def setup_deap(self):
        # Create the DEAP types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create a client for Dask   # DASK-ADDED
        self.client = Client()      # You can customize arguments like n_workers, etc.

        # We'll define closures that capture `self`:
        def init_ind_closure(icls):
            """
            Replaces the old init_ind, but uses self.indicators, self.model_params, self.parameter_indices.
            """
            keys = list(self.parameter_indices.keys())
            ind = []
            for k in keys:
                if k[0] == 'model':
                    low, high = self.model_params[k[1]]
                else:
                    indicator_name, param_name, timeframe = k
                    low, high = self.indicators[indicator_name][param_name]
                if isinstance(low, int):
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
                ind.append(val)
            return icls(ind)

        def eval_wrapper_closure(individual):
            """
            Replaces old global eval_wrapper.
            It calls `self.evaluate_individual` under the hood.
            """
            return self.evaluate_individual(individual)

        # Now register them
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", init_ind_closure, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # DASK-based map function
        def dask_map_fn(func, items):
            futures = self.client.map(func, items)
            return self.client.gather(futures)

        # We use `eval_wrapper_closure` in the DEAP pipeline
        self.toolbox.register("map", dask_map_fn)  # DASK-CHANGED

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self):
        # Create initial population of 1000
        initial_pop_size = 1000
        pop = self.toolbox.population(n=initial_pop_size)
        NGEN = 100
        CXPB = 0.5
        MUTPB = 0.1
        current_datetime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        logger.info("Evaluating initial population...")
        # Actually, we can just do:
        results = self.toolbox.map(self.evaluate_individual, pop)

        for ind, (fit, avg_profit, agent) in zip(pop, results):
            ind.fitness.values = fit
            ind.avg_profit = avg_profit
            ind.agent = agent

        for gen in range(1, NGEN + 1):
            logger.info(f"=== Generation {gen} ===")

            # From the second generation onward, we want only 100 individuals
            desired_pop_size = 100 if gen > 1 else 1000

            offspring = self.toolbox.select(pop, desired_pop_size)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Evaluating {len(invalid_inds)} individuals...")
            results = self.toolbox.map(self.evaluate_individual, invalid_inds)  # DASK-based
            for ind, (fit, avg_profit, agent) in zip(invalid_inds, results):
                ind.fitness.values = fit
                ind.avg_profit = avg_profit
                ind.agent = agent

            pop[:] = offspring

            pop_file = f"population/{current_datetime}-gen{gen}.json"
            os.makedirs('population', exist_ok=True)
            population_data = []
            for i, ind in enumerate(pop):
                population_data.append({
                    'individual_index': i,
                    'parameters': list(ind),
                    'fitness': ind.fitness.values[0],
                    'avg_profit': getattr(ind, 'avg_profit', None)
                })
            with open(pop_file, 'w') as f:
                json.dump(population_data, f, indent=2)

            best_ind = tools.selBest(pop, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            best_reward = getattr(best_ind, 'avg_profit', None)
            best_agent = getattr(best_ind, 'agent', None)

            logger.info(f"Best Fitness: {best_fitness} | Best Reward: {best_reward}")

            if best_agent is not None:
                os.makedirs('weights', exist_ok=True)
                weight_file = f"weights/{current_datetime}-gen{gen}-best-{best_reward:.4f}.pth"
                best_agent.save(weight_file)
                logger.info(f"Saved RL agent's weights to {weight_file}")

            for ind in pop:
                if hasattr(ind, 'agent'):
                    del ind.agent
            gc.collect()

            logger.info(f"Generation {gen} completed. Best fitness: {best_fitness} | Best reward: {best_reward}")

        logger.info("DEAP Optimization Completed")

        # Gracefully close the Dask client
        self.client.close()

    def load_checkpoint(self, checkpoint_file):
        # Implement if needed
        pass

    def test_individual(self, individual_params=None):
        # Implement if needed
        pass
