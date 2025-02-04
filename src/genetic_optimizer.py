# src/genetic_optimizer.py

import datetime
import gc
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
from deap import base, creator, tools

import ray
from multiprocessing import Pool

from src.config_loader import Config
from src.data_loader import DataLoader

logger = logging.getLogger('GeneticOptimizer')
logger.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional: You could remove or keep the Named Pipe Handler if you want logging to a pipe
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


@ray.remote
def ray_evaluate_individual(pickled_class, individual):
    """
    Evaluate one individual using a 'GeneticOptimizer' or the relevant evaluate method.
    We call pickled_class.evaluate_individual(individual).
    """
    return pickled_class.evaluate_individual(individual)


def local_evaluate_individual(args):
    optimizer, individual = args
    return optimizer.evaluate_individual(individual)


class GeneticOptimizer:
    def __init__(
        self,
        data_loader: DataLoader,
        session_id=None,
        gen=None,
        indicators_dir='precalculated_indicators_parquet',
        checkpoint_dir='checkpoints',
        checkpoint_file=None
    ):
        """
        :param data_loader: DataLoader instance
        :param session_id: string for the GA session
        :param gen: if not None, load population from that generation
                              else we minimize
        """
        self.data_loader = data_loader
        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.config = Config()

        # Store the unique session_id for this entire training run
        self.session_id = session_id if session_id else "unknown-session"
        self.gen = gen  # generation to load

        # define indicators
        self.indicators = self.define_indicators()
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9)
        }
        self.parameter_indices = self.create_parameter_indices()
        self.prepare_data()

        # Setup DEAP
        self.setup_deap()

        # Optionally load checkpoint
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def define_indicators(self):
        # return {
        #     'sma': {'length': (5, 200)},
        #     'ema': {'length': (5, 200)},
        #     'rsi': {'length': (5, 30)},
        #     'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
        #     'atr': {'length': (5, 50)},
        #     'stoch': {'k': (5, 20), 'd': (3, 10)},
        # }
        return {
            'vwap': {'offset': (0, 50)},
            'vwma': {'length': (5, 200)},
            'vpvr': {
                '1min':{'width': (1000, 50000)},
                '5min':{'width': (200, 10000)},
                '15min':{'width': (100, 3500)},
                '30min':{'width': (50, 2000)},
                '1h':{'width': (30, 1000)},
                '4h':{'width': (10, 250)},
                '1d':{'width': (10, 60)},
            },
        }

    def create_parameter_indices(self):
        parameter_indices = {}
        idx = 0
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
        for indicator_name, data in self.indicators.items():
            # data might look like: {'offset': (0,50)}  OR  {'1min': {'width':(...), ...}, '5min': {...}}
            if isinstance(data, dict):
                # Check if ALL keys are recognized timeframes
                # e.g. set(data.keys()) <= set(['1min','5min','15min',...])
                if set(data.keys()).issubset(set(self.timeframes)):
                    # => The user gave us per-timeframe params
                    for tf, param_dict in data.items():
                        # param_dict e.g. {'width': (1000,200000), ...}
                        for param_name, (low, high) in param_dict.items():
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
                else:
                    # => The user gave us a flat param dict for all timeframes
                    #    e.g. {'offset': (0,50)}
                    for param_name, (low, high) in data.items():
                        for tf in self.timeframes:
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
            else:
                raise ValueError(f"Indicators config for '{indicator_name}' must be a dict, got {type(data)}")

        # Finally, add our model params
        for param_name in self.model_params.keys():
            parameter_indices[('model', param_name)] = idx
            idx += 1

        return parameter_indices

    def prepare_data(self):
        base_tf = self.data_loader.base_timeframe
        self.base_price_data = self.data_loader.tick_data[base_tf].copy()

    def setup_deap(self):
        # By default DEAP tries to *minimize* the fitness
        # We'll keep that, but if we are maximizing, we invert sign in evaluate_individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_ind, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def init_ind(self, icls):
        """
        Creates a single individual with random parameters from the parameter range.
        """
        keys = list(self.parameter_indices.keys())
        ind = []
        for k in keys:
            if k[0] == 'model':
                # k = ('model', 'threshold_buy')
                low, high = self.model_params[k[1]]
            else:
                # k = (indicator_name, param_name, timeframe)
                indicator_name, param_name, timeframe = k
                indicator_dict = self.indicators[indicator_name]

                if set(indicator_dict.keys()).issubset(set(self.timeframes)):
                    # => "indicator_dict" is a per-timeframe dict
                    # e.g.  { '1min': {'width': (1000,200000)}, '5min': {...} }
                    low, high = indicator_dict[timeframe][param_name]
                else:
                    # => "indicator_dict" is a single dict of param ranges
                    # e.g.  {'offset': (0,50)}
                    low, high = indicator_dict[param_name]

            if isinstance(low, int):
                val = random.randint(low, high)
            else:
                val = random.uniform(low, high)
            ind.append(val)

        return icls(ind)

    def evaluate_individual(self, individual):
        """
        Evaluate an individual:
          - build config from the individual's parameters
          - run RL training
        """
        config = self.extract_config_from_individual(individual)
        indicators = self.load_indicators(config)
        features_df = self.prepare_features(indicators)

        if len(features_df) < 100:
            logger.warning("Not enough data to run RL.")
            # For a short population, fitness is poor
            return (9999999.0,), 0.0, None  # or large positive if we are minimizing

        if 'close' not in features_df.columns:
            logger.error("'close' column missing in features_df.")
            return (9999999.0,), 0.0, None

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')

        env = self.create_environment(price_data, indicators_only)
        agent, avg_profit = self.run_rl_training(env, episodes=200)

        # If we are maximizing profit, but GA is a minimizer => return negative
        return (-avg_profit,), avg_profit, agent

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

    def load_indicators(self, config):
        indicator_params = config['indicator_params']
        indicators = {}

        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}

            for timeframe, params in timeframes_params.items():
                indicator_df = self.data_loader.calculate_indicator(indicator_name, params, timeframe)

                # If not 1min, shift & resample so it lines up with base 1-minute data
                if timeframe != '1min':
                    shift_duration = pd.to_timedelta(timeframe)
                    indicator_df_shifted = indicator_df.copy()
                    indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                    indicator_df_1min = indicator_df_shifted.resample('1min').ffill()
                    indicator_df_1min.dropna(inplace=True)
                    indicator_df = indicator_df_1min

                # Store under the *actual* timeframe key (e.g. "5min", "15min", etc.)
                indicators[indicator_name][timeframe] = indicator_df

        return indicators

    def prepare_features(self, indicators):
        """
        Concatenate all timeframe data into a single DataFrame at 1-minute resolution.
        We keep the base self.base_price_data as the foundation (which is 1min bars).
        """
        features_df = self.base_price_data.copy()

        # For each indicator (e.g. 'vwap', 'vwma', 'vpvr')...
        for indicator_name, tf_dict in indicators.items():
            # And for each timeframe (e.g. '1min', '5min', '15min', etc.)
            for timeframe, df in tf_dict.items():
                # 1) Clean column names (remove any \n, trailing spaces)
                df.columns = [col.replace('\n', '').strip() for col in df.columns]

                # 2) Reindex to the base_price_data's index => ensures alignment at 1min
                df = df.reindex(features_df.index, method='ffill')

                # 3) Add a suffix that includes indicator_name + timeframe
                #    e.g. "VWAP" => "VWAP_vwap_5min"
                df = df.add_suffix(f'_{indicator_name}_{timeframe}')

                # 4) Join them into features_df
                features_df = features_df.join(df)

        # Drop rows with NaNs from any newly joined columns
        features_df.dropna(inplace=True)

        # Finally, respect the date range from the config
        filtered = self.data_loader.filter_data_by_date(
            features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )
        return filtered

    def create_environment(self, price_data, indicators):
        from src.rl_environment import TradingEnvironment

        initial_capital = 100000
        transaction_cost = 0.005
        mode = self.config.get('training_mode')
        return TradingEnvironment(
            price_data,
            indicators,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            mode=mode
        )

    def run_rl_training(self, env, episodes=10):
        from src.rl_agent import DQNAgent
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

    def evaluate_individuals(self, individuals):
        """
        Evaluate a list of individuals using the processing method specified in the config.
        """
        processing = self.config.get("processing", "ray").lower()
        if processing == "ray":
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            tasks = [ray_evaluate_individual.remote(self, ind) for ind in individuals]
            results = ray.get(tasks)
        elif processing == "local":
            with Pool() as pool:
                results = pool.map(local_evaluate_individual, [(self, ind) for ind in individuals])
        else:
            results = [self.evaluate_individual(ind) for ind in individuals]
        return results

    def run(self):
        NGEN = 100
        CXPB = 0.5
        MUTPB = 0.1

        INITIAL_POPULATION = 100

        os.makedirs(f"population/{self.session_id}", exist_ok=True)
        os.makedirs(f"weights/{self.session_id}", exist_ok=True)

        if self.gen is not None:
            # load population from file
            pop = self.load_population(self.session_id, self.gen)
            if not pop:
                # if it's empty or not found, fallback to generating
                pop = self.toolbox.population(n=INITIAL_POPULATION)
        else:
            # default create population
            pop = self.toolbox.population(n=INITIAL_POPULATION)

        logger.info("Evaluating initial population...")
        results = self.evaluate_individuals(pop)
        for ind, (fit, avg_profit, agent) in zip(pop, results):
            ind.fitness.values = fit
            ind.avg_profit = avg_profit
            ind.agent = agent

        for gen in range(1, NGEN + 1):
            logger.info(f"=== Generation {gen} ===")

            # after gen=1, we want 100 individuals, for example
            desired_pop_size = 1000 if gen > 1 else INITIAL_POPULATION

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
            logger.info(f"Evaluating {len(invalid_inds)} individuals with {self.config.get('processing', 'ray')} processing...")
            results = self.evaluate_individuals(invalid_inds)
            for ind, (fit, avg_profit, agent) in zip(invalid_inds, results):
                ind.fitness.values = fit
                ind.avg_profit = avg_profit
                ind.agent = agent

            pop[:] = offspring

            # Save population to JSON
            pop_file = f"population/{self.session_id}/gen{gen}.json"
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

            # Save RL agent if profit > 100000
            for ind_id, ind in enumerate(pop):
                agent_to_save = getattr(ind, 'agent', None)
                ind_profit = getattr(ind, 'avg_profit', 0)
                if agent_to_save is not None and ind_profit > 100000:
                    use_id = str(ind_id)
                    use_profit = int(ind_profit)
                    weight_file = f"weights/{self.session_id}/gen{gen}-{use_id}-{use_profit}.pth"
                    agent_to_save.save(weight_file)
                    logger.info(f"Saved RL agent's weights to {weight_file} because profit {ind_profit} > 100000")

            for ind in pop:
                if hasattr(ind, 'agent'):
                    del ind.agent
            gc.collect()

            logger.info(f"Generation {gen} completed. Best fitness: {best_fitness} | Best reward: {best_reward}")

        logger.info("DEAP Optimization Completed")

    def load_population(self, session_id: str, generation_number: int):
        """
        Load a population from a saved JSON file under population/{session_id}/gen{generation_number}.json
        and return a list of Individuals. The individuals will have their parameters set,
        but fitness is cleared (they'll need re-evaluation).
        """
        pop_file = f"population/{session_id}/gen{generation_number}.json"
        if not os.path.exists(pop_file):
            logger.warning(f"Population file {pop_file} not found.")
            return []

        with open(pop_file, 'r') as f:
            population_data = json.load(f)

        # We'll create an empty population of the right length
        pop = []
        for item in population_data:
            params = item['parameters']  # list of floats
            ind = creator.Individual(params)
            # We won't set fitness here => we'll re-evaluate
            pop.append(ind)

        logger.info(f"Loaded population of size {len(pop)} from {pop_file}")
        return pop

    def load_checkpoint(self, checkpoint_file):
        """
        Stub: not implementing global resume from file here.
        """
        pass

    def test_individual(self, individual_params=None):
        pass

    def debug_single_individual(self, individual_params=None):
        """
        Debug a single individual WITHOUT using Ray or the GA loop.
        Use this method to step through the code with breakpoints in your IDE.

        :param individual_params: list of parameter values, same length as self.parameter_indices
                                 or None => create random
        """

        """
        TODO: Genera e esporta individui di esempio e focalizzati su rl_environment per otimmizzare il più possibile il training
        """
        from deap import creator

        # 1) If not provided, create random parameters
        if individual_params is None:
            keys = list(self.parameter_indices.keys())
            individual_params = []
            for k in keys:
                if k[0] == 'model':
                    # k = ('model', 'threshold_buy')
                    low, high = self.model_params[k[1]]
                else:
                    # k = (indicator_name, param_name, timeframe)
                    indicator_name, param_name, timeframe = k
                    indicator_dict = self.indicators[indicator_name]

                    if set(indicator_dict.keys()).issubset(set(self.timeframes)):
                        # => "indicator_dict" is a per-timeframe dict
                        # e.g.  { '1min': {'width': (1000,200000)}, '5min': {...} }
                        low, high = indicator_dict[timeframe][param_name]
                    else:
                        # => "indicator_dict" is a single dict of param ranges
                        # e.g.  {'offset': (0,50)}
                        low, high = indicator_dict[param_name]

                if isinstance(low, int):
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
                individual_params.append(val)

        # 2) Create a DEAP Individual
        ind = creator.Individual(individual_params)

        # 3) Evaluate directly
        logger.info("=== Debugging single individual ===")
        fit, avg_profit, agent = self.evaluate_individual(ind)
        logger.info(f"Evaluation finished. Fit: {fit}, Profit: {avg_profit}")
        if agent is not None:
            logger.info("An RL agent was created; you can place breakpoints inside run_rl_training or environment.")
        else:
            logger.info("No RL agent was created (possibly not enough data).")

        return fit, avg_profit, agent
