# src/genetic_optimizer.py

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import multiprocessing

class GeneticOptimizer:
    def __init__(self, strategy_class, data_loader):
        self.strategy_class = strategy_class
        self.data_loader = data_loader
        self.indicator_cache = {}
        self.indicators = self.define_indicators()
        self.timeframes = [
            '1T', '5T', '15T', '30T', '45T', '1H', '2H', '4H', '8H', '12H', '1D', '1W'
        ]
        self.parameter_indices = self.create_parameter_indices()

    def define_indicators(self):
        """
        Define the indicators and their parameter ranges.
        """
        return {
            'sma': {'length': (5, 200)},
            'ema': {'length': (5, 200)},
            'rsi': {'length': (5, 30)},
            'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
            'bbands': {'length': (5, 50), 'std_dev': (1.0, 3.0)},
            'atr': {'length': (5, 50)},
            'stoch': {'k': (5, 20), 'd': (3, 10)},
            'cci': {'length': (5, 50)},
            'adx': {'length': (5, 50)},
            'cmf': {'length': (5, 50)},
            'mfi': {'length': (5, 30)},
            'roc': {'length': (5, 50)},
            'willr': {'length': (5, 30)},
            'psar': {'acceleration': (0.01, 0.1), 'max_acceleration': (0.1, 0.5)},
            'ichimoku': {'tenkan': (5, 20), 'kijun': (20, 60), 'senkou': (40, 100)},
            'keltner': {'length': (5, 50), 'multiplier': (1.0, 3.0)},
            'donchian': {'lower_length': (5, 50), 'upper_length': (5, 50)},
            'emv': {'length': (5, 20)},
            'force': {'length': (1, 20)},
            'uo': {'short': (5, 7), 'medium': (8, 14), 'long': (15, 28)},
            'volatility': {'length': (5, 50)},
            'dpo': {'length': (5, 50)},
            'trix': {'length': (5, 50)},
            'chaikin_osc': {'fast': (3, 10), 'slow': (10, 20)}
            # Indicators without parameters are not included
        }

    def create_parameter_indices(self):
        """
        Create a mapping of parameter indices for the individual's genes.
        """
        parameter_indices = {}
        idx = 0
        for indicator_name, params in self.indicators.items():
            for param_name in params.keys():
                for timeframe in self.timeframes:
                    key = (indicator_name, param_name, timeframe)
                    parameter_indices[key] = idx
                    idx += 1
        return parameter_indices

    def get_total_parameters(self):
        """
        Get the total number of parameters.
        """
        return len(self.parameter_indices)

    def get_varbound_and_vartype(self):
        """
        Create varbound and vartype arrays for the genetic algorithm.
        """
        total_params = self.get_total_parameters()
        varbound = np.zeros((total_params, 2))
        vartype = []
        for key, idx in self.parameter_indices.items():
            indicator_name, param_name, timeframe = key
            param_range = self.indicators[indicator_name][param_name]
            varbound[idx, 0] = param_range[0]
            varbound[idx, 1] = param_range[1]
            if isinstance(param_range[0], int):
                vartype.append('int')
            else:
                vartype.append('real')
        vartype = np.array(vartype).reshape(-1, 1)
        return varbound, vartype

    def evaluate(self, individual):
        """
        Evaluate the fitness of an individual.
        """
        config = self.extract_config_from_individual(individual)
        indicators = self.load_indicators(config)
        strategy = self.strategy_class(self.data_loader, config, indicators)
        result = -strategy.calculate_profit()
        # Save results to a file if needed
        return result

    def extract_config_from_individual(self, individual):
        """
        Extract the configuration from the individual's genes.
        """
        config = {}
        indicator_params = {}
        for key, idx in self.parameter_indices.items():
            indicator_name, param_name, timeframe = key
            value = individual[idx]
            if indicator_name not in indicator_params:
                indicator_params[indicator_name] = {}
            if timeframe not in indicator_params[indicator_name]:
                indicator_params[indicator_name][timeframe] = {}
            indicator_params[indicator_name][timeframe][param_name] = value
        config['indicator_params'] = indicator_params
        # Add other global parameters if needed
        return config

    def load_indicators(self, config):
        """
        Load indicators based on the individual's parameters.
        """
        indicators = {}
        indicator_params = config['indicator_params']
        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}
            for timeframe, params in timeframes_params.items():
                key = (indicator_name, tuple(params.items()), timeframe)
                if key in self.indicator_cache:
                    indicator_df = self.indicator_cache[key]
                else:
                    indicator_df = self.data_loader.calculate_indicator(
                        indicator_name, params, timeframe)
                    self.indicator_cache[key] = indicator_df
                indicators[indicator_name][timeframe] = indicator_df
        return indicators

    def run(self):
        varbound, vartype = self.get_varbound_and_vartype()
        algorithm_param = {
            'max_num_iteration': None,
            'population_size': 100,
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
