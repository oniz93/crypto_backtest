# main.py

from src.data_loader import DataLoader
from src.trading_strategy import TradingStrategy
from src.simulation import Simulation
from src.genetic_optimizer import GeneticOptimizer


def main():
    # Load data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Run genetic optimization
    optimizer = GeneticOptimizer(TradingStrategy, data_loader)
    optimizer.test_individual()

if __name__ == '__main__':
    main()
