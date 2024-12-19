# main.py

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer

def main():
    # Load data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Instantiate GeneticOptimizer correctly with data_loader only
    optimizer = GeneticOptimizer(data_loader)
    optimizer.test_individual()

if __name__ == '__main__':
    main()
