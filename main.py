# main.py

import argparse

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Optimizer for Trading Strategy')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to load')
    args = parser.parse_args()

    # Load data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Instantiate GeneticOptimizer with checkpoint if provided
    optimizer = GeneticOptimizer(data_loader, checkpoint_file=args.checkpoint)

    # Run the genetic optimization
    optimizer.run()


if __name__ == '__main__':
    main()
