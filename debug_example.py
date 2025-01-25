# debug_example.py

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer

def debug_single():
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    go = GeneticOptimizer(data_loader, session_id="debug123")
    go.debug_single_individual()

if __name__ == "__main__":
    debug_single()
