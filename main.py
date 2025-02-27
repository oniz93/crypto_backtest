"""
main.py
-------
This is the main entry point for the genetic algorithm optimization.
It loads market data, instantiates the GeneticOptimizer, and starts the optimization process.
"""

import argparse
import random
from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer
import multiprocessing as mp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask
import warnings

warnings.filterwarnings("ignore")
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        import cudf as pd
        USING_CUDF = True
        NUM_GPU = pd.cuda.get_device_count()
    else:
        USING_CUDF = False
        NUM_GPU = 0
except Exception:
    USING_CUDF = False

def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Optimizer for Trading Strategy')
    parser.add_argument('--session_id', type=str, default=None, help='Session ID for this GA run')
    parser.add_argument('--gen', type=int, default=None, help='Generation number to load population from')
    args = parser.parse_args()

    session = args.session_id if args.session_id is not None else random.randint(100000, 999999)

    # Initialize Dask CUDA cluster if using cuDF and multiple GPUs
    client = None
    if USING_CUDF and NUM_GPU > 1:
        cluster = LocalCUDACluster(
            n_workers=NUM_GPU,
            threads_per_worker=1,
            memory_limit="16GB",  # Adjust based on your GPU memory
            CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(NUM_GPU))
        )
        client = Client(cluster)
        dask.config.set({"dataframe.backend": "cudf"})
        print(f"Dask CUDA cluster initialized with {NUM_GPU} GPUs: {client}")

    # Load market data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Instantiate the GeneticOptimizer
    optimizer = GeneticOptimizer(data_loader, session_id=session, gen=args.gen)
    # Run the genetic optimization
    optimizer.run()

    # Cleanup
    if client:
        client.close()
        cluster.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()