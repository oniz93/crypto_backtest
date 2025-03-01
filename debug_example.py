# debug_example.py

import argparse
import random
from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer
import multiprocessing as mp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask
import warnings

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

def debug_single():
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

    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    params = [
        0, 2, 4, 8, 16, 8, 4, # VWAP
        200, 200, 200, 200, 200, 200, 200, # VWMA
        10, 20, 30, 40, 50, 100, 200, # VPVR
        8, 10, 12, 14, 16, 18, 20, # RSI
        8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, # MACD
        20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0,  # BBands length, std_dev (7 timeframes)
        14, 3, 14, 3, 14, 3, 14, 3, 14, 3, 14, 3, 14, 3,  # Stoch k, d (7 timeframes)
        14, 14, 14, 14, 14, 14, 14,  # ROC length (7 timeframes)
        20, 20, 20, 20, 20, 20, 20,  # Hist_Vol length (7 timeframes)
        14, 14, 14, 14, 14, 14, 15,  # ATR length (7 timeframes)
        0.6, 0.6  # Model thresholds (buy, sell)
    ]
    go = GeneticOptimizer(data_loader, session_id="debug123")
    go.debug_single_individual(params)

    if USING_CUDF and NUM_GPU > 1:
        client.close()
        cluster.close()

if __name__ == "__main__":
    debug_single()
