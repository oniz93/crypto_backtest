import dask
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
import cudf
import multiprocessing as mp
import time

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    # Step 1: Create a LocalCUDACluster using all available GPUs
    cluster = LocalCUDACluster(
        n_workers=6,  # One worker per GPU; adjust based on your 7 GPUs
        threads_per_worker=1,  # Typically 1 is fine for GPU tasks
        memory_limit="16GB",  # Adjust based on your GPU memory (e.g., 16GB per GPU)
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"  # Explicitly list 6 GPUs here
    )

    # Step 2: Connect a Dask Client to the cluster
    client = Client(cluster)
    print(client)  # Verify the cluster setup

    # Step 3: Set cuDF as the DataFrame backend
    dask.config.set({"dataframe.backend": "cudf"})

    t_start = time.time()
    # Step 4: Load data into a dask_cudf DataFrame
    df = dask_cudf.read_parquet("output_parquet/BTCUSDT-trades.parquet", npartitions=6)

    # Step 5: Perform operations (example: compute mean of a column)
    result = df['quantity'].mean().compute()  # Replace 'some_column' with an actual column like 'close'
    print(f"Mean of 'quantity': {result}")
    elapsed = time.time() - t_start
    print(f"computed in {elapsed:.4f} seconds.")

    # Optional: Shut down the cluster when done
    client.close()
    cluster.close()