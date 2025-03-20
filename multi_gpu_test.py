import multiprocessing

import numba.cuda as cuda
import cudf
from cudf.datasets import randomdata

def process_on_gpu(device_id):
    cuda.select_device(device_id)
    df = randomdata(nrows=10000000,
                    dtypes={'a': int, 'b': str, 'c': str, 'd': int},
                    seed=12)
    df['e'] = df['a'] * 2
    print(f"Process on GPU {device_id} completed")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Set to "spawn" before any imports or process creation
    p0 = multiprocessing.Process(target=process_on_gpu, args=(0,))
    p0.start()
    p1 = multiprocessing.Process(target=process_on_gpu, args=(1,))
    p1.start()
    p0.join()
    p1.join()