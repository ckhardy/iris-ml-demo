from numba import cuda
import numpy as np
from timeit import default_timer as timer

# Normal function to run on CPU
def func(a):
    for i in range(a.size):
        a[i] += 1

# Function optimized to run on GPU
@cuda.jit
def func2(a):
    idx = cuda.grid(1)
    if idx < a.size:
        a[idx] += 1

if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)

    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    # GPU operations typically require device arrays and explicit copying
    a_gpu = cuda.to_device(a)
    threadsperblock = 1024
    blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock

    start = timer()
    func2[blockspergrid, threadsperblock](a_gpu)
    a_gpu.copy_to_host(a)
    print("with GPU:", timer() - start)
