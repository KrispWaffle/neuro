import time
import cupy as cp 

N = 4096

if __name__ == "__main__":
    A = cp.random.randn(N, N, dtype=cp.float32)
    B = cp.random.randn(N, N, dtype=cp.float32)

    flops = N * N * 2 * N

    for i in range(10000000):
        cp.cuda.Device(0).synchronize()  
        start = time.perf_counter()
        C = A @ B 
        cp.cuda.Device(0).synchronize()  
        end = time.perf_counter()

        elapsed_time = end - start
        print(f"{flops / elapsed_time * 1e-12 :.2f} Tflops/s")
