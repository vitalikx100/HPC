import os
import numpy as np
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import curandom
import time
import matplotlib.pyplot as plt

cuda_kernel = """
__global__ void pi_monte_carlo(float *points, int *count, int N)
{
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = 0;
    if (idx < N) {
        float x = points[2 * idx];
        float y = points[2 * idx + 1];

        if (x * x + y * y <= 1.0f) {
            shared[tid] = 1;
        }
    }

    __syncthreads();

    for (int k = blockDim.x / 2; k > 0; k /= 2) {
        if (tid < k) {
            shared[tid] += shared[tid + k];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, shared[0]);
    }
}
"""

source_module = SourceModule(cuda_kernel)
pi_monte_carlo_gpu = source_module.get_function("pi_monte_carlo")

def compute_pi_cpu(N):
    start = time.time()
    x = np.random.rand(N).astype(np.float32)
    y = np.random.rand(N).astype(np.float32)
    count_cpu = np.sum(x * x + y * y <= 1)
    pi = 4 * count_cpu / N
    end = time.time()
    return pi, (end - start) * 1000


def compute_pi_gpu(N):
    start_total = time.time()

    random_generator = curandom.XORWOWRandomNumberGenerator()
    points_gpu = random_generator.gen_uniform((N, 2), dtype=np.float32)

    count_gpu = cuda.mem_alloc(4)
    cuda.memcpy_htod(count_gpu, np.array([0], dtype=np.int32))

    block_size = 256
    grid_size = math.ceil(N / block_size)

    pi_monte_carlo_gpu(
        points_gpu, count_gpu, np.int32(N),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        shared=block_size * 4
    )

    cuda.Context.synchronize()

    count = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(count, count_gpu)

    pi = 4 * count[0] / N

    end_total = time.time()

    return pi, (end_total - start_total) * 1000


sizes = [10_000_000, 50_000_000, 100_000_000, 200_000_000, 350_000_000, 500_000_000]
runs = 10

cpu_times = []
gpu_times = []
speedups = []

for N in sizes:
    cpu_time_sum = 0
    pi_cpu_total = 0

    gpu_time_sum = 0
    pi_gpu_total = 0

    for _ in range(runs):
        pi_cpu, t_cpu = compute_pi_cpu(N)
        pi_gpu, t_gpu = compute_pi_gpu(N)

        cpu_time_sum += t_cpu
        pi_cpu_total = pi_cpu

        gpu_time_sum += t_gpu
        pi_gpu_total = pi_gpu

    cpu_average_time = cpu_time_sum / runs
    gpu_average_time = gpu_time_sum / runs
    speedup = cpu_average_time / gpu_average_time

    cpu_times.append(cpu_average_time)
    gpu_times.append(gpu_average_time)
    speedups.append(speedup)

    print(f"N: {N}, CPU: {cpu_average_time:.2f} мс, GPU: {gpu_average_time:.2f} мс, Ускорение в: {speedup:.3f} раз, π CPU = {pi_cpu_total:.8f}, π GPU = {pi_gpu_total:.8f}")


os.makedirs("images", exist_ok=True)

plt.figure()
plt.plot(sizes, cpu_times, marker='o')
plt.xlabel("Количество точек, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на CPU")
plt.grid(True)
plt.savefig("images/pi_cpu_time.png")

plt.figure()
plt.plot(sizes, gpu_times, marker='o')
plt.xlabel("Количество точек, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на GPU")
plt.grid(True)
plt.savefig("images/pi_gpu_time.png")

plt.figure()
plt.plot(sizes, speedups, marker='o')
plt.xlabel("Количество точек, N")
plt.ylabel("Ускорение, раз")
plt.title("Ускорение GPU относительно CPU")
plt.grid(True)
plt.savefig("images/pi_speedup.png")