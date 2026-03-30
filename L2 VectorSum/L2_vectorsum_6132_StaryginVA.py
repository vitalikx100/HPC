import os
import numpy as np
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt

# CUDA ядро для параллельного суммирования элементов вектора с редукцией внутри блока
cuda_kernel = """
__global__ void vector_sum(float *vec, float *result, int N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        shared[tid] = vec[idx];
    else
        shared[tid] = 0.0f;

    __syncthreads();

    for (int k = blockDim.x / 2; k > 0; k /= 2) {
        if (tid < k) {
            shared[tid] += shared[tid + k];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}
"""

# компиляция CUDA кода
source_module = SourceModule(cuda_kernel)
# получение скомпилированной функции
vector_sum_gpu = source_module.get_function("vector_sum")

def sum_cpu(vec):
    start = time.time()
    result = np.sum(vec)
    end = time.time()
    return result, (end - start) * 1000

def sum_gpu(vec):
    N = len(vec)

    # cuda.mem_alloc() - выделение памяти на GPU
    vec_gpu = cuda.mem_alloc(vec.nbytes)
    result_gpu = cuda.mem_alloc(np.dtype(np.float32).itemsize)

    # cuda.memcpy_htod() - копирование данных с CPU на GPU
    cuda.memcpy_htod(vec_gpu, vec)
    cuda.memcpy_htod(result_gpu, np.zeros(1, dtype=np.float32))

    block_size = 256
    grid_size = math.ceil(N / block_size)

    # cuda.Event() - создание события для измерения времени
    start = cuda.Event()
    end = cuda.Event()

    start.record()

    vector_sum_gpu(
        vec_gpu, result_gpu,
        np.int32(N),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        shared=block_size * np.dtype(np.float32).itemsize
    )

    end.record()
    # synchronize() - ожидание завершения операций на GPU
    end.synchronize()

    gpu_time = start.time_till(end)

    result = np.empty(1, dtype=np.float32)
    # cuda.memcpy_dtoh() - копирование данных с GPU на CPU
    cuda.memcpy_dtoh(result, result_gpu)

    return result[0], gpu_time


sizes = [1000, 10000, 50000, 100000, 200000, 500000, 1000000]
runs = 10

cpu_times = []
gpu_times = []
speedups = []

for N in sizes:
    cpu_sum_time = 0
    gpu_sum_time = 0

    for _ in range(runs):
        vec = np.random.rand(N).astype(np.float32)

        cpu_res, t_cpu = sum_cpu(vec)
        gpu_res, t_gpu = sum_gpu(vec)

        cpu_sum_time += t_cpu
        gpu_sum_time += t_gpu

        if not np.allclose(cpu_res, gpu_res, atol=1e-10):
            raise ValueError("Ошибка вычислений")

    cpu_average_time = cpu_sum_time / runs
    gpu_average_time = gpu_sum_time / runs
    speedup = cpu_average_time / gpu_average_time

    cpu_times.append(cpu_average_time)
    gpu_times.append(gpu_average_time)
    speedups.append(speedup)

    print(f"Размерность: {N}, Время CPU={cpu_average_time:.2f} мс, Время GPU={gpu_average_time:.2f} мс, Ускорение в {speedup:.2f} раз")

os.makedirs("images", exist_ok=True)

plt.figure()
plt.plot(sizes, cpu_times, marker='o')
plt.xlabel("Размер вектора, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на CPU")
plt.grid(True)
plt.savefig("images/cpu_time.png")

plt.figure()
plt.plot(sizes, gpu_times, marker='o')
plt.xlabel("Размер вектора, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на GPU")
plt.grid(True)
plt.savefig("images/gpu_time.png")

plt.figure()
plt.plot(sizes, speedups, marker='o')
plt.xlabel("Размер вектора, N")
plt.ylabel("Ускорение, раз")
plt.title("Ускорение")
plt.grid(True)
plt.savefig("images/speedup.png")
