import os
import numpy as np
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt

# CUDA ядро для умножения матриц, каждый поток вычисляет один элемент итоговой матрицы
cuda_kernel = """
__global__ void mat_mul(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

# компиляция CUDA кода
source_module = SourceModule(cuda_kernel)
# получение скомпилированной фукнции
mat_mul_gpu = source_module.get_function("mat_mul")

def multiply_cpu(A, B):
    start = time.time()
    C = np.dot(A, B)
    end = time.time()
    return C, (end - start) * 1000


def multiply_gpu(A, B):
    N = A.shape[0]

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # cuda.mem_alloc() - выделение памяти на GPU
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(N * N * np.dtype(np.float32).itemsize)

    # cuda.memcpy_htod() - копирование данных с CPU на GPU
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    block_size = 32
    grid_size = math.ceil(N / block_size)

    # cuda.Event() - создание события для измерения времени
    start = cuda.Event()
    end = cuda.Event()

    start.record()

    mat_mul_gpu(
        A_gpu, B_gpu, C_gpu,
        np.int32(N),
        block=(block_size, block_size, 1),
        grid=(grid_size, grid_size)
    )

    end.record()
    # synchronize() - ожидание завершения операций на GPU
    end.synchronize()

    gpu_time = start.time_till(end)

    C = np.empty((N, N), dtype=np.float32)
    # cuda.memcpy_dtoh() - копирование данных с GPU на CPU
    cuda.memcpy_dtoh(C, C_gpu)

    return C, gpu_time


sizes = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
runs = 10

cpu_times = []
gpu_times = []
speedups = []

for N in sizes:
    cpu_sum_time = 0
    gpu_sum_time = 0

    for _ in range(runs):
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)

        C_cpu, t_cpu = multiply_cpu(A, B)
        C_gpu, t_gpu = multiply_gpu(A, B)

        cpu_sum_time += t_cpu
        gpu_sum_time += t_gpu

        if not np.allclose(C_cpu, C_gpu, atol=1e-10):
            raise ValueError("Ошибка вычислений")

    cpu_average_time = cpu_sum_time / runs
    gpu_average_time = gpu_sum_time / runs
    speedup = cpu_average_time / gpu_average_time

    cpu_times.append(cpu_average_time)
    gpu_times.append(gpu_average_time)
    speedups.append(speedup)

    print(f"Размерность: {N}x{N}, Время CPU={cpu_average_time:.2f} мс, Время GPU={gpu_average_time:.2f} мс, Ускорение в {speedup:.2f} раз")

os.makedirs("images", exist_ok=True)

plt.figure()
plt.plot(sizes, cpu_times, marker='o')
plt.xlabel("Размерность матриц, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на CPU")
plt.grid(True)
plt.savefig("images/cpu_time.png")

plt.figure()
plt.plot(sizes, gpu_times, marker='o')
plt.xlabel("Размерность матриц, N")
plt.ylabel("Время, мс")
plt.title("Время выполнения на GPU")
plt.grid(True)
plt.savefig("images/gpu_time.png")

plt.figure()
plt.plot(sizes, speedups, marker='o')
plt.xlabel("Размерность матриц, N")
plt.ylabel("Ускорение, раз")
plt.title("Ускорение")
plt.grid(True)
plt.savefig("images/speedup.png")
