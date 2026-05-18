import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import argparse


cuda_kernel = """
__global__ void mass_search(
    const unsigned char *buf, // входной буфер
    int buf_len,
    int *R, // [n_patterns * buf_len]
    const int *pair_starts, // pair_starts[c] - начало блока пар для символа c
    const int *pair_counts, // pair_counts[c] - количество пар для символа c
    const int *pair_pat, // идентификатор подстроки для каждой пары
    const int *pair_off // смещение символа внутри подстроки для каждой пары
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buf_len) return;

    unsigned char c = buf[i];

    int start = pair_starts[c];
    int count = pair_counts[c];

    for (int t = 0; t < count; t++) {
        int pat = pair_pat[start + t];
        int off = pair_off[start + t];
        int col = i - off;

        if (col >= 0 && col < buf_len) {
            atomicSub(&R[pat * buf_len + col], 1);
        }
    }
}
"""

source_module = SourceModule(cuda_kernel)
mass_search_gpu_kernel = source_module.get_function("mass_search")

def generate_data(buf_len, n_patterns, min_pat_len, max_pat_len, seed=123):
    rng = np.random.default_rng(seed)
    buf = bytearray(rng.integers(0, 256, buf_len, dtype=np.uint8).tobytes())

    patterns = []
    for _ in range(n_patterns):
        length = int(rng.integers(min_pat_len, max_pat_len + 1))
        pat = bytes(rng.integers(0, 256, length, dtype=np.uint8))
        patterns.append(pat)

    # Вставка трех подстрок в известные позиции буфера для гарантированного результата
    guaranteed_positions = []
    insert_at = [10, buf_len // 3, buf_len // 2]
    for idx, pos in enumerate(insert_at):
        if idx < len(patterns):
            pat = patterns[idx]
            end = min(pos + len(pat), buf_len)
            buf[pos:end] = pat[:end - pos]
            guaranteed_positions.append((idx, pos))

    return bytes(buf), patterns, guaranteed_positions

def build_pair_tables(patterns):
    """
    Для каждого байта алфавита строит список пар
    """
    symbol_pairs = [[] for _ in range(256)]
    for pid, pat in enumerate(patterns):
        for k, byte in enumerate(pat):
            symbol_pairs[byte].append((pid, k))

    pair_starts = np.zeros(256, dtype=np.int32)
    pair_counts = np.zeros(256, dtype=np.int32)

    all_pat = []
    all_off = []

    offset = 0
    for sym in range(256):
        pairs = symbol_pairs[sym]
        pair_starts[sym] = offset
        pair_counts[sym] = len(pairs)
        for pid, k in pairs:
            all_pat.append(pid)
            all_off.append(k)
        offset += len(pairs)

    pair_pat = np.array(all_pat, dtype=np.int32) if all_pat else np.zeros(1, dtype=np.int32)
    pair_off = np.array(all_off, dtype=np.int32) if all_off else np.zeros(1, dtype=np.int32)

    return pair_starts, pair_counts, pair_pat, pair_off


def build_initial_matrix(patterns, buf_len):
    """
    Матрица R размером n_patterns * buf_len
    Строка i заполняется длиной подстроки i
    """
    n = len(patterns)
    R = np.zeros((n, buf_len), dtype=np.int32)
    for i, pat in enumerate(patterns):
        R[i, :] = len(pat)
    return R


def mass_search_cpu(buf, patterns):
    buf_len = len(buf)
    n = len(patterns)
    R = build_initial_matrix(patterns, buf_len)

    pair_starts, pair_counts, pair_pat, pair_off = build_pair_tables(patterns)

    for i in range(buf_len):
        c = buf[i]
        start = pair_starts[c]
        count = pair_counts[c]
        for t in range(count):
            pid = pair_pat[start + t]
            off = pair_off[start + t]
            col = i - off
            if 0 <= col < buf_len:
                R[pid, col] -= 1

    return collect_results(R, n)


def collect_results(R, n_patterns):
    results = {}
    for p in range(n_patterns):
        positions = [int(x) for x in np.where(R[p] == 0)[0]]
        results[p] = positions
    return results


def mass_search_gpu(buf, patterns):
    buf_len = len(buf)
    n = len(patterns)

    buf_np = np.frombuffer(buf, dtype=np.uint8)
    R_np = build_initial_matrix(patterns, buf_len)
    pair_starts, pair_counts, pair_pat, pair_off = build_pair_tables(patterns)

    buf_gpu = cuda.mem_alloc(buf_np.nbytes)
    R_gpu = cuda.mem_alloc(R_np.nbytes)
    starts_gpu = cuda.mem_alloc(pair_starts.nbytes)
    counts_gpu = cuda.mem_alloc(pair_counts.nbytes)
    pair_pat_gpu = cuda.mem_alloc(pair_pat.nbytes)
    pair_off_gpu = cuda.mem_alloc(pair_off.nbytes)

    cuda.memcpy_htod(buf_gpu, buf_np)
    cuda.memcpy_htod(R_gpu, R_np)
    cuda.memcpy_htod(starts_gpu, pair_starts)
    cuda.memcpy_htod(counts_gpu, pair_counts)
    cuda.memcpy_htod(pair_pat_gpu, pair_pat)
    cuda.memcpy_htod(pair_off_gpu, pair_off)

    block_size = 256
    grid_size = (buf_len + block_size - 1) // block_size

    mass_search_gpu_kernel(
        buf_gpu, np.int32(buf_len),
        R_gpu,
        starts_gpu, counts_gpu, pair_pat_gpu, pair_off_gpu,
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )

    cuda.Context.synchronize()

    R_result = np.empty_like(R_np)
    cuda.memcpy_dtoh(R_result, R_gpu)

    return collect_results(R_result, n)


def compare_results(cpu_res, gpu_res, n_patterns):
    all_match = True
    for p in range(n_patterns):
        cpu_pos = sorted(cpu_res.get(p, []))
        gpu_pos = sorted(gpu_res.get(p, []))
        if cpu_pos != gpu_pos:
            all_match = False
            print(f"Несовпадение для подстроки {p}: CPU={cpu_pos}, GPU={gpu_pos}")
    return all_match


parser = argparse.ArgumentParser(
    description="Mass Search"
)
parser.add_argument("--buf_len", type=int, default=100_000,
                    help="Длина буфера поиска (по умолчанию 100 000)")
parser.add_argument("--n_patterns", type=int, default=50,
                    help="Количество подстрок (по умолчанию 50)")
parser.add_argument("--min_len", type=int, default=3,
                    help="Минимальная длина подстроки (по умолчанию 3)")
parser.add_argument("--max_len", type=int, default=10,
                    help="Максимальная длина подстроки (по умолчанию 10)")
parser.add_argument("--seed", type=int, default=123,
                    help="Повторяемость (по умолчанию 123)")

args = parser.parse_args()

print(f"Длина буфера: {args.buf_len}")
print(f"Количество подстрок: {args.n_patterns}")
print(f"Длина подстрок от {args.min_len} до {args.max_len}")
print(f"Seed: {args.seed}")

print("\nГЕНЕРАЦИЯ ВХОДНЫХ ДАННЫХ:")

buf, patterns, guaranteed = generate_data(
    args.buf_len, args.n_patterns, args.min_len, args.max_len, args.seed
)

print(f"Буфер: {len(buf)}")
print(f"Подстрок: {len(patterns)}")
print(f"Гарантированно вставлено совпадений: {len(guaranteed)}")

for pid, pos in guaranteed:
    print(f"Подстрока {pid} (длина {len(patterns[pid])}) позиция {pos}")

runs = 10

cpu_time_sum = 0
gpu_time_sum = 0

print(f"\nЗАПУСК ({runs} раз):")

for _ in range(runs):
    t_cpu = time.time()
    cpu_results = mass_search_cpu(buf, patterns)
    cpu_time_sum += (time.time() - t_cpu) * 1000

    t_gpu = time.time()
    gpu_results = mass_search_gpu(buf, patterns)
    gpu_time_sum += (time.time() - t_gpu) * 1000

cpu_average_time = cpu_time_sum / runs
gpu_average_time = gpu_time_sum / runs
speedup = cpu_average_time / gpu_average_time

print(f"CPU время (среднее): {cpu_average_time:.2f} мс")
print(f"GPU время (среднее): {gpu_average_time:.2f} мс")
print(f"Ускорение GPU: {speedup:.2f} раз")

print("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ CPU и GPU:")

match = compare_results(cpu_results, gpu_results, args.n_patterns)

if match:
    print("Результаты CPU и GPU совпадают")
else:
    print("Результаты CPU и GPU не совпали")


print("\nРЕЗУЛЬТАТЫ ПОИСКА:")

found_count = 0

for p in range(args.n_patterns):
    positions = gpu_results.get(p, [])
    if positions:
        found_count += 1
        print(f"Подстрока {p} (длина {len(patterns[p])}): вхождений = {len(positions)}, позиции: {positions}")

print(f"\nНайдено подстрок: {found_count} из {args.n_patterns}")

print(f"\nCPU время: {cpu_average_time:.2f} мс")
print(f"GPU время: {gpu_average_time:.2f} мс")
print(f"Ускорение: {speedup:.2f} раз")
print(f"Совпадение: {'ДА' if match else 'НЕТ'}")

