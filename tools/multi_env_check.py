"""
Diagnostico multi-entorno para spmm_hbag_omp.

Uso: pegar en una celda de Colab/Kaggle (despues de instalar hbag) y
correr tal cual, sin modificar nada, en cada variante de acelerador
(CPU/GPU/TPU). Imprime os.cpu_count() junto a cada resultado -- esa
linea es la que explica cualquier diferencia entre sesiones, no el
nombre del acelerador seleccionado (este kernel es C puro, no usa
GPU ni TPU).

Requiere: pip install git+https://github.com/dcpracmatic-prog/hbag-spmm-benchmark.git
"""
import os
import subprocess
import time
import ctypes
import gc
import numpy as np
import scipy.sparse as sp
import statistics

nproc = os.cpu_count()
print(f"os.cpu_count() = {nproc}")
try:
    with open("/proc/cpuinfo") as f:
        model = [l for l in f if "model name" in l][0].strip()
    print(model)
except Exception:
    pass
print(f"OMP_NUM_THREADS env = {os.environ.get('OMP_NUM_THREADS', '(no seteado)')}")
print()

KERNEL_SRC = "hbag_fusion_kernel.c"
KERNEL_LIB = "./hbag_fusion_kernel.so"
TRIALS = 20
WARMUP = 5


def compile_fusion_kernel():
    c_code = """
    #include <immintrin.h>
    #include <string.h>
    #include <omp.h>

    void spmm_fusion_kernel(int M, int N, const int *row_ptr, const int *col_idx,
                            const float *values, const float *dense_B, float *dense_C) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < M; i++) {
            float *c_row = &dense_C[i * N];
            int row_start = row_ptr[i];
            int row_end   = row_ptr[i + 1];
            memset(c_row, 0, N * sizeof(float));
            for (int p = row_start; p < row_end; p++) {
                int col = col_idx[p];
                float val = values[p];
                __m256 v_val = _mm256_set1_ps(val);
                const float *b_row = &dense_B[col * N];
                int j = 0;
                for (; j <= N - 16; j += 16) {
                    _mm256_storeu_ps(&c_row[j], _mm256_fmadd_ps(v_val, _mm256_loadu_ps(&b_row[j]), _mm256_loadu_ps(&c_row[j])));
                    _mm256_storeu_ps(&c_row[j+8], _mm256_fmadd_ps(v_val, _mm256_loadu_ps(&b_row[j+8]), _mm256_loadu_ps(&c_row[j+8])));
                }
                for (; j < N; j++) c_row[j] += val * b_row[j];
            }
        }
    }
    """
    with open(KERNEL_SRC, "w") as f:
        f.write(c_code)
    cmd = f"gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -fPIC {KERNEL_SRC} -o {KERNEL_LIB}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Fallo de compilacion: {res.stderr}")


def run_suite():
    compile_fusion_kernel()
    lib = ctypes.CDLL(KERNEL_LIB)
    lib.spmm_fusion_kernel.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]

    configs = [(512, 64, 0.05), (2048, 128, 0.02), (4096, 128, 0.01), (2048, 67, 0.02)]

    print(f"{'Config (N,K,D)':<22} | {'SciPy (ms)':<12} | {'Fusion (ms)':<12} | {'Ratio':<8} | {'Validacion'}")
    print("-" * 84)

    for n, k, d in configs:
        gc.collect()
        A = sp.random(n, n, density=d, format="csr", dtype=np.float32, random_state=42)
        B = np.random.RandomState(42).rand(n, k).astype(np.float32)
        ref_C = A.dot(B)

        for _ in range(WARMUP):
            _ = A @ B
        scipy_times = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            _ = A @ B
            scipy_times.append((time.perf_counter() - t0) * 1000)

        C_out = np.zeros((n, k), dtype=np.float32)
        ptr_r = A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ptr_c = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ptr_v = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_B = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_C = C_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        for _ in range(WARMUP):
            lib.spmm_fusion_kernel(n, k, ptr_r, ptr_c, ptr_v, ptr_B, ptr_C)

        lib.spmm_fusion_kernel(n, k, ptr_r, ptr_c, ptr_v, ptr_B, ptr_C)
        ok = np.allclose(C_out, ref_C, rtol=1e-4, atol=1e-4)
        max_err = np.abs(C_out - ref_C).max()

        fusion_times = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            lib.spmm_fusion_kernel(n, k, ptr_r, ptr_c, ptr_v, ptr_B, ptr_C)
            fusion_times.append((time.perf_counter() - t0) * 1000)

        t_scipy = statistics.median(scipy_times)
        t_fusion = statistics.median(fusion_times)
        ratio = t_scipy / t_fusion
        status = f"OK (err={max_err:.1e})" if ok else f"FALLA (err={max_err:.1e})"

        print(f"N={n}, K={k}, D={d*100:.0f}%".ljust(22) +
              f" | {t_scipy:12.3f} | {t_fusion:12.3f} | {ratio:7.2f}x | {status}")

    for f in [KERNEL_SRC, KERNEL_LIB]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    print(f"=== nucleos reportados: {nproc} ===\n")
    run_suite()
