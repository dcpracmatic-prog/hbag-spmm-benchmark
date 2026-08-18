"""
Verificacion a gran escala: spmm_hbag_native (indices de 64 bits) vs
PyTorch/MKL, sobre 500 millones de elementos no-cero (5 bloques de 100M).

NO se ejecuta en CI: el volumen de datos (~3.5 GB por la generacion
sintetica + arrays intermedios) excede la RAM tipica de un runner
gratuito de GitHub Actions, y el tiempo de ejecucion (~20-25 segundos
solo para HBAG, mas el equivalente para PyTorch/MKL) haria el pipeline
de CI lento para cada push. tests/test_correctness.py incluye una
version a escala reducida (2000x3000, K=64) de esta MISMA metodologia,
que si corre en CI.

Este archivo documenta la metodologia completa para que el resultado a
gran escala sea reproducible por cualquiera con una sesion de Kaggle,
no solo verificable de palabra. Ver BENCHMARKS.md, seccion 'Escala
grande: HBAG vs PyTorch MKL' para los resultados obtenidos.

Diseñado para correr en 3 celdas separadas de un notebook Kaggle/Jupyter
(no como script .py de una sola pieza), para que la medicion de tiempo
de cada motor (HBAG, PyTorch/MKL) no comparta cache/RAM "precalentada"
por el otro -- cada celda es un proceso Python limpio en Kaggle si se
reinicia el kernel entre celdas, o al menos un scope de memoria separado.

Uso:
    1. pip install git+https://github.com/dcpracmatic-prog/hbag-spmm-benchmark.git
    2. Copiar CELDA 1 en una celda de notebook, correr.
    3. Copiar CELDA 2 en la siguiente celda, correr.
    4. Copiar CELDA 3 en la siguiente celda, correr -- compara los
       resultados guardados por las dos celdas anteriores y reporta el
       error absoluto maximo por bloque.
"""

CELDA_1_HBAG = r'''
import os, gc, time, numpy as np
import scipy.sparse as sp
import hbag

def generate_500m_stream(num_blocks=5, block_nnz=100_000_000, cols=100_000):
    rows_per_block = 100_000
    for block_idx in range(num_blocks):
        np.random.seed(42 + block_idx)
        data = np.random.randn(block_nnz).astype(np.float32)
        indices = np.random.randint(0, cols, size=block_nnz, dtype=np.int64)
        counts = np.bincount(np.random.randint(0, rows_per_block, size=block_nnz), minlength=rows_per_block)
        indptr = np.zeros(rows_per_block + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        yield block_idx, rows_per_block, cols, block_nnz, data, indices, indptr

NUM_BLOCKS, BLOCK_NNZ, COLS, K = 5, 100_000_000, 100_000, 128
THREADS = os.cpu_count() or 4

np.random.seed(100)
B_dense = np.random.rand(COLS, K).astype(np.float32)

times = []
for b_idx, rows, cols, nnz, data, indices, indptr in generate_500m_stream(NUM_BLOCKS, BLOCK_NNZ, COLS):
    A = sp.csr_matrix((data, indices, indptr), shape=(rows, cols))

    t0 = time.perf_counter()
    C_out = hbag.spmm_hbag_native(A, B_dense, threads=THREADS)
    t_spmm = (time.perf_counter() - t0) * 1000.0
    times.append(t_spmm)

    # Guardado FUERA del bloque cronometrado -- no contamina la medicion
    np.save(f"hbag_block_{b_idx}.npy", C_out)
    print(f"Bloque {b_idx+1}/5 | SpMM HBAG: {t_spmm:.2f}ms")

    del A, data, indices, indptr, C_out
    gc.collect()

print(f"\nTiempo Total SpMM HBAG: {sum(times):.2f} ms")
'''

CELDA_2_MKL = r'''
import os, gc, time, numpy as np
import torch

def generate_500m_stream(num_blocks=5, block_nnz=100_000_000, cols=100_000):
    rows_per_block = 100_000
    for block_idx in range(num_blocks):
        np.random.seed(42 + block_idx)
        data = np.random.randn(block_nnz).astype(np.float32)
        indices = np.random.randint(0, cols, size=block_nnz, dtype=np.int64)
        counts = np.bincount(np.random.randint(0, rows_per_block, size=block_nnz), minlength=rows_per_block)
        indptr = np.zeros(rows_per_block + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        yield block_idx, rows_per_block, cols, block_nnz, data, indices, indptr

NUM_BLOCKS, BLOCK_NNZ, COLS, K = 5, 100_000_000, 100_000, 128
np.random.seed(100)
B_dense = np.random.rand(COLS, K).astype(np.float32)
B_torch = torch.from_numpy(B_dense)

times = []
for b_idx, rows, cols, nnz, data, indices, indptr in generate_500m_stream(NUM_BLOCKS, BLOCK_NNZ, COLS):
    crow = torch.from_numpy(indptr)
    col = torch.from_numpy(indices)
    val = torch.from_numpy(data)
    A_torch = torch.sparse_csr_tensor(crow, col, val, size=(rows, cols), dtype=torch.float32)

    t0 = time.perf_counter()
    C_torch = torch.matmul(A_torch, B_torch)
    t_spmm = (time.perf_counter() - t0) * 1000.0
    times.append(t_spmm)

    # Guardado FUERA del bloque cronometrado
    np.save(f"mkl_block_{b_idx}.npy", C_torch.numpy())
    print(f"Bloque {b_idx+1}/5 | SpMM MKL: {t_spmm:.2f}ms")

    del A_torch, C_torch, data, indices, indptr
    gc.collect()

print(f"\nTiempo Total SpMM MKL: {sum(times):.2f} ms")
'''

CELDA_3_VERIFICACION = r'''
import numpy as np

all_ok = True
for b_idx in range(5):
    C_hbag = np.load(f"hbag_block_{b_idx}.npy")
    C_mkl = np.load(f"mkl_block_{b_idx}.npy")
    max_err = np.abs(C_hbag - C_mkl).max()
    ok = np.allclose(C_hbag, C_mkl, rtol=1e-3, atol=1e-4)
    all_ok = all_ok and ok
    print(f"Bloque {b_idx+1}/5 | max_err={max_err:.2e} | {'OK' if ok else 'FALLA'}")

print("\nVEREDICTO:", "verificado" if all_ok else "DISCREPANCIA -- no confiar en el speedup")
'''

if __name__ == "__main__":
    print(__doc__)
    print("Este archivo es documentacion ejecutable por celdas, no un script")
    print("de una sola pieza. Copia CELDA_1_HBAG, CELDA_2_MKL y")
    print("CELDA_3_VERIFICACION (las variables de este mismo archivo) en tres")
    print("celdas separadas de un notebook Kaggle/Jupyter con GPU/CPU y")
    print("suficiente RAM (~8GB+).")
