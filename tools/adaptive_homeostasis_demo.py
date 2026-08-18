#!/usr/bin/env python3
"""Demo: streaming SpMM with SOL-ART adaptive schedule feedback."""
import os
import sys
import time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# Ensure .so exists
so = os.path.join(ROOT, "hbag", "libhbag.so")
if not os.path.exists(so):
    src = os.path.join(ROOT, "src", "spmm.c")
    cmd = f"gcc -O3 -march=native -mavx2 -mfma -fopenmp -Wall -fPIC -shared '{src}' -o '{so}' -lm"
    assert os.system(cmd) == 0

from hbag import spmm_hbag_native, spmm_hbag_adaptive, SOLARTEngine

MODE_NAME = {0: "dynamic", 1: "guided", 2: "static"}

def make_block(rows, cols, nnz, seed, skew=0.0):
    rng = np.random.default_rng(seed)
    # skew>0 → power-law row nnz
    if skew > 0:
        w = rng.random(rows) ** (1.0 + 3.0 * skew)
        w = w / w.sum()
        counts = rng.multinomial(nnz, w)
    else:
        counts = rng.multinomial(nnz, np.ones(rows) / rows)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    indices = np.empty(nnz, dtype=np.int64)
    data = rng.standard_normal(nnz).astype(np.float32)
    for i in range(rows):
        s, e = indptr[i], indptr[i + 1]
        if e > s:
            indices[s:e] = rng.choice(cols, size=e - s, replace=False)
    return indptr, indices, data

def main():
    rows, cols, nnz, K = 20_000, 50_000, 2_000_000, 64
    B = np.random.default_rng(100).random((cols, K), dtype=np.float32)
    eng = SOLARTEngine(base_chunk=64)
    threads = os.cpu_count() or 4

    print("=" * 78)
    print(" HBAG adaptive homeostasis demo (streaming blocks)")
    print("=" * 78)
    print(f"  rows={rows}  cols={cols}  nnz/block={nnz}  K={K}  threads={threads}")
    print()

    t_native = 0.0
    t_adapt = 0.0

    for b in range(5):
        skew = 0.0 if b < 2 else 0.7  # later blocks become irregular
        indptr, indices, data = make_block(rows, cols, nnz, seed=42 + b, skew=skew)

        t0 = time.perf_counter()
        C_n = spmm_hbag_native((indptr, indices, data), B, threads=threads)
        dt_n = (time.perf_counter() - t0) * 1000.0
        t_native += dt_n

        t0 = time.perf_counter()
        C_a, st = spmm_hbag_adaptive(
            (indptr, indices, data), B, threads=threads,
            engine=eng, t=float(b), return_state=True
        )
        dt_a = (time.perf_counter() - t0) * 1000.0
        t_adapt += dt_a

        max_err = float(np.max(np.abs(C_n - C_a)))
        print(
            f"  bloque {b+1}/5 | skew={skew:.1f} | "
            f"mode={MODE_NAME[st.schedule_mode]:7s} chunk={st.chunk:3d} | "
            f"TERM_U={st.TERM_U:.4f} CV={st.nnz_cv:.3f} | "
            f"native={dt_n:7.1f}ms adapt={dt_a:7.1f}ms | "
            f"max_err={max_err:.1e}"
        )

    print()
    print(f"  total native : {t_native:.1f} ms")
    print(f"  total adapt  : {t_adapt:.1f} ms")
    print(f"  ratio (native/adapt) : {t_native / max(t_adapt, 1e-9):.3f}x")
    print("=" * 78)
    print(" Nota: en matrices pequeñas el overhead de decisión puede dominar;")
    print(" el valor real aparece en streaming multi-GB con patrones variables.")
    print("=" * 78)

if __name__ == "__main__":
    main()
