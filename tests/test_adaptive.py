"""Correctness + schedule decision tests for homeostasis-adaptive SpMM."""
import os
import sys
import numpy as np

# Allow running without install when lib is built in-tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hbag import (
    spmm_hbag_native,
    spmm_hbag_adaptive,
    SOLARTEngine,
)


def _make_csr(rows=2000, cols=3000, density=0.02, seed=7, unbalanced=False):
    rng = np.random.default_rng(seed)
    if unbalanced:
        # Power-law-ish row densities
        probs = rng.random(rows) ** 3
        probs = probs / probs.sum() * (rows * cols * density)
        nnz_per_row = np.clip(probs.astype(int), 1, cols)
    else:
        nnz_per_row = rng.poisson(cols * density, size=rows)
        nnz_per_row = np.clip(nnz_per_row, 0, cols)

    indptr = np.zeros(rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(nnz_per_row)
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int64)
    data = rng.standard_normal(nnz).astype(np.float32)
    for i in range(rows):
        s, e = indptr[i], indptr[i + 1]
        indices[s:e] = rng.choice(cols, size=e - s, replace=False)
    return indptr, indices, data, cols


def test_adaptive_matches_native_bitexact():
    indptr, indices, data, cols = _make_csr(unbalanced=True)
    K = 64
    B = np.random.default_rng(0).random((cols, K), dtype=np.float32)

    C_ref = spmm_hbag_native((indptr, indices, data), B, threads=2)
    eng = SOLARTEngine()
    C_ad, state = spmm_hbag_adaptive(
        (indptr, indices, data), B, threads=2, engine=eng, return_state=True
    )

    assert C_ref.shape == C_ad.shape
    max_err = float(np.max(np.abs(C_ref - C_ad)))
    assert max_err == 0.0, f"max_err={max_err} (must be bit-exact)"
    assert 0.0 <= state.TERM_U <= 1.5
    assert state.schedule_mode in (0, 1, 2, 3)
    assert state.chunk >= eng.min_chunk


def test_schedule_reacts_to_irregularity():
    eng = SOLARTEngine(base_chunk=64)
    # Uniform → prefer static/guided with larger chunk
    uniform = np.arange(0, 1001, 10, dtype=np.int64)  # 100 rows, 10 nnz each
    cv_u = eng.row_nnz_irregularity(uniform)
    chunk_u, mode_u = eng.decide_schedule(term_u=0.85, nnz_cv=cv_u, rows=100)

    # Highly irregular
    irreg = np.array([0] + sorted(np.random.default_rng(1).integers(1, 500, size=100).cumsum().tolist()), dtype=np.int64)
    cv_i = eng.row_nnz_irregularity(irreg)
    chunk_i, mode_i = eng.decide_schedule(term_u=0.40, nnz_cv=cv_i, rows=100)

    assert cv_i > cv_u
    assert mode_i == 0  # dynamic under stress
    assert chunk_i <= chunk_u


def test_engine_history_streaming():
    eng = SOLARTEngine()
    indptr, indices, data, cols = _make_csr(rows=500, cols=800, density=0.03)
    B = np.random.default_rng(3).random((cols, 32), dtype=np.float32)
    for t in range(3):
        _, state = spmm_hbag_adaptive(
            (indptr, indices, data), B, engine=eng, t=float(t), return_state=True
        )
    assert len(eng.history) == 3
    assert eng.last() is not None


def test_balanced_mode_bitexact():
    """schedule_mode=3 (partición exacta por nnz) debe dar el mismo
    resultado que el kernel clásico, forzado explícitamente y sin pasar
    por el engine."""
    indptr, indices, data, cols = _make_csr(unbalanced=True)
    K = 64
    B = np.random.default_rng(0).random((cols, K), dtype=np.float32)

    C_ref = spmm_hbag_native((indptr, indices, data), B, threads=4)
    C_bal = spmm_hbag_adaptive(
        (indptr, indices, data), B, threads=4, chunk=64, schedule_mode=3
    )

    max_err = float(np.max(np.abs(C_ref - C_bal)))
    assert max_err == 0.0, f"max_err={max_err} (must be bit-exact)"


def test_schedule_picks_balanced_under_extreme_irregularity():
    eng = SOLARTEngine(base_chunk=64)
    # CV >= 1.0 forzado a mano: 190 filas vacias + 10 filas con 500 nnz
    # cada una -- irregularidad extrema y deterministica, no depende de
    # semilla aleatoria.
    nnz_per_row = np.array([0] * 190 + [500] * 10, dtype=np.int64)
    indptr = np.zeros(201, dtype=np.int64)
    indptr[1:] = np.cumsum(nnz_per_row)
    cv = eng.row_nnz_irregularity(indptr)
    assert cv >= 1.0, f"cv={cv} (test fixture debe producir CV alto)"
    chunk, mode = eng.decide_schedule(term_u=0.5, nnz_cv=cv, rows=200)
    assert mode == 3


if __name__ == "__main__":
    # Build shared lib if missing
    root = os.path.join(os.path.dirname(__file__), "..")
    so = os.path.join(root, "hbag", "libhbag.so")
    if not os.path.exists(so):
        src = os.path.join(root, "src", "spmm.c")
        cmd = (
            f"gcc -O3 -march=native -mavx2 -mfma -fopenmp -Wall -fPIC -shared "
            f"'{src}' -o '{so}' -lm"
        )
        print("[build]", cmd)
        assert os.system(cmd) == 0, "compile failed"

    test_adaptive_matches_native_bitexact()
    print("[OK] adaptive matches native bit-exact")
    test_schedule_reacts_to_irregularity()
    print("[OK] schedule reacts to irregularity")
    test_engine_history_streaming()
    print("[OK] engine history / streaming")
    test_balanced_mode_bitexact()
    print("[OK] balanced mode (3) bit-exact")
    test_schedule_picks_balanced_under_extreme_irregularity()
    print("[OK] engine escalates to balanced mode under extreme CV")
    print("ALL ADAPTIVE TESTS PASSED")
