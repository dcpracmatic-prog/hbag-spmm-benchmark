"""Adaptive HBAG correctness and scheduling tests.

This suite tests the real C/OpenMP implementation through the public Python API.
It deliberately separates:

1. row-NNZ irregularity measurement;
2. adaptive schedule selection;
3. deterministic balanced-mode execution;
4. bit-exact numerical equivalence with the native 64-bit kernel;
5. pathological heavy-row cases, where exact equal NNZ per thread is impossible
   because CSR rows cannot be split.

The tests do NOT claim that balanced mode produces equal NNZ per thread.
The implementation chooses deterministic row boundaries near cumulative-NNZ
targets while preserving CSR row boundaries.
"""

import os
import sys

import numpy as np

# Allow running against an in-tree build.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hbag import SOLARTEngine, spmm_hbag_adaptive, spmm_hbag_native


def _make_csr_from_row_nnz(row_nnz, cols=1024, seed=0):
    """Build a valid CSR tuple from an explicit NNZ-per-row vector."""
    row_nnz = np.asarray(row_nnz, dtype=np.int64)
    if np.any(row_nnz < 0):
        raise ValueError("row_nnz must be non-negative")
    if np.any(row_nnz > cols):
        raise ValueError("a row cannot contain more than cols unique entries")

    rng = np.random.default_rng(seed)
    rows = row_nnz.size
    indptr = np.zeros(rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_nnz)

    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int64)
    data = rng.standard_normal(nnz).astype(np.float32)

    for i, count in enumerate(row_nnz):
        start, end = int(indptr[i]), int(indptr[i + 1])
        if count:
            indices[start:end] = np.sort(
                rng.choice(cols, size=int(count), replace=False)
            )

    return indptr, indices, data, cols


def _make_power_law_csr(
    rows=1024,
    cols=1024,
    avg_density=0.01,
    alpha=2.2,
    seed=1,
):
    """Generate a deterministic, strongly skewed CSR workload."""
    rng = np.random.default_rng(seed)

    raw = rng.zipf(alpha, size=rows).astype(np.float64)
    target_mean = cols * avg_density
    raw = raw / raw.mean() * target_mean
    row_nnz = np.clip(raw.astype(np.int64), 0, cols)

    return _make_csr_from_row_nnz(row_nnz, cols=cols, seed=seed + 1000)


def _cv_from_indptr(indptr):
    nnz_per_row = np.diff(np.asarray(indptr, dtype=np.int64)).astype(np.float64)
    mean = float(np.mean(nnz_per_row))
    if mean == 0.0:
        return 0.0
    return float(np.std(nnz_per_row) / mean)


def _make_dense(rows, cols, seed=0):
    return np.random.default_rng(seed).random(
        (rows, cols), dtype=np.float32
    )


def test_uniform_workload_is_not_marked_extreme():
    """Uniform row work should have CV≈0 and must not select balanced mode."""
    eng = SOLARTEngine(base_chunk=64)

    row_nnz = np.full(128, 16, dtype=np.int64)
    indptr = np.zeros(row_nnz.size + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_nnz)

    cv = eng.row_nnz_irregularity(indptr)
    chunk, mode = eng.decide_schedule(
        term_u=0.85,
        nnz_cv=cv,
        rows=row_nnz.size,
    )

    assert cv == 0.0
    assert mode == 2  # static
    assert chunk >= eng.min_chunk


def test_extreme_irregularity_selects_balanced():
    """CV>=1 is the documented threshold for mode 3."""
    eng = SOLARTEngine(base_chunk=64)

    row_nnz = np.array([0] * 190 + [500] * 10, dtype=np.int64)
    indptr = np.zeros(row_nnz.size + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_nnz)

    cv = eng.row_nnz_irregularity(indptr)
    chunk, mode = eng.decide_schedule(
        term_u=0.50,
        nnz_cv=cv,
        rows=row_nnz.size,
    )

    assert cv >= 1.0
    assert mode == 3
    assert eng.min_chunk <= chunk <= eng.max_chunk


def test_power_law_workload_reveals_irregularity():
    """The power-law fixture must actually stress the scheduler."""
    indptr, _, _, _ = _make_power_law_csr()

    cv = _cv_from_indptr(indptr)
    eng = SOLARTEngine()

    chunk, mode = eng.decide_schedule(
        term_u=0.55,
        nnz_cv=cv,
        rows=len(indptr) - 1,
    )

    assert cv > 1.0, f"fixture is not sufficiently irregular: CV={cv:.3f}"
    assert mode == 3
    assert eng.min_chunk <= chunk <= eng.max_chunk


def test_adaptive_auto_selection_matches_engine_decision():
    """The public adaptive API must use the same mode chosen by SOLARTEngine."""
    indptr, indices, data, cols = _make_power_law_csr(
        rows=1024, cols=1024, avg_density=0.01, seed=1
    )
    B = _make_dense(cols, 32, seed=42)

    eng = SOLARTEngine()
    expected_cv = eng.row_nnz_irregularity(indptr)
    _, expected_mode = eng.decide_schedule(
        term_u=0.55, nnz_cv=expected_cv, rows=len(indptr) - 1
    )

    _, state = spmm_hbag_adaptive(
        (indptr, indices, data),
        B,
        threads=2,
        engine=eng,
        return_state=True,
    )

    assert state.nnz_cv == expected_cv
    assert state.schedule_mode == expected_mode
    assert state.schedule_mode == 3


def test_adaptive_matches_native_bitexact_on_power_law_workload():
    """Adaptive execution must be bit-identical to the native 64-bit kernel."""
    indptr, indices, data, cols = _make_power_law_csr(
        rows=1024,
        cols=1024,
        avg_density=0.01,
        alpha=2.2,
        seed=1,
    )
    B = _make_dense(cols, 64, seed=7)

    C_ref = spmm_hbag_native(
        (indptr, indices, data), B, threads=2
    )

    eng = SOLARTEngine()
    C_adaptive, state = spmm_hbag_adaptive(
        (indptr, indices, data),
        B,
        threads=2,
        engine=eng,
        return_state=True,
    )

    assert C_adaptive.shape == C_ref.shape
    assert np.array_equal(
        C_adaptive, C_ref
    ), "adaptive kernel changed float32 results"
    assert np.all(np.isfinite(C_adaptive))
    assert state.schedule_mode == 3


def test_all_explicit_schedules_are_bitexact():
    """Dynamic, guided, static and balanced must preserve row-local arithmetic."""
    indptr, indices, data, cols = _make_power_law_csr(
        rows=512,
        cols=512,
        avg_density=0.03,
        alpha=2.2,
        seed=11,
    )
    B = _make_dense(cols, 32, seed=19)

    C_ref = spmm_hbag_native(
        (indptr, indices, data), B, threads=4
    )

    for mode in (0, 1, 2, 3):
        C = spmm_hbag_adaptive(
            (indptr, indices, data),
            B,
            threads=4,
            chunk=64,
            schedule_mode=mode,
        )

        assert np.array_equal(
            C, C_ref
        ), f"schedule_mode={mode} is not bit-exact"
        assert np.all(np.isfinite(C))


def test_balanced_mode_handles_single_heavy_row():
    """A huge indivisible row must not make balanced mode fail.

    This is intentionally NOT an equality-of-NNZ test: one CSR row cannot
    be split across threads by the current implementation.
    """
    row_nnz = np.array([0] * 31 + [1024] + [0] * 32, dtype=np.int64)
    indptr, indices, data, cols = _make_csr_from_row_nnz(
        row_nnz, cols=1024, seed=23
    )
    B = _make_dense(cols, 16, seed=29)

    C_ref = spmm_hbag_native(
        (indptr, indices, data), B, threads=4
    )
    C_balanced = spmm_hbag_adaptive(
        (indptr, indices, data),
        B,
        threads=4,
        chunk=64,
        schedule_mode=3,
    )

    assert np.array_equal(C_balanced, C_ref)
    assert np.all(np.isfinite(C_balanced))


def test_zero_nnz_matrix_is_stable_under_adaptive_path():
    """Empty CSR input must remain finite and numerically zero."""
    row_nnz = np.zeros(128, dtype=np.int64)
    indptr, indices, data, cols = _make_csr_from_row_nnz(
        row_nnz, cols=256, seed=31
    )
    B = _make_dense(cols, 8, seed=37)

    C, state = spmm_hbag_adaptive(
        (indptr, indices, data),
        B,
        threads=2,
        return_state=True,
    )

    assert np.array_equal(C, np.zeros_like(C))
    assert np.all(np.isfinite(C))
    assert state.nnz_cv == 0.0
    assert state.schedule_mode in (0, 1, 2, 3)


def test_engine_history_is_streaming_and_deterministic():
    """Reusing an engine records one governance state per processed block."""
    indptr, indices, data, cols = _make_power_law_csr(
        rows=256,
        cols=256,
        avg_density=0.03,
        seed=41,
    )
    B = _make_dense(cols, 16, seed=43)

    eng = SOLARTEngine()
    states = []

    for t in range(3):
        _, state = spmm_hbag_adaptive(
            (indptr, indices, data),
            B,
            threads=2,
            engine=eng,
            t=float(t),
            return_state=True,
        )
        states.append(state)

    assert len(eng.history) == 3
    assert eng.last() is states[-1]
    assert all(state.nnz_cv >= 0.0 for state in states)
    assert all(state.schedule_mode in (0, 1, 2, 3) for state in states)


if __name__ == "__main__":
    test_uniform_workload_is_not_marked_extreme()
    test_extreme_irregularity_selects_balanced()
    test_power_law_workload_reveals_irregularity()
    test_adaptive_auto_selection_matches_engine_decision()
    test_adaptive_matches_native_bitexact_on_power_law_workload()
    test_all_explicit_schedules_are_bitexact()
    test_balanced_mode_handles_single_heavy_row()
    test_zero_nnz_matrix_is_stable_under_adaptive_path()
    test_engine_history_is_streaming_and_deterministic()
    print("ALL ADAPTIVE TESTS PASSED")
