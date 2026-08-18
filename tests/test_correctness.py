import csv as _csv
import os
import subprocess

import numpy as np
import pytest
from scipy.sparse import random as sp_random

import hbag

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [(128, 32, 0.05), (512, 64, 0.05), (2048, 128, 0.02),
           (2048, 512, 0.02), (64, 1, 0.10)]


@pytest.mark.parametrize("n,k,density", CONFIGS)
def test_spmm_hbag_matches_scipy(n, k, density):
    A = sp_random(n, n, density=density, format="csr", dtype=np.float32, random_state=42)
    B = np.random.default_rng(42).random((n, k)).astype(np.float32)
    C = hbag.spmm_hbag(A, B)
    C_ref = A.dot(B)
    np.testing.assert_allclose(C, C_ref, rtol=1e-3, atol=1e-4)


UGLY_K = [(512, 128, 0.02), (512, 100, 0.02), (512, 67, 0.02),
          (512, 7, 0.02), (512, 1, 0.02)]


@pytest.mark.parametrize("n,k,density", UGLY_K)
def test_spmm_hbag_omp_matches_scipy(n, k, density):
    if not hasattr(hbag, "spmm_hbag_omp"):
        pytest.skip("spmm_hbag_omp ausente")
    A = sp_random(n, n, density=density, format="csr", dtype=np.float32, random_state=7)
    B = np.random.default_rng(7).random((n, k)).astype(np.float32)
    try:
        C = hbag.spmm_hbag_omp(A, B)
    except RuntimeError:
        pytest.skip("sin -fopenmp")
    np.testing.assert_allclose(C, A.dot(B), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("n,k,density", UGLY_K)
def test_spmm_hbag_native_matches_scipy(n, k, density):
    if not hasattr(hbag, "spmm_hbag_native"):
        pytest.skip("spmm_hbag_native ausente")
    A = sp_random(n, n, density=density, format="csr", dtype=np.float32, random_state=11)
    B = np.random.default_rng(11).random((n, k)).astype(np.float32)
    try:
        C = hbag.spmm_hbag_native(A, B)
    except RuntimeError:
        pytest.skip("sin int64")
    np.testing.assert_allclose(C, A.dot(B), rtol=1e-3, atol=1e-4)


def test_spmm_hbag_native_explicit_threads():
    if not hasattr(hbag, "spmm_hbag_native"):
        pytest.skip()
    n, k, d = 512, 64, 0.02
    A = sp_random(n, n, density=d, format="csr", dtype=np.float32, random_state=13)
    B = np.random.rand(n, k).astype(np.float32)
    try:
        Cd = hbag.spmm_hbag_native(A, B)
        Ce = hbag.spmm_hbag_native(A, B, threads=2)
    except RuntimeError:
        pytest.skip()
    ref = A.dot(B)
    np.testing.assert_allclose(Cd, ref, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(Ce, ref, rtol=1e-3, atol=1e-4)


def test_spmm_hbag_native_matches_torch_small():
    torch = pytest.importorskip("torch")
    if not hasattr(hbag, "spmm_hbag_native"):
        pytest.skip()
    rows, cols, k, d = 2000, 3000, 64, 0.02
    A = sp_random(rows, cols, density=d, format="csr", dtype=np.float32, random_state=17)
    B = np.random.rand(cols, k).astype(np.float32)
    try:
        C_hbag = hbag.spmm_hbag_native(A, B)
    except RuntimeError:
        pytest.skip()
    crow = torch.from_numpy(A.indptr.astype(np.int64))
    col = torch.from_numpy(A.indices.astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    A_t = torch.sparse_csr_tensor(crow, col, val, size=(rows, cols), dtype=torch.float32)
    B_t = torch.from_numpy(B)
    np.testing.assert_allclose(C_hbag, torch.matmul(A_t, B_t).numpy(), rtol=1e-3, atol=1e-4)


def test_spmm_hbag_zero_matrix():
    n, k = 64, 16
    A = sp_random(n, n, density=0.0, format="csr", dtype=np.float32)
    B = np.random.rand(n, k).astype(np.float32)
    assert np.allclose(hbag.spmm_hbag(A, B), 0.0)


def test_bench_c_compiles_and_runs():
    subprocess.run(["make", "bench"], cwd=REPO, check=True, capture_output=True)
    r = subprocess.run([os.path.join(REPO, "bench")],
                       capture_output=True, text=True, check=True)
    assert "FALLA" not in r.stdout


def test_bench_c_self_test_passes():
    subprocess.run(["make", "bench"], cwd=REPO, check=True, capture_output=True)
    r = subprocess.run([os.path.join(REPO, "bench"), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"--self-test fallo: STDOUT={r.stdout} STDERR={r.stderr}"


def test_bench_c_csv_output_is_stable():
    """Verifica que el CSV del bench tiene la estructura correcta: header
    canónico, 6 filas en single-shot, status OK en todas, y t_ms > 0.
    t_ms bit-bit NO es assertable con OMP (jitter OS ~3%); lo que SÍ es
    estable entre dos corridas del mismo binario determinista es la cabina:
    shapes (N,K,density,nnz), threads, kernel, max_err, status."""
    subprocess.run(["make", "bench"], cwd=REPO, check=True, capture_output=True)
    BIN = os.path.join(REPO, "bench")
    for p in ("/tmp/repro_a.csv", "/tmp/repro_b.csv"):
        if os.path.exists(p):
            os.unlink(p)
    cmd = [BIN, "--rows", "512", "--cols", "512", "--k", "64",
           "--density", "0.05", "--trials", "1", "--threads", "1", "--csv"]
    a = subprocess.run([*cmd, "/tmp/repro_a.csv"], cwd=REPO, capture_output=True, text=True, check=True)
    b = subprocess.run([*cmd, "/tmp/repro_b.csv"], cwd=REPO, capture_output=True, text=True, check=True)

    EXPECTED = ["config", "label", "N", "K", "density", "nnz",
                "threads", "kernel", "t_ms", "max_err", "status"]
    rows_a = list(_csv.reader(open("/tmp/repro_a.csv")))
    rows_b = list(_csv.reader(open("/tmp/repro_b.csv")))

    # Cabecera canonica en ambos archivos
    assert rows_a[0] == rows_b[0] == EXPECTED, f"header invalido: A={rows_a[0]} B={rows_b[0]}"
    # 1 single-shot config x 6 kernels = 6 filas + header
    assert len(rows_a) == len(rows_b) == 7, f"filas: a={len(rows_a)} b={len(rows_b)}"
    # Todas las filas terminaron OK en ambas corridas
    for r in rows_a[1:]:
        assert r[-1] == "OK", f"fila A no OK: {r}"
    for r in rows_b[1:]:
        assert r[-1] == "OK", f"fila B no OK: {r}"
    # Cabin deterministic (N, K, density, nnz, threads, kernel, max_err, status)
    # debe coincidir bit-a-bit
    cabin = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    cabin_a = sorted(tuple(r[i] for i in cabin) for r in rows_a[1:])
    cabin_b = sorted(tuple(r[i] for i in cabin) for r in rows_b[1:])
    assert cabin_a == cabin_b, "cabecera determinista (no t_ms) cambia entre corridas"
    # t_ms > 0 y razonable
    for r in rows_a[1:] + rows_b[1:]:
        assert float(r[8]) > 0.0, f"t_ms no positivo: {r}"


def test_make_test_c_passes():
    subprocess.run(["make", "bench"], cwd=REPO, check=True, capture_output=True)
    r = subprocess.run(["make", "test-c"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, f"make test-c fallo: STDOUT={r.stdout} STDERR={r.stderr}"


def test_make_shared_exports_symbols():
    """nm -D --defined-only sobre libhbag.so debe enumerar los 4 simbolos
    publicos que requiere hbag/__init__.py."""
    subprocess.run(["make", "clean"], cwd=REPO, check=True, capture_output=True)
    subprocess.run(["make", "hbag/libhbag.so"], cwd=REPO, check=True, capture_output=True)
    so = os.path.join(REPO, "hbag", "libhbag.so")
    assert os.path.exists(so)
    nm = subprocess.run(["nm", "-D", "--defined-only", so],
                       capture_output=True, text=True, check=True)
    syms = set()
    for ln in nm.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[1] in ("T", "W", "R"):
            syms.add(parts[2])
    for needed in ("spmm_hbag_core", "spmm_hbag_core_omp",
                   "spmm_hbag_core_omp64", "spmm_hbag_core_omp64_adaptive",
                   "spmm_csr_std", "generate_sparse"):
        assert needed in syms, f"falta {needed} (encontrados: {sorted(syms)})"
