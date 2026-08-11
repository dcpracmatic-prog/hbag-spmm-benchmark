import ctypes
import numpy as np
import os
import subprocess

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libhbag.so")

# Fallback: Si se importa sin haber instalado via pip, intenta autocompilar localmente
if not os.path.exists(_lib_path):
    _src_c = os.path.abspath(os.path.join(_dir, "..", "src", "spmm.c"))
    if os.path.exists(_src_c):
        cmd = f"gcc -O3 -march=native -Wall -fPIC -shared {_src_c} -o {_lib_path} -lm"
        subprocess.run(cmd, shell=True, check=True)
    else:
        raise FileNotFoundError(f"No se encontró el binario compilado ni el fuente C en {_src_c}")

_lib = ctypes.CDLL(_lib_path)

class CSRMatrix(ctypes.Structure):
    _fields_ = [
        ("row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("col_idx", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("nnz", ctypes.c_int)
    ]

_lib.spmm_hbag_core.argtypes = [
    ctypes.POINTER(CSRMatrix),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]
_lib.spmm_hbag_core.restype = None

def spmm_hbag(A_sparse, B: np.ndarray) -> np.ndarray:
    """
    Multiplica una matriz dispersa (SciPy csr_matrix o tupla CSR) 
    por una matriz densa B utilizando el motor acelerado HBAG-Core.
    """
    if hasattr(A_sparse, 'indptr'):
        row_ptr = A_sparse.indptr
        col_idx = A_sparse.indices
        values = A_sparse.data
    else:
        row_ptr, col_idx, values = A_sparse

    row_ptr_c = np.ascontiguousarray(row_ptr, dtype=np.int32)
    col_idx_c = np.ascontiguousarray(col_idx, dtype=np.int32)
    values_c = np.ascontiguousarray(values, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)

    N = len(row_ptr_c) - 1
    K = B_c.shape[1] if B_c.ndim > 1 else 1

    A_struct = CSRMatrix(
        row_ptr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(values_c)
    )

    C = np.zeros((N, K), dtype=np.float32)

    _lib.spmm_hbag_core(
        ctypes.byref(A_struct),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N,
        K
    )
    return C
