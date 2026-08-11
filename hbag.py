import ctypes
import numpy as np
import os

# Obtener ruta absoluta de libhbag.so en el mismo directorio del script
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libhbag.so")

if not os.path.exists(_lib_path):
    raise FileNotFoundError(
        f"No se encontró 'libhbag.so' en {_lib_path}. "
        "Ejecute 'make shared' o 'make' antes de importar este módulo."
    )

_lib = ctypes.CDLL(_lib_path)

# Mapeo exacto de la estructura C 'CSRMatrix' de src/spmm.h
class CSRMatrix(ctypes.Structure):
    _fields_ = [
        ("row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("col_idx", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("nnz", ctypes.c_int)
    ]

# Configuración de firma para: void spmm_hbag_core(const CSRMatrix *A, const float *B, float *C, int n, int k)
_lib.spmm_hbag_core.argtypes = [
    ctypes.POINTER(CSRMatrix),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]
_lib.spmm_hbag_core.restype = None

def spmm_hbag(row_ptr: np.ndarray, col_idx: np.ndarray, values: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Ejecuta el kernel HBAG-Core SpMM sobre arreglos NumPy.
    """
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
