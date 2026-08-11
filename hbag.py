import ctypes
import numpy as np
import os

# 1. Cargar la librería compartida compilada
lib_path = os.path.abspath("libhbag.so")
if not os.path.exists(lib_path):
    raise FileNotFoundError("No se encontró libhbag.so. Ejecute 'make' primero.")

_lib = ctypes.CDLL(lib_path)

# 2. Definir la estructura CSRMatrix de C en Python
class _CCSRMatrix(ctypes.Structure):
    _fields_ = [
        ("row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("col_idx", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("nnz", ctypes.c_int)
    ]

# Configurar firmas de las funciones de C
_lib.spmm_hbag_core.argtypes = [
    ctypes.POINTER(_CCSRMatrix),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int
]

class HBAGSpMM:
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def compute(self, row_ptr, col_idx, values, b_matrix):
        """
        Ejecuta la multiplicación SpMM optimizada por HBAG-Core.
        Recibe arreglos de NumPy o listas convertibles.
        """
        nnz = len(values)
        
        # Preparar estructura C
        c_A = _CCSRMatrix(
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nnz
        )
        
        # Preparar matrices B (entrada) y C (salida)
        b_flat = np.ascontiguousarray(b_matrix, dtype=np.float32)
        c_flat = np.zeros(self.n * self.k, dtype=np.float32)
        
        # Llamar al núcleo optimizado en C
        _lib.spmm_hbag_core(
            ctypes.byref(c_A),
            b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n,
            self.k
        )
        
        return c_flat.reshape((self.n, self.k))
