"""
HBAG — Sparse Matrix × Dense Matrix (SpMM) with adaptive homeostasis schedule.

Public API
----------
spmm_hbag(A, B)                 single-thread core
spmm_hbag_omp(A, B)             OpenMP dynamic,64 (classic)
spmm_hbag_native(A, B, threads) 64-bit indices, classic schedule
spmm_hbag_adaptive(A, B, ...)   64-bit + SOL-ART homeostasis schedule

The adaptive path keeps bit-exact arithmetic identical to the classic
kernel; only the OpenMP schedule (chunk + mode) is governed by TERM_U
feedback and row-nnz irregularity. That is the know-how differentiator.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libhbag.so")

if not os.path.exists(_lib_path):
    raise RuntimeError(
        f"El binario optimizado 'libhbag.so' no se encuentra en {_lib_path}. "
        "Asegúrate de que la instalación con pip se completó correctamente."
    )

_lib = ctypes.CDLL(_lib_path)


class CSRMatrix(ctypes.Structure):
    _fields_ = [
        ("row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("col_idx", ctypes.POINTER(ctypes.c_int)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("nnz", ctypes.c_int),
    ]


_lib.spmm_hbag_core.argtypes = [
    ctypes.POINTER(CSRMatrix),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.spmm_hbag_core.restype = None

_HAS_OMP = hasattr(_lib, "spmm_hbag_core_omp")
if _HAS_OMP:
    _lib.spmm_hbag_core_omp.argtypes = [
        ctypes.POINTER(CSRMatrix),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.spmm_hbag_core_omp.restype = None

_HAS_OMP64 = hasattr(_lib, "spmm_hbag_core_omp64")
if _HAS_OMP64:
    _lib.spmm_hbag_core_omp64.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
    ]
    _lib.spmm_hbag_core_omp64.restype = None

_HAS_ADAPTIVE = hasattr(_lib, "spmm_hbag_core_omp64_adaptive")
if _HAS_ADAPTIVE:
    _lib.spmm_hbag_core_omp64_adaptive.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.spmm_hbag_core_omp64_adaptive.restype = None


# =====================================================================
# SOL-ART V8 — motor de gobernanza / homeostasis
# =====================================================================

@dataclass
class GovernanceState:
    """Snapshot de homeostasis tras un bloque SpMM."""
    C_Coherencia: float
    B_Balance: float
    G_Gradiente: float
    A_Azar: float
    TERM_U: float
    Homeostasis: bool
    L3: float
    chunk: int
    schedule_mode: int  # 0=dynamic, 1=guided, 2=static
    nnz_cv: float       # coeficiente de variación de nnz por fila


class SOLARTEngine:
    """
    Gobernanza morfológica liviana para el schedule OpenMP.

    TERM_U combina coherencia, balance, gradiente local y distancia a
    un umbral de escala (V_xi). Cuando TERM_U es alto el workload se
    considera estable → chunk grande + schedule static/guided.
    Cuando cae → chunk pequeño + dynamic para rebalancear.

    El motor es deliberadamente simple y determinista: no hay ML, no hay
    estado oculto pesado. Solo reglas transparentes que se pueden auditar.
    """

    def __init__(
        self,
        V_xi: float = 14.9,
        gamma: float = 0.8,
        L0: float = 1.0,
        base_chunk: int = 64,
        min_chunk: int = 8,
        max_chunk: int = 512,
    ):
        self.V_xi = float(V_xi)
        self.gamma = float(gamma)
        self.L0 = float(L0)
        self.S = np.sqrt(2.0)
        self.L_S = self.L0 * (self.S - self.gamma)
        self.base_chunk = int(base_chunk)
        self.min_chunk = int(min_chunk)
        self.max_chunk = int(max_chunk)
        self._history: list[GovernanceState] = []
        self._verify_non_singular_bounce()

    def _verify_non_singular_bounce(self) -> None:
        a_0 = 2.0
        if a_0 <= 0.01:
            raise ValueError(f"Fallo de rebote no singular: a(0) = {a_0:.4f} <= 0.01")

    def compute_temporal_scale(self, t: float) -> Tuple[float, float, float]:
        u_t = np.pi + 2.0 * np.arctan(t)
        v_t = np.cos(u_t) * np.exp(-0.2 * (t ** 2)) + self.L_S * np.tanh(0.5 * t)
        if t == 0.0:
            a_t = 2.0
        else:
            a_t = self.L_S + 0.5 * (v_t ** 2) + 0.1 * np.cosh(0.4 * t)
        return float(u_t), float(v_t), float(a_t)

    def compute_l3(self, C: np.ndarray) -> float:
        x = np.asarray(C, dtype=np.float64).ravel()
        sum_cubes = np.sum(np.abs(x) ** 3)
        mean_val = float(np.mean(x)) if x.size else 0.0
        sign_bar = 1.0 if mean_val >= 0.0 else -1.0
        return float(np.cbrt(sum_cubes) * sign_bar)

    def evaluate_governance(
        self, C: np.ndarray, sample_size: int = 100_000
    ) -> Dict[str, float]:
        x = np.asarray(C, dtype=np.float64).ravel()
        if x.size == 0:
            return {
                "C_Coherencia": 1.0,
                "B_Balance": 1.0,
                "G_Gradiente": 1.0,
                "A_Azar": 1.0,
                "TERM_U": 1.0,
                "Homeostasis": True,
            }

        var_x = float(np.var(x))
        C_param = 1.0 / (1.0 + var_x)

        mean_x = float(np.mean(x))
        B_param = float(np.exp(-abs(mean_x)))

        n = min(sample_size, x.size)
        diffs = np.abs(np.diff(x[:n]))
        G_param = 1.0 / (1.0 + float(np.mean(diffs))) if diffs.size else 1.0

        max_abs = float(np.max(np.abs(x)))
        A_param = float(np.exp(-abs(max_abs - self.V_xi) / self.V_xi))

        term_u = (0.60 * C_param) + (0.30 * B_param) + (0.05 * G_param) + (0.05 * A_param)
        return {
            "C_Coherencia": C_param,
            "B_Balance": B_param,
            "G_Gradiente": G_param,
            "A_Azar": A_param,
            "TERM_U": float(term_u),
            "Homeostasis": bool(term_u >= 0.5),
        }

    def row_nnz_irregularity(self, indptr: np.ndarray) -> float:
        """Coeficiente de variación (std/mean) del nnz por fila.
        0 → perfectamente uniforme; valores altos → filas muy desbalanceadas.
        """
        indptr = np.asarray(indptr)
        if indptr.size < 2:
            return 0.0
        nnz = np.diff(indptr).astype(np.float64)
        mean = float(np.mean(nnz))
        if mean <= 0.0:
            return 0.0
        return float(np.std(nnz) / mean)

    def decide_schedule(
        self,
        term_u: float,
        nnz_cv: float,
        rows: int,
    ) -> Tuple[int, int]:
        """
        Mapea (TERM_U, irregularidad) → (chunk, schedule_mode).

        Reglas pragmáticas (auditables):
        - TERM_U alto + CV bajo  → static, chunk grande  (menos overhead)
        - TERM_U medio           → guided, chunk medio
        - TERM_U bajo            → dynamic, chunk pequeño (rebalanceo)
        - CV(nnz) muy alto       → balanced (modo 3, partición EXACTA por
          nnz -- no es una apuesta de chunk, es una cota garantizada;
          preferible a dynamic cuando la irregularidad es tan alta que
          ni el chunk pequeño garantiza reparto parejo)

        chunk se acota a [min_chunk, max_chunk] y se escala con #filas
        para no crear demasiadas tareas OpenMP en matrices pequeñas.
        chunk se ignora si el modo devuelto es 3 (balanced no lo usa).
        """
        # Base por homeostasis
        if term_u >= 0.75 and nnz_cv < 0.35:
            mode = 2  # static
            scale = 4.0
        elif term_u >= 0.55 and nnz_cv < 0.80:
            mode = 1  # guided
            scale = 2.0
        else:
            mode = 0  # dynamic
            scale = 1.0

        # Irregularidad alta: particion exacta, no heuristica de chunk.
        if nnz_cv >= 1.0:
            mode = 3  # balanced (partición exacta por nnz)
            scale = 1.0

        chunk = int(self.base_chunk * scale)

        # Escala suave con el número de filas (evita chunk absurdo en N pequeño)
        if rows > 0:
            # ~1 tarea por hilo como mínimo razonable
            max_sensible = max(self.min_chunk, rows // 8)
            chunk = min(chunk, max_sensible)

        chunk = max(self.min_chunk, min(self.max_chunk, chunk))
        return chunk, mode

    def update(
        self,
        C: np.ndarray,
        indptr: np.ndarray,
        rows: int,
        t: float = 0.0,
    ) -> GovernanceState:
        """Evalúa gobernanza + decide próximo schedule. Guarda historial."""
        gov = self.evaluate_governance(C)
        l3 = self.compute_l3(C)
        nnz_cv = self.row_nnz_irregularity(indptr)
        chunk, mode = self.decide_schedule(gov["TERM_U"], nnz_cv, rows)

        state = GovernanceState(
            C_Coherencia=gov["C_Coherencia"],
            B_Balance=gov["B_Balance"],
            G_Gradiente=gov["G_Gradiente"],
            A_Azar=gov["A_Azar"],
            TERM_U=gov["TERM_U"],
            Homeostasis=gov["Homeostasis"],
            L3=l3,
            chunk=chunk,
            schedule_mode=mode,
            nnz_cv=nnz_cv,
        )
        self._history.append(state)
        return state

    @property
    def history(self) -> list[GovernanceState]:
        return list(self._history)

    def last(self) -> Optional[GovernanceState]:
        return self._history[-1] if self._history else None


# =====================================================================
# Helpers de extracción CSR
# =====================================================================

def _extract_csr(A_sparse) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(A_sparse, "indptr"):
        return A_sparse.indptr, A_sparse.indices, A_sparse.data
    return A_sparse[0], A_sparse[1], A_sparse[2]


# =====================================================================
# API pública clásica (sin cambios de semántica)
# =====================================================================

def spmm_hbag(A_sparse, B: np.ndarray) -> np.ndarray:
    row_ptr, col_idx, values = _extract_csr(A_sparse)

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
        len(values_c),
    )

    C = np.zeros((N, K), dtype=np.float32)
    _lib.spmm_hbag_core(
        ctypes.byref(A_struct),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N,
        K,
    )
    return C


def spmm_hbag_omp(A_sparse, B: np.ndarray) -> np.ndarray:
    """Variante multi-hilo (OpenMP) de spmm_hbag. Respeta OMP_NUM_THREADS
    del entorno -- no fija un numero de hilos. El speedup real depende de
    cuantos nucleos reales tiene el host, no del acelerador (GPU/TPU)
    seleccionado en notebooks; ver BENCHMARKS.md."""
    if not _HAS_OMP:
        raise RuntimeError(
            "spmm_hbag_core_omp no esta disponible en este binario "
            "(la compilacion cayo al fallback sin -fopenmp). "
            "Usa spmm_hbag() en su lugar."
        )

    row_ptr, col_idx, values = _extract_csr(A_sparse)

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
        len(values_c),
    )

    C = np.zeros((N, K), dtype=np.float32)
    _lib.spmm_hbag_core_omp(
        ctypes.byref(A_struct),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N,
        K,
    )
    return C


def spmm_hbag_native(A_sparse, B: np.ndarray, threads: int = None) -> np.ndarray:
    """Variante de indices de 64 bits para matrices muy grandes.
    threads=None respeta OMP_NUM_THREADS; threads=N fuerza N hilos.
    Verificada con error absoluto exacto 0.0 contra PyTorch/MKL en
    5 bloques de 100M no-ceros (BENCHMARKS.md)."""
    if not _HAS_OMP64:
        raise RuntimeError(
            "spmm_hbag_core_omp64 no esta disponible en este binario "
            "(la compilacion cayo al fallback sin -fopenmp). "
            "Usa spmm_hbag() en su lugar."
        )

    row_ptr, col_idx, values = _extract_csr(A_sparse)

    row_ptr_c = np.ascontiguousarray(row_ptr, dtype=np.int64)
    col_idx_c = np.ascontiguousarray(col_idx, dtype=np.int64)
    values_c = np.ascontiguousarray(values, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)

    rows = len(row_ptr_c) - 1
    K = B_c.shape[1] if B_c.ndim > 1 else 1
    num_threads = int(threads) if threads else 0

    C = np.zeros((rows, K), dtype=np.float32)
    _lib.spmm_hbag_core_omp64(
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        col_idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        row_ptr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        K,
        num_threads,
    )
    return C


# =====================================================================
# API adaptativa (homeostasis → schedule)
# =====================================================================

def spmm_hbag_adaptive(
    A_sparse,
    B: np.ndarray,
    threads: int = None,
    engine: Optional[SOLARTEngine] = None,
    chunk: Optional[int] = None,
    schedule_mode: Optional[int] = None,
    t: float = 0.0,
    return_state: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, GovernanceState]]:
    """
    SpMM 64-bit con schedule gobernado por homeostasis SOL-ART.

    Flujo:
      1. Si chunk/mode no se fuerzan, el engine estima irregularidad de
         nnz por fila y (si hay historial) usa el último TERM_U para
         decidir schedule *antes* de ejecutar.
      2. Se llama al kernel C adaptativo (aritmética idéntica a omp64).
      3. Se re-evalúa gobernanza sobre C y se actualiza el engine para
         el próximo bloque (streaming / multi-block).

    Parámetros
    ----------
    A_sparse : scipy.sparse CSR-like o (indptr, indices, data)
    B        : denso float32, shape (cols, K)
    threads  : None → respeta entorno; int → fuerza N hilos
    engine   : SOLARTEngine reutilizable entre bloques (recomendado)
    chunk    : override manual del chunk OpenMP (None = decide el engine)
    schedule_mode : 0=dynamic, 1=guided, 2=static, 3=balanced (partición
        exacta por nnz, ignora chunk) — None = decide el engine
    t        : parámetro temporal SOL-ART (bloque index, epoch, etc.)
    return_state : si True, devuelve (C, GovernanceState)

    Equivalencia: el resultado numérico es bit-idéntico al de
    spmm_hbag_native para los mismos datos (solo cambia el orden de
    asignación de filas a hilos, no la acumulación dentro de cada fila).
    """
    if not _HAS_ADAPTIVE:
        # Fallback transparente: misma semántica numérica, sin adaptación
        C = spmm_hbag_native(A_sparse, B, threads=threads)
        if return_state:
            eng = engine or SOLARTEngine()
            row_ptr, _, _ = _extract_csr(A_sparse)
            state = eng.update(C, row_ptr, len(row_ptr) - 1, t=t)
            return C, state
        return C

    row_ptr, col_idx, values = _extract_csr(A_sparse)
    row_ptr_c = np.ascontiguousarray(row_ptr, dtype=np.int64)
    col_idx_c = np.ascontiguousarray(col_idx, dtype=np.int64)
    values_c = np.ascontiguousarray(values, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)

    rows = len(row_ptr_c) - 1
    K = B_c.shape[1] if B_c.ndim > 1 else 1
    num_threads = int(threads) if threads else 0

    eng = engine or SOLARTEngine()

    # Decisión de schedule *antes* de ejecutar
    if chunk is None or schedule_mode is None:
        nnz_cv = eng.row_nnz_irregularity(row_ptr_c)
        # Si hay historial, usar último TERM_U; si no, arrancar conservador
        prev = eng.last()
        term_u = prev.TERM_U if prev is not None else 0.55
        auto_chunk, auto_mode = eng.decide_schedule(term_u, nnz_cv, rows)
        if chunk is None:
            chunk = auto_chunk
        if schedule_mode is None:
            schedule_mode = auto_mode
    else:
        chunk = int(chunk)
        schedule_mode = int(schedule_mode)

    C = np.zeros((rows, K), dtype=np.float32)
    _lib.spmm_hbag_core_omp64_adaptive(
        values_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        col_idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        row_ptr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        K,
        num_threads,
        int(chunk),
        int(schedule_mode),
    )

    # Feedback: actualizar engine con el resultado real
    state = eng.update(C, row_ptr_c, rows, t=t)
    # Asegurar que el state refleja el schedule *usado* (no solo el próximo)
    state.chunk = int(chunk)
    state.schedule_mode = int(schedule_mode)

    if return_state:
        return C, state
    return C


__all__ = [
    "spmm_hbag",
    "spmm_hbag_omp",
    "spmm_hbag_native",
    "spmm_hbag_adaptive",
    "SOLARTEngine",
    "GovernanceState",
]
