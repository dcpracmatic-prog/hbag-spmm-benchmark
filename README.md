# HBAG-SpMM-Adaptive v0.3.0

[![tests](https://github.com/dcpracmatic-prog/hbag-spmm-benchmark/actions/workflows/test.yml/badge.svg)](https://github.com/dcpracmatic-prog/hbag-spmm-benchmark/actions/workflows/test.yml)

SpMM (Sparse x Dense) CSR con schedule OpenMP **adaptativo** . con un
schedule que reacciona a la irregularidad de densidad por fila. Bit-exact paralelo al kernel clasico.

> Que trae v0.3.0:
> - `spmm_hbag_core_omp64_adaptive`: 4 modos (`0` dynamic, `2` static, `3` balanced, otros = guided).
> - Modo **balanced** (3): particion EXACTA por nnz acumulado, no es heuristica.
> - Bindings Python: `spmm_hbag` / `spmm_hbag_omp` / `spmm_hbag_native` (int64) + `spmm_hbag_adaptive`.
> - `bench.c`: 7 kernels, `--self-test`, `--csv`, flags `--rows/--cols/--k/--density/--trials/--threads/--seed`.
> - `make test` corre bench + pytest; CI en `.github/workflows/test.yml`.

**No hay binario cerrado dentro del ZIP.** La `.so` que usa la libreria
de Python se compila localmente al instalar; `setup.py` ejecuta `nm -D`
sobre `hbag/libhbag.so` y exige que exporte los 4 simbolos que
`hbag/__init__.py` requiere. Si tu toolchain no los soporta, la build
aborta con `RuntimeError`.

## Quick start

```bash
make all            # bench + cache_probe + libhbag.so
make test           # test-c (./bench --self-test) + pytest
pip install -e . --no-build-isolation
python3 -c 'import hbag; print(hbag.__version__)'   # 0.3.0
```

Uso como libreria:

```python
import hbag, numpy as np
from scipy.sparse import random

A = random(2048, 2048, density=0.02, format='csr', dtype=np.float32)
B = np.random.rand(2048, 128).astype(np.float32)

C    = hbag.spmm_hbag(A, B)                            # single-thread core
C_omp  = hbag.spmm_hbag_omp(A, B)                      # OpenMP 32-bit
C_64   = hbag.spmm_hbag_native(A, B)                   # indices int64
C_ad, state = hbag.spmm_hbag_adaptive(A, B, return_state=True)
print(state.schedule_mode, state.chunk, state.TERM_U, state.nnz_cv)
```

## Comandos del Makefile (todos verificados, exit 0)

| Comando | Que hace |
|---|---|
| `make help` | Lista los targets |
| `make all` | Compila `bench` + `cache_probe` + `hbag/libhbag.so` |
| `make bench` | Solo `./bench` |
| `make cache_probe` | Solo `./cache_probe` (cachegrind helper) |
| `make shared` | Alias de `make hbag/libhbag.so` |
| `make hbag/libhbag.so` | Solo la `.so` + verificacion `nm -D` |
| `make libhbag.so` | `.so` legacy en la raiz |
| `make run` | `./bench --csv /tmp/bench_results.csv` |
| `make run-all` | bench 1/2/4 hilos, 3 CSV en /tmp |
| `make test-c` | `./bench --self-test` |
| `make test-py` | `pytest tests/ -v` |
| `make test` | test-c + test-py |
| `make install` | `pip install -e . --no-build-isolation` |
| `make clean` | Borra binarios, `.so`, `__pycache__` |
| `make distclean` | Borra tambien CSV en /tmp |

## Uso del benchmark `./bench`

```
./bench                                          sweep default (4 configs), tabla texto
./bench --self-test                              exit 2 si algun bit-exact falla
./bench --csv [FILE]                             CSV (stdout si FILE se omite)
./bench --rows N --cols N --k K --density D      una sola config arbitraria
   [--seed S] [--trials T] [--threads T]
./bench --threads T                              fija omp_set_num_threads(T); 0 = entorno
./bench --list                                   imprime el sweep y sale
./bench -h | --help                              ayuda
```

Columnas del CSV (orden estricto):

```
config,label,N,K,density,nnz,threads,kernel,t_ms,max_err,status
```

- `t_ms` = **mejor** de N corridas (`--trials`), con 2 warmups no contados.
- `speedup` = `t(spmm_csr_std) / t(kernel)` (solo en tabla texto).
- `max_err` = error absoluto maximo frente a `spmm_csr_std`.
- `status` = `OK` si `max_err < 1.0` Y los 3 modos adaptativos son **bit-exact** contra `spmm_hbag_core_omp64`.

Los 3 modos de `spmm_hbag_core_omp64_adaptive` (dyn / static / balanced)
deben producir el mismo C bit-a-bit que `spmm_hbag_core_omp64`;
`--self-test` lo verifica y aborta (`exit 2`) si no se cumple.

Ejemplos:

```bash
./bench --csv results.csv
./bench --csv                                # CSV a stdout
./bench --rows 8192 --cols 8192 --k 256 --density 0.005 --trials 3 --csv big.csv
for t in 1 2 4; do OMP_NUM_THREADS=$t ./bench --csv bench_t${t}.csv --threads $t; done
./bench --self-test                          # smoke test bit-exact
./bench --list                                # muestra el sweep por defecto
./bench --help
```

## Suite de pruebas (pytest)

```bash
pip install -e . --no-build-isolation
pip install pytest
make test-py
```

Archivos:
- `tests/test_correctness.py` -- correctness numerico + smoke-tests C/Python.
- `tests/test_adaptive.py` -- comportamiento del schedule SOL-ART.

Cobertura:
- Cada `spmm_hbag_*` vs `scipy.sparse.csr.dot(B)` en multiples K no multiplos de 8.
- Reproducibilidad de las columnas deterministas del CSV (`N,K,density,nnz,threads,kernel,max_err,status`); `t_ms` se excluye porque es tiempo de pared (~3% jitter).
- Existencia y exportacion de los 4 simbolos en `hbag/libhbag.so` via `nm -D`.
- `--self-test` pasa end-to-end.
- `make test-c` y `make test-py` pasan.

## Estructura del repositorio

```
hbag-spmm-adaptive/
├── .github/workflows/test.yml
├── .gitignore
├── BENCHMARKS.md
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── setup.py
├── hbag/__init__.py
├── src/
│   ├── bench.c
│   ├── spmm.c
│   └── spmm.h
├── tests/
│   ├── test_adaptive.py
│   └── test_correctness.py
└── tools/
    ├── adaptive_homeostasis_demo.py
    ├── cache_probe.c
    ├── large_scale_mkl_check.py
    └── multi_env_check.py
```

## Verificar el mecanismo de cache (reproducible)

```bash
make cache_probe
valgrind --tool=cachegrind --cachegrind-out-file=out.csr  ./cache_probe csr
valgrind --tool=cachegrind --cachegrind-out-file=out.hbag ./cache_probe hbag
cg_annotate out.csr  | head -25
cg_annotate out.hbag | head -25
```

## Limitaciones conocidas

- AVX2 + OpenMP requeridos para variantes multi-hilo.
- Python: indices `int32` por defecto en `spmm_hbag` / `spmm_hbag_omp`.
- Para mas de 2.1 mil millones NNZ usa `spmm_hbag_native` (`int64`).
- "Speedup adaptativo" no se reporta como constante: depende del patron de carga.

## Licencia

Apache License 2.0 -- ver [LICENSE](LICENSE).
