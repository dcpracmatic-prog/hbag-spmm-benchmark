## `BENCHMARKS.md`

```markdown
# Metodología y resultados

## Metodología

- **Verificación de corrección en cada corrida**: el harness (`bench.c`)
  calcula el error absoluto máximo entre `spmm_csr_std` y `spmm_hbag_core`
  sobre la salida completa, y lo imprime junto al resultado de tiempo.
  Compilado sin `-ffast-math` (evita relajar IEEE 754 y esconder posibles
  errores reales detrás de reasociación de punto flotante). El error
  residual observado (1e-5 a 1e-4) corresponde al orden distinto de
  acumulación de punto flotante entre las dos versiones, no a un error de
  implementación.
- **5 corridas por configuración, se reporta el mejor tiempo** (mínimo, no
  promedio) para reducir el efecto de ruido del sistema operativo sobre
  cada función individualmente.
- **Semilla fija (`srand(42)`)**: la matriz generada es idéntica entre
  corridas y entre máquinas, para que las comparaciones de speedup entre
  hardware distinto sean válidas.
- **Generación de esparsidad no-uniforme por bloques de filas** (densidad
  variable en bloques de 32 filas) en vez de densidad uniforme pura —
  Nota: esto es una aproximación simple, no un dataset real. Sigue
  pendiente correr contra matrices reales (ej. SuiteSparse Matrix
  Collection) antes de publicar como representativo de carga real.

## Resultados de tiempo — multi-hardware

Configuración de referencia: N=2048, K=128, densidad 2% (136,179 no-ceros).

| Hardware | L3 cache | Speedup (mediana de 3 corridas) |
|---|---|---|
| Entorno sandbox local (contenedor, L3 agregada 260 MiB) | 260 MiB | ~5.0x |
| Google Colab, Xeon @2.20GHz | 55 MiB | ~9.6x – 13.5x |

Rango completo observado en Colab, las 4 configuraciones, 3 corridas
independientes:

| Config | Speedup mín. | Speedup máx. |
|---|---|---|
| N=512, K=64, densidad 5% | 12.58x | 19.30x |
| N=2048, K=128, densidad 2% | 9.62x | 13.47x |
| N=4096, K=128, densidad 1% | 8.56x | 11.81x |
| N=2048, K=512, densidad 2% | 11.49x | 14.19x |

La variación entre corridas en la misma máquina (hasta ~30% relativo) es
consistente con contención normal de CPU en una VM compartida — no se
reporta un número único porque no existe un número único válido.

## Evidencia de mecanismo (no inferida — medida con cachegrind)

Config de referencia, medida de forma aislada por función con
`tools/cache_probe.c` (evita mezclar los conteos de las dos versiones):

| | D1 misses (L1) | D1 miss rate | LLd misses (L3) |
|---|---|---|---|
| `spmm_csr_std` | 12,057,165 | 8.4% | 51,883 |
| `spmm_hbag_core` | 1,287,023 | 0.9% | 51,883 |

**Reducción de fallos L1: 9.37x** — número casi idéntico al speedup de
tiempo real medido en la misma configuración (9.6x–13.5x). Esto es la
prueba de que la ganancia de velocidad viene del mecanismo de caché
propuesto, no de otro factor no identificado.

Nota importante: los fallos de LL (L3) son **idénticos** entre las dos
versiones (51,883 en ambas). El mecanismo real está en la frontera L1/L2,
no en si el working set cabe en L3 — una hipótesis anterior (tamaño de L3)
fue descartada con este dato.

### Reproducir esta medición

```bash
make cache_probe
valgrind --tool=cachegrind --cachegrind-out-file=out.csr ./cache_probe csr
valgrind --tool=cachegrind --cachegrind-out-file=out.hbag ./cache_probe hbag
cg_annotate out.csr | grep -A2 "D1  misses"
cg_annotate out.hbag | grep -A2 "D1  misses"
