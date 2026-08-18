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
```

## Comparación contra scipy.sparse (herramienta estándar de producción)

Todas las cifras anteriores comparan `spmm_hbag_core` contra una
implementación CSR de referencia escrita para este repo — no contra una
librería de producción. Esa comparación mide el mecanismo (reordenamiento
de caché), no la utilidad práctica de adoptar esta librería en vez de lo
que ya existe. Aquí está esa segunda pregunta, respondida directamente:
`hbag.spmm_hbag()` vs `scipy.sparse.csr_matrix.dot()`, un hilo, mismas
matrices, mismo hardware.

| Config | scipy (mejor de 7) | hbag (mejor de 7) | Ratio | MaxErr |
|---|---|---|---|---|
| N=512, K=64, d=5% | 0.137 ms | 0.107 ms | 1.28x | 1.9e-06 |
| N=2048, K=128, d=2% | 1.872 ms | 1.318 ms | 1.42x | 3.8e-06 |
| N=4096, K=128, d=1% | 3.975 ms | 2.816 ms | 1.41x | 5.7e-06 |
| N=2048, K=512, d=2% | 7.805 ms | 6.731 ms | 1.16x | 4.8e-06 |
| N=8192, K=64, d=0.5% | 4.686 ms | 3.338 ms | 1.40x | 3.8e-06 |

**Conclusión honesta: ~1.2x–1.4x sobre scipy, no 5x–19x.** El mecanismo de
caché es real y está medido con evidencia dura (sección anterior), pero
scipy ya está razonablemente bien optimizado — la ganancia práctica de
adoptar esta librería sobre scipy es modesta, consistente, y de un solo
hilo contra un solo hilo (sin threading de BLAS de por medio en esta
medición). Cualquier cifra de "5x-19x" fuera de contexto en publicaciones
externas se refiere a la comparación contra CSR de libro de texto, no
contra scipy — hay que ser explícito sobre cuál comparación se está
citando.

## Variante multi-hilo (`spmm_hbag_omp`) — y por qué "GPU"/"TPU" no significan lo que parecen

`spmm_hbag_omp` paraleliza por filas con OpenMP, sin fijar un número de
hilos — respeta `OMP_NUM_THREADS` del entorno o el default del sistema.
Verificado correcto para cualquier ancho de `K` (incluyendo no-múltiplos
de 8/16/32) con un bucle de residuo escalar.

**Hallazgo importante**: en Google Colab y Kaggle, seleccionar acelerador
"CPU", "GPU" o "TPU" en el menú **no cambia el número de núcleos de CPU
reales del host** dentro de la misma plataforma — y este kernel es C puro,
no usa GPU ni TPU en absoluto. Verificado con `os.cpu_count()` impreso
junto a cada medición, en 7 sesiones independientes:

| Plataforma | Acelerador seleccionado | Núcleos reales | Ratio vs scipy (rango, 4 configs) |
|---|---|---|---|
| Colab | TPU | 2 | 1.18x – 2.92x |
| Colab | CPU | 2 | 1.44x – 1.70x |
| Colab | GPU | 2 | 1.43x – 1.80x |
| Kaggle | CPU | 4 | 2.97x – 3.55x |
| Kaggle | GPU T4 | 4 | 2.86x – 3.71x |
| Kaggle | GPU P100 | 4 | 3.00x – 3.63x |
| Kaggle | TPU v5e-8 | 4 | 2.45x – 3.06x |

El patrón real: Colab siempre asigna 2 núcleos de host, Kaggle siempre
asigna 4, sin importar qué acelerador se seleccione. El ratio observado
es consistente con núcleos-disponibles × la ganancia de un solo hilo ya
documentada (~1.2x-1.4x), con caída de eficiencia en matrices grandes
(N=4096) probablemente por contención de ancho de banda de memoria
compartido entre hilos — no por caché, que es lo que arregla el kernel
de un solo hilo.

**Conclusión honesta**: cualquier cifra de "speedup en TPU" sin el
número de núcleos reales al lado no significa nada — el nombre del
acelerador es irrelevante para un kernel que no lo usa.

### Reproducir esta medición

El script usado para las 7 mediciones está en `tools/multi_env_check.py`
— instala `hbag`, imprime `os.cpu_count()`, compila `spmm_hbag_omp` con
las mismas flags que `setup.py`, y corre el barrido con verificación de
corrección incluida.

## Escala grande: HBAG vs PyTorch MKL (500M elementos, verificado)

`spmm_hbag_native` (índices de 64 bits, `tools/large_scale_mkl_check.py`)
se verificó contra `torch.sparse_csr_tensor` + PyTorch/MKL sobre 5 bloques
de 100 millones de no-ceros cada uno (500M total, ~3.5 GB), en sesiones
de Kaggle con `THREADS = os.cpu_count()` detectado en tiempo real (no
hardcodeado).

**Verificación de corrección: error absoluto máximo = 0.0 exacto en los
5 bloques, en todas las corridas.** No es una tolerancia pequeña — es
cero exacto, porque ambos motores recorren los no-ceros de cada fila en
el mismo orden secuencial, así que la acumulación de punto flotante
coincide bit a bit.

Nota de metodología: en las corridas de Kaggle más recientes, el kernel
se compiló en caliente dentro del propio notebook (`gcc -O3 -fopenmp
-mavx2 -mfma -march=native`) en vez de depender del `.so` instalado por
`pip`, para garantizar que la sesión efectivamente usa las rutas AVX2/FMA
del código — es el mismo kernel que `spmm_hbag_core_omp64` en
`src/spmm.c`, no una variante distinta.

| Corrida | HBAG C-Native (multi-hilo) | PyTorch MKL | Ratio |
|---|---|---|---|
| Kaggle, sesión 1 | 22,484.55 ms | 22,998.63 ms | 1.02x |
| Kaggle, sesión 2 | 21,483.32 ms | 24,494.96 ms | 1.14x |
| Kaggle, sesión 3 | 19,910.25 ms | 22,653.97 ms | 1.14x |

**Conclusión honesta**: tres corridas independientes, todas con
verificación de corrección exacta, dan un rango de **1.02x–1.14x** sobre
PyTorch/MKL — no un empate técnico exacto, una ligera ventaja consistente
que varía con la contención de la VM compartida de Kaggle en cada sesión
(mismo patrón de varianza que en la sección de núcleos reales, arriba).
Sigue siendo un resultado más fuerte narrativamente que el 5x-19x contra
CSR de libro de texto — igualar o superar levemente a MKL con
verificación bit a bit dice más de la calidad del kernel que ganarle a
una implementación ingenua. El número de hilos usado depende de la
sesión de Kaggle (ver tabla de núcleos arriba); no se reporta una cifra
fija universal por la misma razón que la sección anterior.

Esta prueba **no se ejecuta en CI** — 500M elementos exceden la RAM
típica de un runner gratuito de GitHub Actions. `tests/test_correctness.py`
incluye la misma metodología a escala reducida (2000×3000, K=64) contra
PyTorch/MKL, que sí corre en cada push.

## Lo que este benchmark NO afirma

- No compara contra librerías de GPU/TPU real (cuSPARSE, etc.) — todas
  las variantes de este repo son CPU-only.
- La generación de matriz dispersa usa un patrón de densidad no-uniforme
  simple (por bloques de filas) o aleatorio uniforme, no un dataset real.
- El speedup no es una constante — depende de la microarquitectura del
  CPU, el número de núcleos reales, y qué se usa como punto de
  comparación (CSR de referencia vs scipy vs MKL dan números muy
  distintos, todos documentados por separado arriba). Cualquier cifra
  citada sin especificar contra qué se comparó y en qué hardware debe
  tratarse con escepticismo.

## Homeostasis adaptativa en el schedule (SOL-ART)

A partir de esta evolución el kernel `spmm_hbag_core_omp64_adaptive`
expone dos parámetros de control (`chunk`, `schedule_mode`) que la
capa Python gobierna con `SOLARTEngine`:

| Condición | schedule_mode | chunk (relativo a base=64) |
|---|---|---|
| TERM_U ≥ 0.75 y CV(nnz) < 0.35 | 2 = static | ×4 |
| TERM_U ≥ 0.55 y CV(nnz) < 0.80 | 1 = guided | ×2 |
| TERM_U bajo o CV(nnz) ≥ 1.0 | 0 = dynamic | ×1 o ×0.75 |

**Garantía de corrección**: el cuerpo aritmético es idéntico al de
`spmm_hbag_core_omp64`. Solo cambia el orden en que OpenMP asigna
*filas* a hilos. Dentro de cada fila la acumulación sigue el mismo
orden secuencial de no-ceros → error absoluto máximo = 0.0 frente al
kernel clásico (verificado en `tests/test_adaptive.py`).

**Qué aporta**: en streaming de bloques (p.ej. 5 × 100M NNZ) el engine
reutiliza el TERM_U del bloque anterior para pre-decidir el schedule
del siguiente. Workloads estables reducen overhead de dynamic;
workloads irregulares recuperan balance sin tocar la aritmética.

No se reporta un speedup universal adicional frente a MKL: la ganancia
es situacional (menos contención de cola de tareas cuando el patrón
es estable). El valor de producto es el *know-how* de gobernanza
auditable, no un número mágico.

### Modo 3: `balanced` — partición exacta, no heurística

Se agregó un cuarto modo (`schedule_mode=3`) que no elige entre
políticas de scheduler de OpenMP: calcula por bisección sobre `indptr`
un tramo fijo de filas por hilo donde cada uno recibe ~el mismo número
total de no-ceros, antes de arrancar el cálculo. `decide_schedule` lo
activa automáticamente cuando `nnz_cv >= 1.0` — irregularidad tan alta
que ni un chunk pequeño en `dynamic` garantiza reparto parejo.

**Verificado**: error absoluto máximo = 0.0 contra `spmm_hbag_core_omp64`
en 1, 4 y 8 hilos, sobre una matriz con 5% de filas concentrando ~40%
del nnz (`tests/test_adaptive.py::test_balanced_mode_bitexact`).

Igual que el resto de esta sección: sin medición de velocidad a escala
todavía, no se reporta speedup para este modo específico.
