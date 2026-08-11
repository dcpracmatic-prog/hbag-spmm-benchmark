Aquí tienes el contenido completo del archivo README.md listo para copiar y pegar:

# HBAG-Core SpMM: Multiplicación Dispersa×Densa con Reordenamiento Cache-Aware Kernel en C de multiplicación matriz dispersa (CSR) × matriz densa, con una variante que reordena el recorrido de bucles para maximizar el reuso de línea de caché L1 frente a la implementación CSR estándar. **No hay binario cerrado en este repo.** Todo el código —incluyendo el kernel optimizado— es fuente abierta, compilable y auditable. La versión anterior distribuía el kernel como librería estática (`.a`) sin código fuente; se descontinuó porque no aportaba protección real (la licencia MIT del repo cubría igualmente el binario) y sí restaba credibilidad frente a revisión técnica. ## Qué hace, exactamente Dos implementaciones del mismo cálculo matemático: - `spmm_csr_std`: CSR de libro de texto, bucle externo por columna de la matriz densa B, bucle interno por los no-ceros de la fila. - `spmm_hbag_core`: mismo cálculo, bucles invertidos (externo por no-cero, interno por columna). Cada no-cero reutiliza una fila contigua de B antes de saltar al siguiente, en vez de saltar de columna en columna con paso K. Ambas producen el mismo resultado — la corrección se verifica en cada corrida del benchmark, no se asume. ## Resultados Ver [BENCHMARKS.md](BENCHMARKS.md) para metodología completa, tabla multi-configuración, verificación de corrección y evidencia de mecanismo (fallos de caché medidos con `cachegrind`, no inferidos). Resumen: en la configuración de referencia (N=2048, K=128, densidad 2%), la reducción de fallos de caché L1 medida es de **~9.4x** (8.4% → 0.9% tasa de fallo), y el speedup de tiempo de pared observado varía entre **~5x y ~19x según hardware** — el rango depende de la microarquitectura del CPU (tamaño de L1/L2, latencia de fallo). No existe un número único válido para todo hardware; por eso reportamos rango y metodología, no una cifra de portada. ## Compilar y correr Requiere GCC/Clang con soporte AVX2 (opcional — el kernel funciona sin vectorización explícita; el compilador auto-vectoriza según `-march`). ```bash make run 

O manualmente:

gcc -O3 -march=native src/bench.c src/spmm.c -o bench -lm ./bench 

Verificar el mecanismo de caché (reproducible)

make cache_probe valgrind --tool=cachegrind --cachegrind-out-file=out.csr ./cache_probe csr valgrind --tool=cachegrind --cachegrind-out-file=out.hbag ./cache_probe hbag cg_annotate out.csr | head -25 cg_annotate out.hbag | head -25 

Busca las líneas D1 misses y D1 miss rate en cada salida — ahí está la evidencia del mecanismo, no en el tiempo de reloj.

Estructura

├── src/spmm.h # Interfaz de las dos implementaciones ├── src/spmm.c # Kernel: CSR estandar y HBAG-Core ├── src/bench.c # Harness de benchmark: barrido multi-config + verificacion ├── tools/cache_probe.c # Ejecuta una funcion aislada, para profiling de cache ├── Makefile ├── LICENSE # MIT, cubre todo el repo sin excepcion └── BENCHMARKS.md # Metodologia, tabla de resultados, evidencia de mecanismo 

Limitaciones conocidas

El kernel HBAG-Core no divide en bloques (tiling) para K muy grande; para K que exceda el tamaño de L1 disponible, el reuso de línea se degrada. No probado más allá de K=512.

Un solo hilo. No hay paralelización con OpenMP ni comparación contra librerías multi-hilo (MKL, OpenBLAS) — queda pendiente como trabajo futuro, no como reclamo actual.

La generación de matriz dispersa usa un patrón de densidad no-uniforme simple (por bloques de filas), no un dataset real. Ver BENCHMARKS.md para el detalle de qué tan representativo es esto.

