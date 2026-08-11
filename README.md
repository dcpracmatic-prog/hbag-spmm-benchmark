# HBAG-Core SpMM: Multiplicación Dispersa×Densa con Reordenamiento Cache-Aware

Kernel en C de multiplicación matriz dispersa (CSR) × matriz densa, con una
variante que reordena el recorrido de bucles para maximizar el reuso de
línea de caché L1 frente a la implementación CSR estándar.

**No hay binario cerrado en este repo.** Todo el código —incluyendo el
kernel optimizado— es fuente abierta, compilable y auditable. La versión
anterior distribuía el kernel como librería estática (`.a`) sin código
fuente; se descontinuó porque no aportaba protección real (la licencia MIT
del repo cubría igualmente el binario) y sí restaba credibilidad frente a
revisión técnica.

## Qué hace, exactamente

Dos implementaciones del mismo cálculo matemático:

- `spmm_csr_std`: CSR de libro de texto, bucle externo por columna de la
  matriz densa B, bucle interno por los no-ceros de la fila.
- `spmm_hbag_core`: mismo cálculo, bucles invertidos (externo por no-cero,
  interno por columna). Cada no-cero reutiliza una fila contigua de B antes
  de saltar al siguiente, en vez de saltar de columna en columna con paso K.

Ambas producen el mismo resultado — la corrección se verifica en cada
corrida del benchmark, no se asume.

## Resultados

Ver [BENCHMARKS.md](BENCHMARKS.md) para metodología completa, tabla
multi-configuración, verificación de corrección y evidencia de mecanismo
(fallos de caché medidos con `cachegrind`, no inferidos).

Resumen: en la configuración de referencia (N=2048, K=128, densidad 2%),
la reducción de fallos de caché L1 medida es de **~9.4x** (8.4% → 0.9%
tasa de fallo), y el speedup de tiempo de pared observado varía entre
**~5x y ~19x según hardware** — el rango depende de la microarquitectura
del CPU (tamaño de L1/L2, latencia de fallo). No existe un número único
válido para todo hardware; por eso reportamos rango y metodología, no una
cifra de portada.

## Compilar y correr

Requiere GCC/Clang con soporte AVX2 (opcional — el kernel funciona sin
vectorización explícita; el compilador auto-vectoriza según `-march`).

```bash
make run
