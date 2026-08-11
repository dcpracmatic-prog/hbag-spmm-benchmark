#ifndef SPMM_H
#define SPMM_H

typedef struct {
    int *row_ptr;
    int *col_idx;
    float *values;
    int nnz;
} CSRMatrix;

/* Referencia: CSR estandar, orden de bucle c-externo / j-interno.
 * Implementacion de libro de texto, sin optimizaciones de acceso a memoria. */
void spmm_csr_std(const CSRMatrix *A, const float *B, float *C, int n, int k);

/* HBAG-Core: mismo calculo matematico, orden de bucle invertido
 * (j-externo / c-interno). La ganancia viene de que cada valor no-cero
 * reutiliza una fila contigua de K floats de B antes de saltar al
 * siguiente, en vez de saltar de columna en columna con paso K.
 * Medido con cachegrind: reduce fallos de cache L1 en ~9x en la
 * configuracion de referencia (ver BENCHMARKS.md). */
void spmm_hbag_core(const CSRMatrix *A, const float *B, float *C, int n, int k);

/* Genera una matriz dispersa con densidad no-uniforme por bloques de filas
 * (mas representativo de datos reales que densidad uniforme pura). */
void generate_sparse(CSRMatrix *A, int n, double density, unsigned int seed);

#endif
