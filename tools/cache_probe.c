#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/spmm.h"

/* Corre una sola funcion, una sola vez, para que cachegrind (u otro
 * profiler) mida sus fallos de cache sin mezclarlos con la otra version.
 * Uso: ./cache_probe [csr|hbag] */
int main(int argc, char **argv) {
    if (argc < 2) { printf("uso: %s [csr|hbag]\n", argv[0]); return 1; }
    int N = 2048, K = 128;
    double density = 0.02;
    int expected_nnz = (int)(N * N * density * 2.0) + 1024;

    CSRMatrix A;
    A.row_ptr = malloc((N + 1) * sizeof(int));
    A.col_idx = malloc(expected_nnz * sizeof(int));
    A.values = malloc(expected_nnz * sizeof(float));
    float *B = malloc(N * K * sizeof(float));
    float *C = malloc(N * K * sizeof(float));

    generate_sparse(&A, N, density, 42);
    for (int i = 0; i < N * K; i++) B[i] = (float)((i % 7) + 1) * 0.1f;

    if (strcmp(argv[1], "csr") == 0) {
        spmm_csr_std(&A, B, C, N, K);
    } else {
        spmm_hbag_core(&A, B, C, N, K);
    }
    printf("listo. C[0]=%f nnz=%d\n", C[0], A.nnz);

    free(A.row_ptr); free(A.col_idx); free(A.values); free(B); free(C);
    return 0;
}
