#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "spmm.h"

typedef struct { int n, k; double density; const char *label; } Config;

int main(void) {
#if defined(__AVX512F__)
    printf("SIMD detectado en compilacion: AVX-512\n");
#elif defined(__AVX2__)
    printf("SIMD detectado en compilacion: AVX2\n");
#elif defined(__AVX__)
    printf("SIMD detectado en compilacion: AVX\n");
#else
    printf("SIMD detectado en compilacion: ninguno (escalar)\n");
#endif
    fflush(stdout);

    Config configs[] = {
        {512,   64,  0.05, "pequena / alta densidad"},
        {2048,  128, 0.02, "mediana / dispersion tipica"},
        {4096,  128, 0.01, "grande / muy dispersa"},
        {2048,  512, 0.02, "K ancho (mas columnas densas)"}
    };
    int n_configs = sizeof(configs) / sizeof(Config);
    int trials = 5;

    printf("%-45s %10s %12s %12s %8s\n", "Config", "NNZ", "Speedup", "MaxErr", "Estado");
    printf("--------------------------------------------------------------------------------------\n");

    for (int cfg = 0; cfg < n_configs; cfg++) {
        int N = configs[cfg].n, K = configs[cfg].k;
        double density = configs[cfg].density;
        int expected_nnz = (int)(N * N * density * 2.0) + 1024;

        CSRMatrix A;
        A.row_ptr = malloc((N + 1) * sizeof(int));
        A.col_idx = malloc(expected_nnz * sizeof(int));
        A.values = malloc(expected_nnz * sizeof(float));
        float *B = malloc(N * K * sizeof(float));
        float *C_std = malloc(N * K * sizeof(float));
        float *C_hbag = malloc(N * K * sizeof(float));

        generate_sparse(&A, N, density, 42);
        for (int i = 0; i < N * K; i++) B[i] = (float)((i % 7) + 1) * 0.1f;

        double best_std = 1e9, best_hbag = 1e9;
        for (int t = 0; t < trials; t++) {
            struct timespec s, e;
            clock_gettime(CLOCK_MONOTONIC, &s);
            spmm_csr_std(&A, B, C_std, N, K);
            clock_gettime(CLOCK_MONOTONIC, &e);
            double dt = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;
            if (dt < best_std) best_std = dt;

            clock_gettime(CLOCK_MONOTONIC, &s);
            spmm_hbag_core(&A, B, C_hbag, N, K);
            clock_gettime(CLOCK_MONOTONIC, &e);
            dt = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;
            if (dt < best_hbag) best_hbag = dt;
        }

        double max_err = 0.0;
        for (int i = 0; i < N * K; i++) {
            double diff = fabs(C_std[i] - C_hbag[i]);
            if (diff > max_err) max_err = diff;
        }
        double speedup = best_std / best_hbag;
        char label[80];
        snprintf(label, sizeof(label), "N=%d K=%d d=%.1f%% (%s)", N, K, density * 100, configs[cfg].label);
        printf("%-45s %10d %10.2fx %12.2e %8s\n", label, A.nnz, speedup, max_err,
               max_err < 1e-3 ? "OK" : "FALLA");
        fflush(stdout);

        free(A.row_ptr); free(A.col_idx); free(A.values);
        free(B); free(C_std); free(C_hbag);
    }
    return 0;
}
