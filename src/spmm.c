

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "spmm.h"

void spmm_csr_std(const CSRMatrix *A, const float *B, float *C, int n, int k) {
    for (int i = 0; i < n; i++) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        for (int c = 0; c < k; c++) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; j++) {
                sum += A->values[j] * B[A->col_idx[j] * k + c];
            }
            C[i * k + c] = sum;
        }
    }
}

void spmm_hbag_core(const CSRMatrix *A, const float *B, float *C, int n, int k) {
    for (int i = 0; i < n; i++) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        float *c_row = &C[i * k];
        memset(c_row, 0, k * sizeof(float));
        for (int j = row_start; j < row_end; j++) {
            float val = A->values[j];
            int col = A->col_idx[j];
            const float *b_row = &B[col * k];
            for (int c = 0; c < k; c++) {
                c_row[c] += val * b_row[c];
            }
        }
    }
}

void generate_sparse(CSRMatrix *A, int n, double density, unsigned int seed) {
    srand(seed);
    int current_nnz = 0;
    A->row_ptr[0] = 0;
    int block = 32;
    for (int i = 0; i < n; i++) {
        double row_density = density * (0.3 + 1.4 * ((i / block) % 3 == 0 ? 1.6 : 0.6));
        for (int j = 0; j < n; j++) {
            if ((double)rand() / RAND_MAX < row_density) {
                A->col_idx[current_nnz] = j;
                A->values[current_nnz] = (float)(rand() % 10) + 1.0f;
                current_nnz++;
            }
        }
        A->row_ptr[i + 1] = current_nnz;
    }
    A->nnz = current_nnz;
}

void spmm_hbag_core_omp(const CSRMatrix *A, const float *B, float *C, int n, int k) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        float *c_row = &C[i * k];
        memset(c_row, 0, k * sizeof(float));

        for (int j = row_start; j < row_end; j++) {
            float val = A->values[j];
            int col = A->col_idx[j];
            const float *b_row = &B[col * k];
            for (int c = 0; c < k; c++) {
                c_row[c] += val * b_row[c];
            }
        }
    }
}

/* Nucleo compartido de una fila: identico orden de suma en los 3 modos
 * (siempre j = row_start..row_end sobre la misma fila), por eso el
 * resultado es bit-exact sin importar como se repartan las filas entre
 * hilos -- lo unico que cambia entre modos es el *reparto de filas*,
 * nunca el orden de acumulacion dentro de una fila. */
static inline void spmm_row64(
    long long i, const float *data, const long long *indices,
    const long long *indptr, const float *B, float *C, long long K
) {
    long long row_start = indptr[i];
    long long row_end = indptr[i + 1];
    float *c_row = &C[i * K];
    memset(c_row, 0, (size_t)K * sizeof(float));
    for (long long j = row_start; j < row_end; j++) {
        float val = data[j];
        long long col = indices[j];
        const float *b_row = &B[col * K];
        for (long long c = 0; c < K; c++) {
            c_row[c] += val * b_row[c];
        }
    }
}

void spmm_hbag_core_omp64(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K, int num_threads
) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 64)
    for (long long i = 0; i < rows; i++) {
        spmm_row64(i, data, indices, indptr, B, C, K);
    }
}

/* Particion EXACTA por nnz acumulado (modo 3, "balanced"): biseccion sobre
 * indptr para que cada hilo reciba un tramo de filas con ~el mismo total
 * de no-ceros. No es heuristica de scheduler -- es una cota deterministica
 * calculada antes de arrancar, tal como documenta spmm.h. */
static void balanced_row_ranges(
    const long long *indptr, long long rows, int num_threads, long long *starts
) {
    long long total_nnz = indptr[rows];
    starts[0] = 0;
    for (int t = 1; t < num_threads; t++) {
        long long target = (total_nnz * (long long)t) / (long long)num_threads;
        long long lo = starts[t - 1], hi = rows;
        while (lo < hi) {
            long long mid = lo + (hi - lo) / 2;
            if (indptr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        starts[t] = lo;
    }
    starts[num_threads] = rows;
}

void spmm_hbag_core_omp64_adaptive(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K,
    int num_threads, int chunk, int schedule_mode
) {
    int nth = (num_threads > 0) ? num_threads : omp_get_max_threads();
    int eff_chunk = (chunk > 0) ? chunk : 64;

    if (schedule_mode == 3) {
        /* balanced: reparto exacto por nnz, sin schedule() de OpenMP */
        long long *starts = (long long *)malloc((size_t)(nth + 1) * sizeof(long long));
        balanced_row_ranges(indptr, rows, nth, starts);

        #pragma omp parallel num_threads(nth)
        {
            int tid = omp_get_thread_num();
            long long r0 = starts[tid], r1 = starts[tid + 1];
            for (long long i = r0; i < r1; i++) {
                spmm_row64(i, data, indices, indptr, B, C, K);
            }
        }
        free(starts);
        return;
    }

    /* modos 0/1/2 (dynamic/guided/static): schedule(runtime) + omp_set_schedule
     * para que chunk y tipo sean parametros de entrada, no constantes de
     * compilacion -- eso es lo que permite elegirlos en tiempo de ejecucion
     * desde SOLARTEngine (Python) via TERM_U. */
    omp_sched_t kind;
    switch (schedule_mode) {
        case 1:  kind = omp_sched_guided; break;
        case 2:  kind = omp_sched_static; break;
        default: kind = omp_sched_dynamic; break;
    }
    omp_set_num_threads(nth);
    omp_set_schedule(kind, eff_chunk);

    #pragma omp parallel for schedule(runtime)
    for (long long i = 0; i < rows; i++) {
        spmm_row64(i, data, indices, indptr, B, C, K);
    }
}
