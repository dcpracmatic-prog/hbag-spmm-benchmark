#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
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
    /* No se fija un numero de hilos: respeta OMP_NUM_THREADS del entorno,
     * o el default del runtime OpenMP (usualmente = nucleos disponibles). */
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        float *c_row = &C[i * k];
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        memset(c_row, 0, k * sizeof(float));

        for (int j = row_start; j < row_end; j++) {
            float val = A->values[j];
            int col = A->col_idx[j];
            const float *b_row = &B[col * k];
            __m256 v_val = _mm256_set1_ps(val);

            int c = 0;
            for (; c <= k - 16; c += 16) {
                __m256 c0 = _mm256_loadu_ps(&c_row[c]);
                __m256 c1 = _mm256_loadu_ps(&c_row[c + 8]);
                __m256 b0 = _mm256_loadu_ps(&b_row[c]);
                __m256 b1 = _mm256_loadu_ps(&b_row[c + 8]);
                _mm256_storeu_ps(&c_row[c], _mm256_fmadd_ps(v_val, b0, c0));
                _mm256_storeu_ps(&c_row[c + 8], _mm256_fmadd_ps(v_val, b1, c1));
            }
            /* Residuo escalar: cubre cualquier K, no solo multiplos de 16. */
            for (; c < k; c++) {
                c_row[c] += val * b_row[c];
            }
        }
    }
}

void spmm_hbag_core_omp64(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K, int num_threads
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    /* Si num_threads <= 0, se respeta el default del runtime OpenMP
     * (OMP_NUM_THREADS del entorno, o nucleos disponibles) -- nunca se
     * fuerza un valor arbitrario sin que el caller lo pida. */

    #pragma omp parallel for schedule(dynamic, 64)
    for (long long i = 0; i < rows; i++) {
        long long row_start = indptr[i];
        long long row_end = indptr[i + 1];
        float *c_row = C + i * K;

        for (long long k = 0; k < K; k++) {
            c_row[k] = 0.0f;
        }

        for (long long jj = row_start; jj < row_end; jj++) {
            float val = data[jj];
            long long col = indices[jj];
            const float *b_row = B + col * K;
            __m256 v_val = _mm256_set1_ps(val);

            long long k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 v_b = _mm256_loadu_ps(b_row + k);
                __m256 v_c = _mm256_loadu_ps(c_row + k);
                v_c = _mm256_fmadd_ps(v_val, v_b, v_c);
                _mm256_storeu_ps(c_row + k, v_c);
            }
            /* Residuo escalar: cubre cualquier K. */
            for (; k < K; k++) {
                c_row[k] += val * b_row[k];
            }
        }
    }
}

/* ------------------------------------------------------------------
 * Kernel adaptativo: mismo cuerpo aritmetico, schedule gobernado por
 * parametros externos (homeostasis / densidad).
 * ------------------------------------------------------------------
 * Se usan tres pragmas separados porque OpenMP exige que schedule(...)
 * sea una constante de compilacion en la mayoria de implementaciones
 * (no se puede parametrizar con una variable en un solo pragma).
 * El cuerpo del bucle se mantiene identico en las tres ramas para
 * garantizar equivalencia matematica bit-a-bit con omp64 clasico.
 */
void spmm_hbag_core_omp64_adaptive(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K,
    int num_threads, int chunk, int schedule_mode
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    /* Default de seguridad: chunk clasico si el caller pasa <= 0. */
    int csz = (chunk > 0) ? chunk : 64;
    if (csz < 1) csz = 1;
    if (csz > 4096) csz = 4096;  /* techo pragmatico anti-overhead */

    if (schedule_mode == 3) {
        /* balanced: particion EXACTA por nnz acumulado, no heuristica de
         * scheduler. Ignora `chunk` -- el tramo de cada hilo se calcula
         * una vez por biseccion sobre indptr (costo O(num_threads *
         * log(rows)), despreciable frente al SpMM). */
        int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
        if (nt > rows) nt = (int)rows;
        if (nt < 1) nt = 1;

        long long total_nnz = indptr[rows];
        long long *bounds = (long long *)malloc((nt + 1) * sizeof(long long));
        bounds[0] = 0;
        bounds[nt] = rows;
        for (int t = 1; t < nt; t++) {
            long long target = (total_nnz * t) / nt;
            long long lo = bounds[t - 1], hi = rows;
            while (lo < hi) {
                long long mid = lo + (hi - lo) / 2;
                if (indptr[mid] < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            bounds[t] = lo;
        }

        omp_set_num_threads(nt);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long row_lo = bounds[tid];
            long long row_hi = bounds[tid + 1];

            for (long long i = row_lo; i < row_hi; i++) {
                long long row_start = indptr[i];
                long long row_end = indptr[i + 1];
                float *c_row = C + i * K;

                for (long long k = 0; k < K; k++) {
                    c_row[k] = 0.0f;
                }

                for (long long jj = row_start; jj < row_end; jj++) {
                    float val = data[jj];
                    long long col = indices[jj];
                    const float *b_row = B + col * K;
                    __m256 v_val = _mm256_set1_ps(val);

                    long long k = 0;
                    for (; k <= K - 8; k += 8) {
                        __m256 v_b = _mm256_loadu_ps(b_row + k);
                        __m256 v_c = _mm256_loadu_ps(c_row + k);
                        v_c = _mm256_fmadd_ps(v_val, v_b, v_c);
                        _mm256_storeu_ps(c_row + k, v_c);
                    }
                    for (; k < K; k++) {
                        c_row[k] += val * b_row[k];
                    }
                }
            }
        }
        free(bounds);
    } else if (schedule_mode == 2) {
        /* static: maxima localidad, minimo overhead de dispatch.
         * Preferible cuando TERM_U es alto y el nnz por fila es estable. */
        #pragma omp parallel for schedule(static, csz)
        for (long long i = 0; i < rows; i++) {
            long long row_start = indptr[i];
            long long row_end = indptr[i + 1];
            float *c_row = C + i * K;

            for (long long k = 0; k < K; k++) {
                c_row[k] = 0.0f;
            }

            for (long long jj = row_start; jj < row_end; jj++) {
                float val = data[jj];
                long long col = indices[jj];
                const float *b_row = B + col * K;
                __m256 v_val = _mm256_set1_ps(val);

                long long k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 v_b = _mm256_loadu_ps(b_row + k);
                    __m256 v_c = _mm256_loadu_ps(c_row + k);
                    v_c = _mm256_fmadd_ps(v_val, v_b, v_c);
                    _mm256_storeu_ps(c_row + k, v_c);
                }
                for (; k < K; k++) {
                    c_row[k] += val * b_row[k];
                }
            }
        }
    } else if (schedule_mode == 1) {
        /* guided: chunk decreciente, buen compromiso estable/irregular. */
        #pragma omp parallel for schedule(guided, csz)
        for (long long i = 0; i < rows; i++) {
            long long row_start = indptr[i];
            long long row_end = indptr[i + 1];
            float *c_row = C + i * K;

            for (long long k = 0; k < K; k++) {
                c_row[k] = 0.0f;
            }

            for (long long jj = row_start; jj < row_end; jj++) {
                float val = data[jj];
                long long col = indices[jj];
                const float *b_row = B + col * K;
                __m256 v_val = _mm256_set1_ps(val);

                long long k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 v_b = _mm256_loadu_ps(b_row + k);
                    __m256 v_c = _mm256_loadu_ps(c_row + k);
                    v_c = _mm256_fmadd_ps(v_val, v_b, v_c);
                    _mm256_storeu_ps(c_row + k, v_c);
                }
                for (; k < K; k++) {
                    c_row[k] += val * b_row[k];
                }
            }
        }
    } else {
        /* dynamic (default): robusto ante filas con nnz muy desigual.
         * Preferible cuando TERM_U es bajo o la densidad es irregular. */
        #pragma omp parallel for schedule(dynamic, csz)
        for (long long i = 0; i < rows; i++) {
            long long row_start = indptr[i];
            long long row_end = indptr[i + 1];
            float *c_row = C + i * K;

            for (long long k = 0; k < K; k++) {
                c_row[k] = 0.0f;
            }

            for (long long jj = row_start; jj < row_end; jj++) {
                float val = data[jj];
                long long col = indices[jj];
                const float *b_row = B + col * K;
                __m256 v_val = _mm256_set1_ps(val);

                long long k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 v_b = _mm256_loadu_ps(b_row + k);
                    __m256 v_c = _mm256_loadu_ps(c_row + k);
                    v_c = _mm256_fmadd_ps(v_val, v_b, v_c);
                    _mm256_storeu_ps(c_row + k, v_c);
                }
                for (; k < K; k++) {
                    c_row[k] += val * b_row[k];
                }
            }
        }
    }
}
