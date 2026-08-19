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

        int c = 0;
        #if defined(__AVX2__)
        // Process in chunks of 32 floats (128 bytes, 4 YMM registers)
        for (; c <= k - 32; c += 32) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);
                __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
                __m256 b2 = _mm256_loadu_ps(b_ptr + 16);
                __m256 b3 = _mm256_loadu_ps(b_ptr + 24);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
                acc1 = _mm256_fmadd_ps(v_val, b1, acc1);
                acc2 = _mm256_fmadd_ps(v_val, b2, acc2);
                acc3 = _mm256_fmadd_ps(v_val, b3, acc3);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
            _mm256_storeu_ps(&c_row[c + 8], acc1);
            _mm256_storeu_ps(&c_row[c + 16], acc2);
            _mm256_storeu_ps(&c_row[c + 24], acc3);
        }

        // Process remaining 16 floats if any
        for (; c <= k - 16; c += 16) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);
                __m256 b1 = _mm256_loadu_ps(b_ptr + 8);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
                acc1 = _mm256_fmadd_ps(v_val, b1, acc1);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
            _mm256_storeu_ps(&c_row[c + 8], acc1);
        }

        // Process remaining 8 floats if any
        for (; c <= k - 8; c += 8) {
            __m256 acc0 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
        }
        #endif

        // Scalar residue for c < k
        if (c < k) {
            for (int cc = c; cc < k; cc++) {
                float sum = 0.0f;
                for (int j = row_start; j < row_end; j++) {
                    sum += A->values[j] * B[A->col_idx[j] * k + cc];
                }
                c_row[cc] = sum;
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

        int c = 0;
        #if defined(__AVX2__)
        for (; c <= k - 32; c += 32) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);
                __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
                __m256 b2 = _mm256_loadu_ps(b_ptr + 16);
                __m256 b3 = _mm256_loadu_ps(b_ptr + 24);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
                acc1 = _mm256_fmadd_ps(v_val, b1, acc1);
                acc2 = _mm256_fmadd_ps(v_val, b2, acc2);
                acc3 = _mm256_fmadd_ps(v_val, b3, acc3);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
            _mm256_storeu_ps(&c_row[c + 8], acc1);
            _mm256_storeu_ps(&c_row[c + 16], acc2);
            _mm256_storeu_ps(&c_row[c + 24], acc3);
        }

        for (; c <= k - 16; c += 16) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);
                __m256 b1 = _mm256_loadu_ps(b_ptr + 8);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
                acc1 = _mm256_fmadd_ps(v_val, b1, acc1);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
            _mm256_storeu_ps(&c_row[c + 8], acc1);
        }

        for (; c <= k - 8; c += 8) {
            __m256 acc0 = _mm256_setzero_ps();

            for (int j = row_start; j < row_end; j++) {
                float val = A->values[j];
                int col = A->col_idx[j];
                const float *b_ptr = &B[col * k + c];
                __m256 v_val = _mm256_set1_ps(val);

                __m256 b0 = _mm256_loadu_ps(b_ptr);

                acc0 = _mm256_fmadd_ps(v_val, b0, acc0);
            }

            _mm256_storeu_ps(&c_row[c], acc0);
        }
        #endif

        if (c < k) {
            for (int cc = c; cc < k; cc++) {
                float sum = 0.0f;
                for (int j = row_start; j < row_end; j++) {
                    sum += A->values[j] * B[A->col_idx[j] * k + cc];
                }
                c_row[cc] = sum;
            }
        }
    }
}
