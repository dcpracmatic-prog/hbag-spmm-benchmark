
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
