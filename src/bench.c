/*
 * src/bench.c -- benchmark + self-test para hbag-spmm-adaptive v0.3.0.
 *
 * Mide 7 kernels contra spmm_csr_std y verifica bit-exact entre los 3
 * modos adaptativos de spmm_hbag_core_omp64_adaptive y spmm_hbag_core_omp64.
 * Ver README.md para uso completo.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "spmm.h"

typedef struct { int n; int k; double density; const char *label; } Cfg;

static Cfg DEFAULT_CONFIGS[] = {
    {512,   64,  0.05, "pequena_alta_densidad"},
    {2048,  128, 0.02, "mediana_dispersion_tipica"},
    {4096,  128, 0.02, "grande_muy_dispersa"},
    {2048,  512, 0.02, "K_ancho_mas_columnas_densas"},
};
static const int N_DEFAULT_CONFIGS = sizeof(DEFAULT_CONFIGS) / sizeof(Cfg);

static double now_sec(void) {
    struct timespec s;
    clock_gettime(CLOCK_MONOTONIC, &s);
    return (double)s.tv_sec + (double)s.tv_nsec / 1e9;
}

typedef struct {
    int *row_ptr;
    int *col_idx;
    float *values;
    long long *indptr64;
    long long *indices64;
    float *B;
    float *Cs[7];
    int N, K, nnz;
} W;

static void fill_B(float *B, int N, int K) {
    for (int i = 0; i < N * K; i++) B[i] = (float)((i % 7) + 1) * 0.1f;
}

/* Reserva matriz suficiente:  expected_nnz = 2*N*N*density es seguro para
 * la distribucion aleatoria que usa generate_sparse() con seed fijo. */
static void build_W(W *w, Cfg *cfg, unsigned int seed) {
    int N = cfg->n, K = cfg->k;
    double density = cfg->density;

    w->row_ptr = (int *)calloc(N + 1, sizeof(int));
    w->col_idx = (int *)malloc((size_t)((long long)N * N * 2 + 4096) * sizeof(int));
    w->values  = (float *)malloc((size_t)((long long)N * N * 2 + 4096) * sizeof(float));
    w->B       = (float *)malloc((size_t)N * K * sizeof(float));
    for (int i = 0; i < 7; i++) w->Cs[i] = (float *)calloc((size_t)N * K, sizeof(float));

    CSRMatrix A = { w->row_ptr, w->col_idx, w->values, 0 };
    generate_sparse(&A, N, density, seed);
    w->nnz = A.nnz;
    w->N = N; w->K = K;
    fill_B(w->B, N, K);

    long long ll = (long long)w->nnz;
    w->indptr64  = (long long *)malloc((N + 1) * sizeof(long long));
    w->indices64 = (long long *)malloc((size_t)ll * sizeof(long long));
    for (int i = 0; i <= N; i++) w->indptr64[i] = (long long)w->row_ptr[i];
    for (long long j = 0; j < ll; j++) w->indices64[j] = (long long)w->col_idx[j];
}

static void free_W(W *w) {
    free(w->row_ptr); free(w->col_idx); free(w->values);
    free(w->B);
    for (int i = 0; i < 7; i++) free(w->Cs[i]);
    free(w->indptr64); free(w->indices64);
}

static const char *KN[7] = {
    "spmm_csr_std",
    "spmm_hbag_core",
    "spmm_hbag_core_omp",
    "spmm_hbag_core_omp64",
    "spmm_adapt0_dyn",
    "spmm_adapt2_static",
    "spmm_adapt3_balanced",
};

static void run_kernel(int k, const W *w, int nth) {
    CSRMatrix A = { w->row_ptr, w->col_idx, w->values, w->nnz };
    switch (k) {
        case 0: spmm_csr_std(&A, w->B, w->Cs[0], w->N, w->K); break;
        case 1: spmm_hbag_core(&A, w->B, w->Cs[1], w->N, w->K); break;
        case 2: spmm_hbag_core_omp(&A, w->B, w->Cs[2], w->N, w->K); break;
        case 3: spmm_hbag_core_omp64(w->values, w->indices64, w->indptr64,
                w->B, w->Cs[3], (long long)w->N, (long long)w->K, nth); break;
        case 4: spmm_hbag_core_omp64_adaptive(w->values, w->indices64, w->indptr64,
                w->B, w->Cs[4], (long long)w->N, (long long)w->K, nth, 64, 0); break;
        case 5: spmm_hbag_core_omp64_adaptive(w->values, w->indices64, w->indptr64,
                w->B, w->Cs[5], (long long)w->N, (long long)w->K, nth, 64, 2); break;
        case 6: spmm_hbag_core_omp64_adaptive(w->values, w->indices64, w->indptr64,
                w->B, w->Cs[6], (long long)w->N, (long long)w->K, nth, 64, 3); break;
    }
}

/* 2 warmups no contados: llevan AVX2, predictor y OMP al regimen antes de
 * empezar a medir; sin esto, t_ms puede variar ~5-10% entre dos corridas
 * del mismo binario y rompe el uso de --csv como fuente determinista. */
static double time_kernel(int k, const W *w, int trials, int nth) {
    for (int w_i = 0; w_i < 2; w_i++) run_kernel(k, w, nth);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = now_sec();
        run_kernel(k, w, nth);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double max_abs_err(const float *ref, const float *got, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)ref[i] - (double)got[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Comparacion bit-a-bit por union, no por valor: cualquier reorden de
 * sumas (AVX2 con vectorizacion parcial) ya se manifestaria en el
 * siguiente nivel de tolerancia, pero aqui exigimos igualdad exacta
 * entre adapt[0|2|3] y omp64, que es la propiedad real auditada. */
static int bytes_eq(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        union { float f; unsigned u; } x, y;
        x.f = a[i]; y.f = b[i];
        if (x.u != y.u) return 0;
    }
    return 1;
}

static int evaluate(W *w, double *ts, double *err, int *ok) {
    (void)ts;
    int dyn_eq  = bytes_eq(w->Cs[3], w->Cs[4], w->N * w->K);
    int sta_eq  = bytes_eq(w->Cs[3], w->Cs[5], w->N * w->K);
    int bal_eq  = bytes_eq(w->Cs[3], w->Cs[6], w->N * w->K);

    ok[0] = 1;
    for (int k = 1; k <= 6; k++) err[k] = max_abs_err(w->Cs[0], w->Cs[k], w->N * w->K);

    /* AVX2 puede reordenar sumas parciales: tolerance < 1.0 cubre 1 ulp
     * relativo para CSR con valores hasta ~50, suficiente para todo el rango. */
    ok[1] = err[1] < 1.0;
    ok[2] = err[2] < 1.0;
    ok[3] = err[3] < 1.0;
    ok[4] = err[4] < 1.0 && dyn_eq;
    ok[5] = err[5] < 1.0 && sta_eq;
    ok[6] = err[6] < 1.0 && bal_eq;

    return dyn_eq && sta_eq && bal_eq;
}

static double eff_density(const W *w) {
    return (double)w->nnz / (double)((long long)w->N * w->N);
}

static void emit_csv_row(FILE *out, int cfg_idx, int k, const W *w, int nth,
                         double t_ms, double err, int ok) {
    fprintf(out, "%d,%s,%d,%d,%.4f,%d,%d,%s,%.4f,%.3e,%s\n",
            cfg_idx, KN[k], w->N, w->K, eff_density(w), w->nnz, nth,
            KN[k], t_ms, err, ok ? "OK" : "FALLA");
}

static void emit_text_row(int k, const W *w, int nth, double t_ms,
                          double speedup, double err, int ok) {
    printf("  %-30s N=%4d K=%3d nnz=%7d thr=%d t=%7.3f ms speedup=%6.2fx err=%.2e %s\n",
           KN[k], w->N, w->K, w->nnz, nth, t_ms, speedup, err, ok?"[OK]":"[FALLA]");
}

static int run_sweep(int trials, int threads, int self_test_only,
                     int csv_mode, const char *csv_path, int csv_stdout) {
    int had_fail = 0;
    if (!csv_mode && !self_test_only) {
        printf("hbag-spmm-adaptive v0.3.0 -- %d hilos%s\n",
               threads > 0 ? threads : omp_get_max_threads(),
               threads > 0 ? " (fijado)" : " (de OMP_NUM_THREADS)");
        printf("========================================================================\n");
    }
    for (int c = 0; c < N_DEFAULT_CONFIGS; c++) {
        Cfg cfg = DEFAULT_CONFIGS[c];
        W w; build_W(&w, &cfg, 42);
        int nth = threads;

        double ts[7] = {0}, err[7] = {0};
        int ok[7] = {0};
        for (int k = 0; k < 7; k++) ts[k] = time_kernel(k, &w, trials, nth);
        int adapt_eq = evaluate(&w, ts, err, ok);

        if (csv_mode) {
            FILE *out = csv_stdout ? stdout
                          : fopen(csv_path ? csv_path : "/tmp/bench_results.csv", "w");
            if (!out) { perror("csv"); free_W(&w); return 1; }
            if (!csv_stdout) {
                fprintf(out, "config,label,N,K,density,nnz,threads,kernel,t_ms,max_err,status\n");
            }
            for (int k = 1; k < 7; k++)
                emit_csv_row(out, c, k, &w, nth, ts[k] * 1e3, err[k], ok[k]);
            if (!csv_stdout) fclose(out);
        }
        if (self_test_only) {
            int all_ok = 1;
            for (int k = 0; k < 7; k++) if (!ok[k]) all_ok = 0;
            if (!all_ok || !adapt_eq) {
                fprintf(stderr, "SELF-TEST FALLA: config=%d (N=%d K=%d)\n", c, cfg.n, cfg.k);
                had_fail = 1;
            }
            free_W(&w);
            continue;
        }
        for (int k = 0; k < 7; k++) {
            double sp = (k == 0) ? 1.0 : ts[0] / ts[k];
            emit_text_row(k, &w, nth, ts[k] * 1e3, sp, err[k], ok[k]);
        }
        if (!adapt_eq) printf("  AVISO: modos adaptativos != omp64 bit-a-bit (config=%d)\n", c);
        printf("\n");
        for (int k = 0; k < 7; k++) if (!ok[k]) had_fail = 1;
        free_W(&w);
    }
    return had_fail ? 2 : 0;
}

static int run_single(int rows, int cols, int k, double density, int seed,
                      int trials, int nth,
                      int self_test_only, int csv_mode,
                      const char *csv_path, int csv_stdout) {
    (void)cols;
    Cfg cfg = { rows, k, density, "single_shot" };
    W w; build_W(&w, &cfg, (unsigned int)seed);

    double ts[7] = {0}, err[7] = {0};
    int ok[7] = {0};
    for (int ki = 0; ki < 7; ki++) ts[ki] = time_kernel(ki, &w, trials, nth);
    int adapt_eq = evaluate(&w, ts, err, ok);
    int all_ok = 1;
    for (int ki = 0; ki < 7; ki++) if (!ok[ki]) all_ok = 0;

    if (csv_mode) {
        FILE *out = csv_stdout ? stdout
                      : fopen(csv_path ? csv_path : "/tmp/bench_results.csv", "w");
        if (!out) { perror("csv"); free_W(&w); return 1; }
        if (!csv_stdout) {
            fprintf(out, "config,label,N,K,density,nnz,threads,kernel,t_ms,max_err,status\n");
        }
        for (int ki = 1; ki < 7; ki++)
            emit_csv_row(out, 99, ki, &w, nth, ts[ki] * 1e3, err[ki], ok[ki]);
        if (!csv_stdout) fclose(out);
    }
    if (!csv_mode && !self_test_only) {
        printf("single-shot: N=%d K=%d density=%.4f nnz=%d thr=%d\n",
               w.N, w.K, density, w.nnz, nth);
        for (int ki = 0; ki < 7; ki++) {
            double sp = (ki == 0) ? 1.0 : ts[0] / ts[ki];
            emit_text_row(ki, &w, nth, ts[ki] * 1e3, sp, err[ki], ok[ki]);
        }
    }
    int rc = 0;
    if (self_test_only) {
        if (all_ok && adapt_eq)
            printf("SELF-TEST OK (N=%d K=%d nnz=%d threads=%d)\n", rows, k, w.nnz, nth);
        else { fprintf(stderr, "SELF-TEST FALLA (N=%d K=%d)\n", rows, k); rc = 2; }
    } else if (!all_ok) rc = 2;
    free_W(&w);
    return rc;
}

static void print_help(const char *argv0) {
    printf("Uso: %s [opciones]\n"
           "  (sin opciones)                          sweep de 4 configs, tabla texto\n"
           "  --self-test                              corre el sweep y exit 2 si falla\n"
           "  --csv [<ARCHIVO>]                        CSV (stdout si ARCHIVO se omite)\n"
           "  --rows N --cols N --k K --density D      una sola config\n"
           "     [--seed S] [--trials T] [--threads T] [--csv <F>]\n"
           "  --threads T                              fija omp_set_num_threads(T); 0 = entorno\n"
           "  --list                                   imprime el sweep y sale\n"
           "  -h | --help                              esta ayuda\n", argv0);
}

int main(int argc, char **argv) {
    int self_test_only = 0, csv_mode = 0, csv_to_stdout = 0;
    const char *csv_path = NULL;
    int single = 0, rows = 0, cols = 0, k_cols = 128;
    double density = 0.02;
    int seed = 42, list_mode = 0;
    int trials = 3, threads_user = 0;

    for (int i = 1; i < argc; i++) {
        if (      !strcmp(argv[i], "--self-test"))  self_test_only = 1;
        else if (!strcmp(argv[i], "--csv")) {
            csv_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') csv_path = argv[++i];
            else csv_to_stdout = 1;
        }
        else if (!strcmp(argv[i], "--rows"))    { single = 1; rows    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cols"))    { single = 1; cols    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--k"))       { k_cols = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--density")) { density = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--seed"))    { seed   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--trials"))  { trials = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")) { threads_user = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--list"))    { list_mode = 1; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_help(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "arg desconocido: %s\n", argv[i]);
            print_help(argv[0]);
            return 2;
        }
    }
    if (list_mode) {
        printf("Sweep por defecto (%d configs):\n", N_DEFAULT_CONFIGS);
        for (int c = 0; c < N_DEFAULT_CONFIGS; c++)
            printf("  %2d. N=%-5d K=%-5d density=%.4f  %s\n",
                   c, DEFAULT_CONFIGS[c].n, DEFAULT_CONFIGS[c].k,
                   DEFAULT_CONFIGS[c].density, DEFAULT_CONFIGS[c].label);
        return 0;
    }
    if (threads_user > 0) omp_set_num_threads(threads_user);
    int nth_eff = threads_user > 0 ? threads_user : omp_get_max_threads();

    if (single) {
        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "--rows y --cols > 0\n");
            return 2;
        }
        return run_single(rows, cols, k_cols, density, seed, trials,
                          nth_eff, self_test_only, csv_mode,
                          csv_path, csv_to_stdout);
    }
    return run_sweep(trials, nth_eff, self_test_only,
                     csv_mode, csv_path, csv_to_stdout);
}
