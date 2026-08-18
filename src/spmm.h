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

/* Variante multi-hilo de spmm_hbag_core: mismo mecanismo de localidad de
 * cache (reuso de linea L1 por no-cero), paralelizado por filas con OpenMP.
 * NO fija un numero de hilos fijo -- respeta OMP_NUM_THREADS del entorno,
 * o el default del sistema (usualmente = nucleos disponibles). Verificado
 * correcto para cualquier K (no solo multiplos de 8/16/32) mediante un
 * bucle de residuo escalar al final. Ver BENCHMARKS.md para el hallazgo de
 * que el speedup depende de nucleos reales del host, no del acelerador
 * (GPU/TPU) seleccionado en notebooks -- este kernel no usa GPU/TPU.
 * Requiere compilacion con -fopenmp. */
void spmm_hbag_core_omp(const CSRMatrix *A, const float *B, float *C, int n, int k);

/* Variante de indices de 64 bits para matrices muy grandes (cientos de
 * millones de no-ceros), donde el volumen total de indices o nnz podria
 * acercarse al limite de int32 (~2.1 mil millones). Recibe punteros
 * crudos en vez de CSRMatrix, y el numero de hilos como parametro
 * explicito (en vez de depender solo de OMP_NUM_THREADS) para que el
 * caller pueda pasar os.cpu_count() real detectado, sin hardcodear un
 * valor fijo. Verificada con error absoluto exacto 0.0 contra
 * torch.sparse_csr_tensor + MKL en 5 bloques de 100M no-ceros cada uno
 * (ver BENCHMARKS.md, seccion 'Escala grande: HBAG vs PyTorch MKL'). */
void spmm_hbag_core_omp64(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K, int num_threads
);

/* ------------------------------------------------------------------
 * Homeostasis adaptativa (SOL-ART schedule)
 * ------------------------------------------------------------------
 * Mismo kernel matematico que spmm_hbag_core_omp64, pero el chunk size
 * del schedule OpenMP se elige en tiempo de ejecucion a partir de un
 * feedback de homeostasis (TERM_U) y de la irregularidad de densidad
 * por filas. chunk <= 0 usa el default clasico (64).
 *
 * schedule_mode:
 *   0 = dynamic  (default, robusto ante filas desbalanceadas)
 *   1 = guided   (overhead menor cuando TERM_U es alto / estable)
 *   2 = static   (maxima localidad cuando el workload es uniforme)
 *   3 = balanced (particion EXACTA por nnz acumulado -- ignora `chunk`;
 *       cada hilo recibe un tramo fijo de filas con ~el mismo numero
 *       total de no-ceros, calculado por biseccion sobre indptr antes
 *       de arrancar. No es una heuristica de scheduler: es una cota
 *       exacta, deterministica, sin overhead de dispatch dinamico.
 *       Preferible sobre 0/1/2 cuando la irregularidad de nnz por fila
 *       es alta y se quiere el balance garantizado sin depender de que
 *       el chunk elegido sea del tamano correcto.)
 *
 * El caller (capa Python / SOLARTEngine) decide mode + chunk a partir
 * de TERM_U, L3 y estadisticas de nnz por fila. El C solo aplica la
 * decision sin reintroducir heuristica opaca.
 */
void spmm_hbag_core_omp64_adaptive(
    const float *data, const long long *indices, const long long *indptr,
    const float *B, float *C, long long rows, long long K,
    int num_threads, int chunk, int schedule_mode
);

#endif
