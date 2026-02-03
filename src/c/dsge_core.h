#ifndef DSGE_CORE_H
#define DSGE_CORE_H

#include <stdlib.h>
#include <complex.h>

#if defined(_WIN32)
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Define function pointer types for BLAS/LAPACK
typedef void (*dgemm_ptr)(char *transa, char *transb, int *m, int *n, int *k,
                         double *alpha, double *a, int *lda, double *b, int *ldb,
                         double *beta, double *c, int *ldc);

typedef void (*dgemv_ptr)(char *trans, int *m, int *n, double *alpha, double *a, int *lda,
                         double *x, int *incx, double *beta, double *y, int *incy);

typedef void (*dposv_ptr)(char *uplo, int *n, int *nrhs, double *a, int *lda,
                         double *b, int *ldb, int *info);

typedef void (*dcopy_ptr)(int *n, double *x, int *incx, double *y, int *incy);

// ZGGES for Gensys (Complex Generalized Schur)
// Note: setup for double complex
typedef void (*zgges_ptr)(char *jobvsl, char *jobvsr, char *sort, 
                          int (*selctg)(double complex *, double complex *),
                          int *n, double complex *a, int *lda, double complex *b, int *ldb,
                          int *sdim, double complex *alpha, double complex *beta,
                          double complex *vsl, int *ldvsl, double complex *vsr, int *ldvsr,
                          double complex *work, int *lwork, double *rwork, int *bwork, int *info);

// Struct to hold these pointers
typedef struct {
    dgemm_ptr dgemm;
    dgemv_ptr dgemv;
    dposv_ptr dposv;
    dcopy_ptr dcopy;
    zgges_ptr zgges;
} lapack_api_t;

// Global instance
extern lapack_api_t lapack;

// Setup function
EXPORT void setup_lapack(dgemm_ptr p_dgemm, dgemv_ptr p_dgemv, dposv_ptr p_dposv, 
                         dcopy_ptr p_dcopy, zgges_ptr p_zgges);

#endif
