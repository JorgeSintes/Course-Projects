extern "C" {
#include "cblas.h"
#include <omp.h>
#include <stdio.h>

void
matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
  double time = omp_get_wtime();
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.,A,k,B,n,0.,C,n);
  double elapsed = omp_get_wtime() - time;
  printf("CPU time: %f\n", elapsed);
}
}
