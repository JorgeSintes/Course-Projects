#include "cblas.h"

void
matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.,&A[0][0],k,&B[0][0],n,0.,&C[0][0],n);
}

