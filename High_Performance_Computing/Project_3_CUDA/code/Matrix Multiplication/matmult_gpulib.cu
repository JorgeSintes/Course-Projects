
#include <cublas_v2.h>

extern "C"
{

#include <stdio.h>
#include <omp.h>


void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    const double alpha = 1.0, beta = 0.0;
    double *d_A, *d_B, *d_C;

    double time1, time2, elapsed;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }

    int sizeA = m * k *sizeof(double);
    int sizeB = k * n *sizeof(double);
    int sizeC = m * n *sizeof(double);
  
    // Allocate memory on the device
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    time1 = omp_get_wtime();
  
    // Copy the values over
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    time2 = omp_get_wtime();

    /*
    CUBLAS only reads matrices in column major form, and we hace A, B and C stored in row major.
    This will make CUBLAS understand our matrix as the transpose version if we input them normally.
    However, we want the output of C to be in rowmajor form too, so we need CUBLAS to calculate C^T.
    C^T = (AB)^T = B^T * A^T. Good news! We just need to swap the matrices out in the arguments to make
    CUBLAS output our C matrix in rowmajor form!
    */
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasDgemm(handle, transa, transb, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    cudaDeviceSynchronize();
    
    elapsed = omp_get_wtime() - time2;
    printf("Kernel time: %f\n", elapsed);    

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    elapsed = omp_get_wtime() - time1;
    printf("Kernel+copy time: %f\n", elapsed);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
   
}
}