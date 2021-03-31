extern "C"
{

#include <stdio.h>
#include <omp.h>

#define BLOCK_SIZE 16

__global__ 
void
matmultgpu5(int m, int n, int k, double *A, double *B, double *C) {
    int blockRow = blockIdx.y,
        blockCol = blockIdx.x,
        row = threadIdx.y,
        col = threadIdx.x;
    int i, j, Asrow, Ascol, Bsrow, Bscol, Crow, Ccol;
    
    double Cvalue = 0.0;

    for (i = 0; i < (k / BLOCK_SIZE); i++) {
        
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        Asrow = blockRow * BLOCK_SIZE + row;
        Ascol = BLOCK_SIZE * i + col;
        
        As[row][col] = A[Asrow * k + Ascol];

        Bsrow = BLOCK_SIZE * i + row;
        Bscol = blockCol * BLOCK_SIZE + col;

        Bs[row][col] = B[Bsrow * n + Bscol];

        __syncthreads();

        for (j = 0; j < BLOCK_SIZE; j++) {
            Cvalue += As[row][j] * Bs[j][col];
        }
        __syncthreads();
    }

    Crow = blockRow * BLOCK_SIZE + row;
    Ccol = blockCol * BLOCK_SIZE + col;

    C[Crow *n + Ccol] = Cvalue;

}



void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C){

    double *d_A, *d_B, *d_C;
    
    int sizeA = m * k *sizeof(double);
    int sizeB = k * n *sizeof(double);
    int sizeC = m * n *sizeof(double);

    double time1, time2, elapsed;
      
    // Declare the number of threads
    dim3 numOfThreadsPerBlock;
    numOfThreadsPerBlock.x = BLOCK_SIZE;
    numOfThreadsPerBlock.y = BLOCK_SIZE;
  
    dim3 numOfBlocks;
    numOfBlocks.x = n/BLOCK_SIZE;
    numOfBlocks.y = m/BLOCK_SIZE;
  
    // Allocate memory on the device
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    time1 = omp_get_wtime();
  
    // Copy the values over
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);  

    time2 = omp_get_wtime();
  
    matmultgpu5<<<numOfBlocks, numOfThreadsPerBlock>>>(m, n, k, d_A, d_B, d_C);
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
  