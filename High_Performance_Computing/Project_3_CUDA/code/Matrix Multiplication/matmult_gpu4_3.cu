extern "C"
{

#include <stdio.h>
#include <omp.h>

#define BLOCK_SIZE 16

__global__ 
void
matmultgpu4_rowwise(int m, int n, int k, double *A, double *B, double *C) {
    // This is the good one!!!
    	
 double Cvalue1 = 0.0, 
        Cvalue2 = 0.0,
        Cvalue3 = 0.0;

  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = 3*(blockIdx.y*blockDim.y+threadIdx.y);
  
  int e;

  if ((row < m-2) && (col < n)) {
    for(e=0;e<k;++e) {
        Cvalue1 += A[row*k+e]*B[e*n+col];
        Cvalue2 += A[(row+1)*k+e]*B[e*n+col];
        Cvalue3 += A[(row+2)*k+e]*B[e*n+col];
    }
        
    C[row*n+col]=Cvalue1;
    C[(row+1)*n+col]=Cvalue2;
    C[(row+2)*n+col]=Cvalue3;
  }

  else if ((row == m-2) && (col < n)) {
    for(e=0;e<k;++e) {
        Cvalue1 += A[row*k+e]*B[e*n+col];
        Cvalue2 += A[(row+1)*k+e]*B[e*n+col];
    }
        
    C[row*n+col]=Cvalue1;
    C[(row+1)*n+col]=Cvalue2;
  }

  else if ((row == m -1) && (col < n)) {
    for(e=0;e<k;++e)
        Cvalue1+=A[row*k+e]*B[e*n+col];
        
        C[row*n+col]=Cvalue1;
  }
	
}



void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C){

  double *d_A, *d_B, *d_C;

  int blocky;
  int sizeA = m * k *sizeof(double);
  int sizeB = k * n *sizeof(double);
  int sizeC = m * n *sizeof(double);

  double time1, time2, elapsed;

  // Declare the number of threads
  dim3 numOfThreadsPerBlock;
  numOfThreadsPerBlock.x = BLOCK_SIZE;
  numOfThreadsPerBlock.y = BLOCK_SIZE;

  // Initializing for colwise
  // blocky = (n+numOfThreadsPerBlock.x-1)/(numOfThreadsPerBlock.x);
  // dim3 numOfBlocks;
  //   numOfBlocks.x = (blocky+2)/3;
  // numOfBlocks.y = (m+numOfThreadsPerBlock.y-1)/(numOfThreadsPerBlock.y);

  // Initializing for rowwise
  blocky = (m+numOfThreadsPerBlock.y-1)/(numOfThreadsPerBlock.y);
  dim3 numOfBlocks;
  numOfBlocks.x = (n+numOfThreadsPerBlock.x-1)/(numOfThreadsPerBlock.x);
  numOfBlocks.y = (blocky+2)/3;
  
  // Allocate memory on the device
  cudaMalloc((void**)&d_A, sizeA);
  cudaMalloc((void**)&d_B, sizeB);
  cudaMalloc((void**)&d_C, sizeC);

  time1 = omp_get_wtime();

  // Copy the values over
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

  time2 = omp_get_wtime();

  matmultgpu4_rowwise<<<numOfBlocks, numOfThreadsPerBlock>>>(m, n, k, d_A, d_B, d_C);
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
