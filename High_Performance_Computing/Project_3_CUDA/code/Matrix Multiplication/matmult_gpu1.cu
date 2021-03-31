extern "C"{

#include <stdio.h>
#include <omp.h>

__global__ 
void
matmultgpu1(int m, int n, int k, double *A, double *B, double *C) {
    	
     int i1,i2,i3;

    
    for(i1 = 0; i1< m; i1++){
    	for(i2 = 0; i2 < n; i2++){    
             C[i1*n+i2]=0;        
	    for(i3 = 0; i3 < k; i3++){ 
		C[i1*n+i2]+=A[i1*k+i3]*B[i3*n+i2];
	     }
        }
    }
	
}



void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C){

  double   *d_A, *d_B, *d_C;

  int sizeA = m * k *sizeof(double);
  int sizeB = k * n *sizeof(double);
  int sizeC = m * n *sizeof(double);

  double time1, time2, elapsed;
 

  //Alloc memory on the device
  cudaMalloc((void**)&d_A,sizeA);
  cudaMalloc((void**)&d_B,sizeB);
  cudaMalloc((void**)&d_C,sizeC);

  time1 = omp_get_wtime();

  cudaMemcpy(d_A,A,sizeA,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B,sizeB,cudaMemcpyHostToDevice);
   
  time2 = omp_get_wtime();
  
  matmultgpu1<<<1,1>>>(m,n,k,d_A,d_B,d_C);
  cudaDeviceSynchronize();

  elapsed = omp_get_wtime() - time2;
  printf("Kernel time: %f\n", elapsed);

  cudaMemcpy(C,d_C,sizeC,cudaMemcpyDeviceToHost);

  elapsed = omp_get_wtime() - time1;
  printf("Kernel+copy time: %f\n", elapsed);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);



}
}

