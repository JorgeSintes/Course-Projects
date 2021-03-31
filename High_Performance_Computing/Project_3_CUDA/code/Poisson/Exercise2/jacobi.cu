/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

__device__ void print_matrix2(double*** A, int N){
	int i,j,k;
	for (i=0; i<N; i++){
		printf("\n %d -th Layer \n", i);
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){	
				printf("%lf \t", A[i][j][k]);
			}
		printf("\n");
		}
	}
}

__global__ void jacobi_gpu2(double*** u, double***prev_u, double*** f, int N, double step_width, double denominator) {

	double temp;
	
	int j_index=threadIdx.y + blockIdx.y*blockDim.y;
    int k_index= threadIdx.x + blockIdx.x*blockDim.x;
	int i_index=threadIdx.z + blockIdx.z*blockDim.z; 

	
	//printf("%d %d %d \n", j_index, k_index, i_index);
	
	if ((j_index<N-2) && (k_index<N-2) && (i_index<N-2)){
			temp=prev_u[i_index][j_index+1][k_index+1] + prev_u[i_index+2][j_index+1][k_index+1]+ 
				prev_u[i_index+1][j_index][k_index+1] + prev_u[i_index+1][j_index+2][k_index+1] + 
				prev_u[i_index+1][j_index+1][k_index]+ prev_u[i_index+1][j_index+1][k_index+2] + step_width*step_width*f[i_index+1][j_index+1][k_index+1];
			u[i_index+1][j_index+1][k_index+1]=temp*denominator;
		}
	
	//printf("On the GPU we now have matrix:\n");
	//print_matrix2(u,N);
}
