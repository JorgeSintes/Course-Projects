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

__global__ void jacobi_gpu1(double*** u, double***prev_u, double*** f, int N, double step_width, double denominator) {
  //iteration: checking norm and Nr of iterations at the same time
	double temp;
	int i,j,k= 0;
		
	for (i=1; i<N-1; i++){
		for (j=1; j<N-1; j++){
			for (k=1; k<N-1; k++){
				temp=prev_u[i-1][j][k] + prev_u[i+1][j][k]+ prev_u[i][j-1][k] + prev_u[i][j+1][k] + prev_u[i][j][k-1]
						+ prev_u[i][j][k+1] + step_width*step_width*f[i][j][k];
				u[i][j][k]=temp*denominator;
				//printf("For %d %d %d \n", i,j,k,temp*denominator);
				//printf("We have in the matrix: %lf \n", u[i][j][k]);
			}
		}
	}
	//printf("On the GPU we now have matrix:\n");
	//print_matrix2(u,N);
}
