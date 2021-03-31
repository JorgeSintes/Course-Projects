/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

double iteration_step(double*** prev_u, double*** u, double*** f,int N){
	double squarednorm=0.0;
	double step_width=2./(double)N;
	double temp, temp2;
	double denominator = 1.0/(double)6;
	int i,j,k;
	for (i=1; i<N-1; i++){
		for (j=1; j<N-1; j++){
			for (k=1; k<N-1; k++){
//				prev_u[i][j][k]=u[i][j][k];
				temp=prev_u[i-1][j][k] + prev_u[i+1][j][k]+ prev_u[i][j-1][k] + prev_u[i][j+1][k] + prev_u[i][j][k-1] + prev_u[i][j][k+1] + step_width*step_width*f[i][j][k];
				u[i][j][k]=temp*denominator;   
			}
		}
	}
	for (i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){	
				temp2= u[i][j][k]-prev_u[i][j][k];
				squarednorm+=temp2*temp2;
				prev_u[i][j][k]=u[i][j][k];
			}
		}
	}
	
	return sqrt(squarednorm);
}


void
jacobi(double*** u, double***prev_u, double*** f, int N, int iter_max, double tolerance) {
   double realnorm=tolerance+1.0;
  //iteration: checking norm and Nr of iterations at the same time
	int i=0;
	while (realnorm > tolerance && i<iter_max){
		realnorm=iteration_step(prev_u,u,f,N);
		i++;
		}
	printf("We needed %d iterations to converge",i);
}
