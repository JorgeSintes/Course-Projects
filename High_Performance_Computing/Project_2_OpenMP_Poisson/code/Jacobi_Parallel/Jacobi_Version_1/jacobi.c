/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

void jacobi(double*** u, double***prev_u, double*** f, int N, int iter_max, double tolerance) {
  //iteration: checking norm and Nr of iterations at the same time
	double step_width=2./(double)N;
	double temp, ***swap, realnorm = 123456789;
	double denominator = 1.0/(double)6;
	double temp2;
	int i,j,k, t_id = 0;
	int iter = 0;

	tolerance = tolerance * tolerance;

	while (realnorm > tolerance && iter<iter_max) {
		
		swap = u;
		u = prev_u;
		prev_u = swap;
		realnorm = 0.0;
		iter++;
		
		# pragma omp parallel for shared(u, prev_u, f, N, iter_max, step_width, denominator, realnorm) private(i,j,k,temp, temp2)
		for (i=1; i<N-1; i++){
			for (j=1; j<N-1; j++){
				for (k=1; k<N-1; k++){
					
					temp=prev_u[i-1][j][k] + prev_u[i+1][j][k]+ prev_u[i][j-1][k] + prev_u[i][j+1][k] + prev_u[i][j][k-1]
					+ prev_u[i][j][k+1] + step_width*step_width*f[i][j][k];
					u[i][j][k]=temp*denominator;

					temp2= u[i][j][k]-prev_u[i][j][k];
					# pragma omp atomic
					realnorm += temp2*temp2;
				}
			}
		}
		// printf("2. %f, %d. Thread no.: %d\n", realnorm, iter, t_id);
	}
	printf("Tolerance: %.2f/%.2f. Iterations: %d/%d\n", sqrt(realnorm), sqrt(tolerance), iter, iter_max);
}
