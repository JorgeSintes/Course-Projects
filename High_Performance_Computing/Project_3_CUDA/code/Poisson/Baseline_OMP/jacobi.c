/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

void jacobi(double*** u, double***prev_u, double*** f, int N, int iter_max) {
  //iteration: checking norm and Nr of iterations at the same time
	double step_width=2./(double)N;
	double temp, ***swap;
	double denominator = 1.0/(double)6;
	int i,j,k, t_id = 0;
	int iter = 0;

	# pragma omp parallel shared(u, prev_u, swap, f, N, iter_max, step_width, denominator, iter) private(i,j,k,temp, t_id)
	{
		while (iter<iter_max) {
			# pragma omp barrier
			# pragma omp single //nowait
			{
				swap = u;
				u = prev_u;
				prev_u = swap;
				// printf("%f, %d\n", realnorm, iter);
				iter++;
			}
			// t_id = omp_get_thread_num();
			// // printf("1. %f, %d. Thread no.: %d\n", realnorm, iter, t_id);
			// // // # pragma omp barrier
			
			# pragma omp for 
			for (i=1; i<N-1; i++){
				for (j=1; j<N-1; j++){
					for (k=1; k<N-1; k++){
						
						temp=prev_u[i-1][j][k] + prev_u[i+1][j][k]+ prev_u[i][j-1][k] + prev_u[i][j+1][k] + prev_u[i][j][k-1]
						+ prev_u[i][j][k+1] + step_width*step_width*f[i][j][k];
						u[i][j][k]=temp*denominator;

					}
				}
			}
			// printf("2. %f, %d. Thread no.: %d\n", realnorm, iter, t_id);
		}
	} // end of parallel region
	//printf("Tolerance: %.2f/%.2f. Iterations: %d/%d\n", sqrt(realnorm), sqrt(tolerance), iter, iter_max);
}
