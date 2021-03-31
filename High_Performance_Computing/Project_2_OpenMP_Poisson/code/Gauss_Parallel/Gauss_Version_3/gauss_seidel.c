/* gauss_seidel.c - Poisson problem in 3d
 *
 */
/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

double iteration_step(double*** u, double*** f, int N, double *squarednorm){
	static double loopnorm;
	double step_width=2./(double)N;
	double placeholder, temp;
	double denominator = 1/(double)6;
	int i,j,k;
	
	loopnorm = 0.0;
	
	# pragma omp for ordered(2) reduction(+:loopnorm)
	for (i=1; i<N-1; i++) {
		for (j=1; j<N-1; j++) {
		
			# pragma omp ordered depend(sink: i-1,j) depend(sink: i,j-1)
				for (k=1; k<N-1; k++) {
					placeholder=u[i][j][k];
					temp=u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + step_width*step_width*f[i][j][k];
					u[i][j][k]=temp*denominator;
					placeholder -= u[i][j][k];
					loopnorm+=placeholder*placeholder;
				}
			# pragma omp ordered depend(source)
		}
	}
	
	*squarednorm = sqrt(loopnorm);
}


void
gauss_seidel(double*** u, double*** f, int N, int iter_max, double tolerance) {
  	double squarednorm= tolerance+1;

	int i=0;
	# pragma omp parallel shared(squarednorm)
	{
		while(i<iter_max && squarednorm > tolerance){
			# pragma omp barrier
			# pragma omp single
			{
				squarednorm = 0.0;
				i++;
			}

			iteration_step(u,f,N, &squarednorm);
			}
		}
	printf("Tolerance: %.2f/%.2f. Iterations: %d/%d\n", (squarednorm), (tolerance), i, iter_max);
}

