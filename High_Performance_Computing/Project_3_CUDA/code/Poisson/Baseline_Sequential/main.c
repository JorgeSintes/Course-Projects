/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <omp.h>

#ifdef _JACOBI
#include "jacobi.h"
#endif

#define N_DEFAULT 100

void
assign_ufu_old(double*** u, double*** f, double*** u_old, int N, double start_T){
	int radxi = 0,
        radxf = (5 * N)/16, // (-3/8 + 1) * N/2
        radyi = 0,
        radyf = N/4, // (_1/2 + 1) * N/2
        radzi = N/6 + (N%6 > 0), // (-2/3 + 1) * N/2 truncating upwards if there's some remainder.
        radzf = N/2; // (0 + 1) * N/2
	int i,j,k;

	for (i=0; i<N; i++){
		for (j=0;j<N;j++){
			for (k=0; k<N; k++){
				u_old[i][j][k]=start_T;
				u[i][j][k]=start_T;
				f[i][j][k]=0;
			}
		}
	}
		
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			
			u_old[i][0][j] = 0.;
			u_old[i][N-1][j]=20.;
			u[i][0][j] = 0.;
			u[i][N-1][j]=20.;
				
			u_old[i][j][0]=20.;
			u_old[i][j][N-1]=20.;
			u[i][j][0]=20.;
			u[i][j][N-1]=20.;

			u_old[0][i][j]=20.;
			u_old[N-1][i][j]=20.;
			u[0][i][j]=20.;
			u[N-1][i][j]=20.;
		}
	}
			
		
	for (i = radxi; i <= radxf; i++) {
		for (j = radyi; j <= radyf; j++) {
			for (k = radzi; k <= radzf; k++) {
				f[i][j][k] = 200;
			}
		}
	}
}

void assign_uf(double*** u, double*** f, int N, double start_T){
	int radxi = 0,
        radxf = (5 * N)/16, // (-3/8 + 1) * N/2
        radyi = 0,
        radyf = N/4, // (_1/2 + 1) * N/2
        radzi = N/6 + (N%6 > 0), // (-2/3 + 1) * N/2 truncating upwards if there's some remainder.
        radzf = N/2; // (0 + 1) * N/2
	int i,j,k;
	
	for (i=0; i<N; i++){
		for (j=0;j<N;j++){
			for (k=0; k<N; k++){
				u[i][j][k]=start_T;
				f[i][j][k]=0;
			}
		}
	}
		
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			
			u[i][N-1][j]=20.;

			u[i][j][0]=20.;
			u[i][j][N-1]=20.;

			u[0][i][j]=20.;
			u[N-1][i][j]=20.;
		}
	}
			
		
	for (i = radxi; i <= radxf; i++) {
		for (j = radyi; j <= radyf; j++) {
			for (k = radzi; k <= radzf; k++) {
				f[i][j][k] = 200;
			}
		}
	}
}

//testing function: simply printing matrix
void print_matrix(double*** A, int N){
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

int
main(int argc, char *argv[]) {

    int 	N = 5;
    int 	iter_max = 10;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double  ***u_old=NULL;
    double  ***f = NULL;


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    start_T   = atof(argv[3]);  // start T for all inner grid points
    if (argc == 5) {
	output_type = atoi(argv[4]);  // ouput type
    }

    //allocate memory
    if ( (u = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
	//allocate f	
    if ( (f = d_malloc_3d(N, N, N)) == NULL ) {
		perror("array f: allocation failed");
		exit(-1);
    }
	
	#if _JACOBI
	printf("We are going to start the Jacobi-Iteration \n");
	//only allocate prev_u here:
	if ( (u_old = d_malloc_3d(N, N, N)) == NULL ) {
		perror("array u: allocation failed");
		exit(-1);
    }


	assign_ufu_old(u,f,u_old,N, start_T);
    double time_1 = omp_get_wtime();    
	jacobi(u,u_old,f,N, iter_max);
	#endif
	
	double elapsed1 = omp_get_wtime() - time_1;
	
	printf("Elapsed Time: %lf\n", elapsed1 );
	
	
    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 2:
		print_matrix(u,N);
		break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);
    free(u_old);
    free(f);

    return(0);
}
