#include <stdio.h>
#include "jacobi.h"
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include "alloc3d.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <omp.h>

#define BLOCK_SIZE 8
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
			
// printf("X: %d - %d. Y: %d - %d. Z: %d - %d\n", radxi, radxf, radyi, radyf, radzi, radzf);
		
	for (i = radxi; i <= radxf; i++) {
		for (j = radyi; j <= radyf; j++) {
			for (k = radzi; k <= radzf; k++) {
				f[i][j][k] = 200;
			}
		}
	}
}

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

int main(int argc, char *argv[]){
	
    int 	N = 5;
    int 	iter_max = 10;
    double	tolerance = 5; //possibly initialize to avoid segmentation error 
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u_h = NULL;
    double 	***f_h = NULL;
    double  ***prev_u_h=NULL;
    double  ***u_d=NULL;
	double  ***f_d=NULL;
	double  ***prev_u_d=NULL;


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    start_T   = atof(argv[3]);  // start T for all inner grid points
    if (argc == 5) {
	output_type = atoi(argv[4]);  // ouput type
    }
    //ON CPU
    //allocate memory
    if ( (u_h = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
	//allocate f	
    if ( (f_h = d_malloc_3d(N, N, N)) == NULL ) {
		perror("array f: allocation failed");
		exit(-1);
    }
    if ( (prev_u_h = d_malloc_3d(N, N, N)) == NULL ) {
		perror("array f: allocation failed");
		exit(-1);
    }

    //ON GPU 1
    checkCudaErrors(cudaSetDevice(0));
    if ( (u_d = d_malloc_3d_gpu(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
	
    if ( (f_d = d_malloc_3d_gpu(N,N,N)) == NULL ) {
		perror("array f: allocation failed");
		exit(-1);
    }
    if ( (prev_u_d = d_malloc_3d_gpu(N, N, N)) == NULL ) {
		perror("array f: allocation failed");
		exit(-1);
    }   

    printf("We are going to start the Jacobi-Iteration \n");
    assign_ufu_old(u_h,f_h,prev_u_h,N, start_T);
    
	double time2= omp_get_wtime();
    
    //copy assigned Matrices to GPU
    transfer_3d(u_d, u_h, N,N,N, cudaMemcpyHostToDevice);
    transfer_3d(prev_u_d, prev_u_h, N,N,N, cudaMemcpyHostToDevice);
    transfer_3d(f_d,f_h, N,N,N, cudaMemcpyHostToDevice);

    //start for-loop 
    double step_width=2./(double)N;
    double denominator = 1.0/(double)6;
    int i;
    double ***swap;
    //create 3-dimensional grid
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    //reduce size of grid for smaller matrix
    dim3 grid_size(((N-2)*0.5+block_size.x-1)/block_size.x, ((N-2)+block_size.y-1)/ block_size.y, ((N-2)+block_size.z-1)/block_size.z);
    
 	//enable peer access between devices
 	cudaSetDevice(0);
 	checkCudaErrors(cudaDeviceEnablePeerAccess(1,0));
 	cudaSetDevice(1);
 	checkCudaErrors(cudaDeviceEnablePeerAccess(0,0));
 	
    //start iteration
	double time1=omp_get_wtime();
    for (i=0; i< iter_max; i++){
		//printf("At iteration Nr %d \n", i);
		checkCudaErrors(cudaSetDevice(0));
		jacobi_gpu2<<<grid_size,block_size>>>(u_d,prev_u_d,f_d,N, step_width, denominator, 0);	
		cudaDeviceSynchronize();
		
		cudaSetDevice(1);
		jacobi_gpu2<<<grid_size,block_size>>>(u_d,prev_u_d,f_d,N, step_width, denominator, 1);	
		cudaDeviceSynchronize();
		
		swap = u_d;
		u_d = prev_u_d;
		prev_u_d = swap;
		
		
	}
	double elapsed1=omp_get_wtime() - time1;

    //copy matrices back to CPU
    cudaSetDevice(0);
    transfer_3d(u_h, u_d, N,N,N, cudaMemcpyDeviceToHost);
    transfer_3d(prev_u_h, prev_u_d, N,N,N, cudaMemcpyDeviceToHost);
    //print_matrix(u_h,N);
    //print_matrix(prev_u_h,N);
    
	double elapsed2=omp_get_wtime() - time2; 
	printf("Elapsed Time (both kernels) %lf %lf \n", elapsed1, elapsed2);

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 2:
		print_matrix(prev_u_h,N);
		break;
		
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u_h);
    free(prev_u_h);
    free(f_h);
    
    free_gpu(u_d);
    free_gpu(prev_u_d);
    free_gpu(f_d);

    return(0);

}
