#include <stdio.h>
#include <stdlib.h>

double ** malloc_2d(int m, int n){
	// initializes a matrix of size mxn. It is needed in order to make the matrix passable as an argument to other functions. DINAMICALLY ALLOCATED.
	int i, j;
	double **A;

	A = malloc(m * sizeof(double *));
	A[0] = malloc(m * n * sizeof(double));
	for(i = 1; i < m; i++){
		A[i] = A[0] + i*n;
	}
	
	return A;
}

void getBlock(int row, int col, int bs, int m, int n, double **A, double **Z){
    // This functions copies the values from A we want for the block into a matrix Z of size bs x bs i.e. make a bs x bs block starting in position A[row][col].
    // Z needs to be initialized with: Z = malloc_2d(bs, bs); before being used.

    int i, j, limi, limj;

	for(i = 0; i < bs; i++){
        for(j = 0; j < bs; j++){
            Z[i][j] = 0;
        }
    }

	limi = bs;
	limj = bs;

	if(row >= m - m % bs){
		limi = m % bs;
	}

	if(col >= n- n % bs){
		limj = n % bs;
	}

    for(i = 0; i < limi; i++){
        for(j = 0; j < limj; j++){
            Z[i][j] = A[row + i][col + j];
        }
    }
}


void matmult_blk_inside(int row, int col, int bs, double **ZA, double **ZB, double **C){
    // Calculates the multiplication of the block matrices ZA*ZB and stores it in the corresponding block for matrix C.

	int i1, i2, i3;
    for(i1 = 0; i1 < bs; i1++){
    	for(i3 = 0; i3 < bs; i3++){
	    	for(i2 = 0; i2 < bs; i2++){
				C[row + i1][col + i2] += ZA[i1][i3] * ZB[i3][i2];
	    	}
        }
    }
}

void matmult_blk(int m, int n, int k, double **A, double **B, double **D, int bs){
    int i1, i2, i3;
    double **ZA, **ZB, **C;

    // Create C to work with it in this function
    C = malloc_2d(m + (bs - m % bs), n + (bs - n % bs));
    
    // Initialize C to be a 0 matrix.
    for(i1 = 0; i1 < m; i1++){
		for(i2 = 0; i2 < n; i2++){
	    	C[i1][i2] = 0;
	 	}
    }

    // First we calculate all entries that can be subdivided in blocks of bs x bs. 
	// We allocate in memory the two matrices were we will be storing the blocks.
    // printf("%d", bs);
	ZA = malloc_2d(bs, bs);
    ZB = malloc_2d(bs, bs);

    for(i1 = 0; i1 < m; i1 += bs){
    	for(i3 = 0; i3 < k; i3 += bs){
	    	for(i2 = 0; i2 < n; i2 += bs){
				getBlock(i1, i3, bs, m, k, A, ZA);
                getBlock(i3, i2, bs, k, n, B, ZB);
                matmult_blk_inside(i1, i2, bs, ZA, ZB, C);
	    	}
        }
    }

    // Copy the results over to D, the matrix we are passing as argument.
    for(i1 = 0; i1 < m; i1++){
		for(i2 = 0; i2 < n; i2++){
	    	D[i1][i2] = C[i1][i2];
	 	}
    }
}