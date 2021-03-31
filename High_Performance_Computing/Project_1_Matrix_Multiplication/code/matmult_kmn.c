#include <stdio.h>

void
matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {
    
    int i1, i2,i3;
    for(i1 = 0; i1< m; i1++){
	for(i2 = 0; i2 < n; i2++){
	    C[i1][i2]=0;
	 }
    }
		

    
    for(i3=0; i3<k; i3++){
    	for(int i1=0; i1<m; i1++){
	    for(i2 = 0; i2 < n; i2++){
		C[i1][i2]+=A[i1][i3]*B[i3][i2];
	     }
        }
    }
	
}
