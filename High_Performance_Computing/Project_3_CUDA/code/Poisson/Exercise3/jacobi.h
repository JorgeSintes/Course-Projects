/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

__global__ void jacobi_gpu2(double *** u, double*** prev_u, double*** f, int N, double step_width, double denominator, int deviceID);

#endif
