#ifndef classifier_math_H
#define classifier_math_H

#include "fmatrix.cuh"

/* Compute the elementwise logarithm on the device */
__global__ void log_kernel(fmatrix Z_d, fmatrix d_logP);

/* Compute the sum of the diagonal of the product of A and B sum(diag(Yt*P)) */
__global__ void sum_diag_kernel(fmatrix d_A, float *J);

/* Returns a random float between min and max (including). */
float float_rand(float min, float max);

/* Initialize W with Xavier's method, scaled by a. */
void xavier_weight_init(float a, fmatrix W, unsigned seed = 42);

/* Compute the softmax for each column of Z and store in P */
fmatrix softmax_col(fmatrix Z);

/* Compute the sum of two matrices */
fmatrix fmatrix_add(fmatrix P, float a, fmatrix Y);

#endif