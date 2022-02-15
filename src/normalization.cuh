#ifndef normalization_H
#define normalization_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "fmatrix.cuh"
#include <assert.h>

/* Compute and return the row-wise mean of the data */
fmatrix compute_mean(fmatrix d_X);

/* Compute and return the row-wise std of the data */
fmatrix compute_std(fmatrix d_X, fmatrix d_Mu);

/* Normalize the matrix given in input with the precomputed mean and std */
void parametered_normalize(fmatrix d_X, fmatrix d_Mu, fmatrix d_Std);

/* Normalize the input matrix */
void normalize(fmatrix d_X);

#endif