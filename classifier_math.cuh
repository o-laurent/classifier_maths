#ifndef classifier_math_H
#define classifier_math_H

#include "fmatrix.cuh"

/* Returns a random float between min and max (including). */
float float_rand(float min, float max);

/* Initialize W with Xavier's method, scaled by a. */
void xavier_weight_init(float a, fmatrix W);

/* Compute the softmax for each column of Z and store in P */
fmatrix softmax_col(fmatrix Z);

/* Compute the sum of two matrices */
fmatrix fmatrix_add(fmatrix P, float a, fmatrix Y);

#endif