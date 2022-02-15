#ifndef evaluate_accuracy_H
#define evaluate_accuracy_H
#include "cublas_v2.h"
#include "fmatrix.cuh"
#include "classifier_math.cuh"
#include <assert.h>

/** 
 * Evaluate the accuracy of a linear classifier with D x M weight
 * matrix W, using D x N input data X and M x N output labels Y.
 * Z is a temporary matrix with dimensions M x N,
 * which must be previously allocated.
 */
float evaluate_accuracy(cublasHandle_t handle, fmatrix d_W, fmatrix d_X, fmatrix d_Y, fmatrix d_Z, bool verbose = true);

/** 
 * Compute the logloss given M x N matrices of 
 * probabilities P and output labels Y
 * and stores it in J.
 * J is a matrix with dimensions 1 x 1,
 * which must be previously allocated.
 * logloss = sum_k sum_j -Y(j,k)*log(P(j,k))
 * logloss = sum_k sum_j -Y^T(k,j)*log(P(j,k))
 */
float evaluate_logloss(cublasHandle_t handle, fmatrix d_P, fmatrix d_Y, bool verbose = true);

#endif