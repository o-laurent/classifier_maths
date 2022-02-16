#include "classifier_math.cuh"
#include "cuda_stuff.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include "stable_softmax.cuh"

#define THREADS_PER_BLOCK 1024

/* Compute the elementwise logarithm on the device */
__global__ void log_kernel(fmatrix Z_d, float *d_logP)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id % Z_d.rows; // line id
    int j = case_id / Z_d.cols; // col id

    /* If j is coherent */
    if (j < Z_d.cols)
    {
        /* Compute the log */
        d_logP[IDX2C(i, j, Z_d.rows)] = logf(getfm(Z_d, i, j));
    }
}

/* Compute the sum of the diagonal of the product of A and B sum(diag(Yt*P)) */
__global__ void sum_diag_kernel(fmatrix d_A, float *J)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int j = case_id;
    int i = case_id;

    /* If j is coherent */
    if (j < d_A.cols && i < d_A.rows)
    {
        atomicAdd(J, getfm(d_A, i, j));
    }
}

/* generate random numbers in interval [min,max] */
float float_rand(float min, float max)
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return min + scale * (max - min);       /* [min, max] */
}

void xavier_weight_init(float a, fmatrix W)
{
    for (int j = 0; j < W.rows; ++j)
    {
        for (int i = 0; i < W.cols; ++i)
        {
            getfm(W, j, i) = a * (1.0 / sqrt(W.cols + W.rows)) * float_rand(-1.0, 1.0);
        }
    }
}

fmatrix softmax_col(fmatrix Z)
{
    fmatrix P = stable_softmax(Z);
    gpuErrchk(cudaPeekAtLastError());
    return P;
}

static __global__ void fmatrix_add_kernel(fmatrix P, float a, fmatrix Y, fmatrix Q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx / P.rows;
    int i = idx % P.rows;
    if (i < P.rows && j < P.cols)
    {
        getfm(Q, i, j) = getfm(P, i, j) + a * getfm(Y, i, j);
    }
}

/** 
 * Computes Q = P + a*Y 
 * Frees P
*/
fmatrix fmatrix_add(fmatrix P, float a, fmatrix Y)
{
    fmatrix_assert(P);
    fmatrix_assert(Y);
    assert(P.rows == Y.rows);
    assert(P.cols == Y.cols);

    /* One thread per element */
    int threadsPerBlock = fmatrix_elements(P);
    int blocksPerGrid = 1;
    if (threadsPerBlock > THREADS_PER_BLOCK)
    {
        blocksPerGrid = (threadsPerBlock - 1) / THREADS_PER_BLOCK + 1;
        threadsPerBlock = THREADS_PER_BLOCK;
    }

    fmatrix Q = fmatrix_create_null_on_device(P.rows, P.cols);

    fmatrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(P, a, Y, Q);
    gpuErrchk(cudaPeekAtLastError());

    fmatrix_free_on_device(&P);
    return Q;
}