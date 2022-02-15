#include "classifier_math.cuh"
#include "cuda_stuff.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include "stable_softmax.cuh"

#define THREADS_PER_BLOCK 1024

// generate random numbers in interval [min,max]

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
    // printf("P\n");
    // fmatrix_device_print(P);

    gpuErrchk(cudaPeekAtLastError());
    device_synchronize();

    return P;
}

__global__ void fmatrix_add_kernel(fmatrix P, float a, fmatrix Y, fmatrix Q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx / P.rows;
    int i = idx % P.rows;
    if (i < P.rows && j < P.cols)
    {
        getfm(Q, i, j) = getfm(P, i, j) + a * getfm(Y, i, j);
    }
}

/** Compute Q = P + a*Y */
fmatrix fmatrix_add(fmatrix P, float a, fmatrix Y)
{
    fmatrix_assert(P);
    fmatrix_assert(Y);
    assert(P.rows == Y.rows);
    assert(P.cols == Y.cols);
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
    return Q;
}