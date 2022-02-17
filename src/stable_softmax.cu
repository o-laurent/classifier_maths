#include "stable_softmax.cuh"

#define THREADS_PER_BLOCK 1024

/* Compute the max of each col on the device */
static __global__ void max_kernel(fmatrix d_Z, fmatrix d_M)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int j = case_id;

    /* If j is coherent */
    if (j < d_Z.cols)
    {
        for (int i = 0; i < d_Z.rows; i++)
        {
            /* Replace the maximum if greater */
            if (getfm(d_Z, i, j) > getfm(d_M, j, 0))
            {
                getfm(d_M, j, 0) = getfm(d_Z, i, j);
            }
        }
    }
}

/* Start computing the softmax on the device */
static __global__ void softmax_kernel(fmatrix d_Z, fmatrix d_M, fmatrix d_S, fmatrix d_expZ)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id % d_Z.rows; // line id
    int j = case_id / d_Z.rows; // col id

    /* If j is coherent */
    if (j < d_Z.cols)
    {
        /* Compute the stable exponential */
        getfm(d_expZ, i, j) = expf(getfm(d_Z, i, j) - getfm(d_M, j, 0));

        /* Compute the sum */
        atomicAdd(&getfm(d_S, j, 0), getfm(d_expZ, i, j));
    }
}

/* Compute the softmax on the device */
static __global__ void softmax_kernel_div(fmatrix d_Z, fmatrix d_S)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id % d_Z.rows; // line id
    int j = case_id / d_Z.rows; // col id

    /* If j is coherent */
    if (j < d_Z.cols)
    {
        /* Divide by the sum */
        getfm(d_Z, i, j) /= getfm(d_S, j, 0);
    }
}

fmatrix stable_softmax(fmatrix Z_d)
{
    /* Check the input matrix */
    fmatrix_assert(Z_d);

    // printf("Z_d\n");
    // fmatrix_device_print(Z_d);

    /* One thread per column */
    int thread_nb = Z_d.cols;
    dim3 dimGrid(1 + (thread_nb / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);

    /* Create the matrix to hold the maximum values */
    fmatrix d_M = fmatrix_create_minus_inf_on_device(Z_d.cols, 1);
    gpuErrchk(cudaPeekAtLastError());

    /* Compute the maximum over each col */
    max_kernel<<<dimGrid, dimBlock>>>(Z_d, d_M);
    gpuErrchk(cudaPeekAtLastError());
    // fmatrix_device_to_csv("soft_dM.csv", d_M);
    // printf("max\n");
    // fmatrix_device_print(d_M);

    /* One thread per element */
    thread_nb = Z_d.cols * Z_d.rows;
    dimGrid = 1 + (thread_nb / THREADS_PER_BLOCK);
    dimBlock = THREADS_PER_BLOCK;

    /* Compute exponentials and compute the sum */
    fmatrix d_expZ = fmatrix_create_null_on_device(Z_d.rows, Z_d.cols);
    fmatrix d_S = fmatrix_create_null_on_device(Z_d.cols, 1);
    gpuErrchk(cudaPeekAtLastError());

    softmax_kernel<<<dimGrid, dimBlock>>>(Z_d, d_M, d_S, d_expZ);
    gpuErrchk(cudaPeekAtLastError());

    /* Divide by the sum */
    softmax_kernel_div<<<dimGrid, dimBlock>>>(d_expZ, d_S);
    gpuErrchk(cudaPeekAtLastError());
    // fmatrix_device_to_csv("soft_dexpZ_2.csv", d_expZ);

    /* Free the memory */
    fmatrix_free_on_device(&d_M);
    fmatrix_free_on_device(&d_S);

    return d_expZ;
}