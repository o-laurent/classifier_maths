#include "stable_softmax.cuh"

#define THREADS_PER_BLOCK 1024

/* Compute the max of each col on the device */
__global__ void max_kernel(float *Z_d, int nb_LigneZ, int nb_ColZ, float *M, int nb_LigneM)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int j = case_id;

    /* If j is coherent */
    if (j < nb_ColZ)
    {
        for (int i = 0; i < nb_LigneZ; i++)
        {
            /* Replace the maximum if greater */
            if (Z_d[IDX2C(i, j, nb_LigneZ)] > M[IDX2C(j, 0, nb_LigneM)])
            {
                M[IDX2C(j, 0, nb_LigneM)] = Z_d[IDX2C(i, j, nb_LigneZ)];
            }
        }
    }
}

/* Start computing the softmax on the device */
__global__ void softmax_kernel(fmatrix d_Z, float *M, int nb_Lignem, float *S, int nb_LigneS, float *expZ_d)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id % d_Z.rows; // line id
    int j = case_id / d_Z.rows; // col id

    /* If j is coherent */
    if (j < d_Z.cols)
    {
        /* Compute the stable exponential */
        expZ_d[IDX2C(i, j, d_Z.rows)] = expf(d_Z.data[IDX2C(i, j, d_Z.rows)] - M[IDX2C(j, 0, nb_LigneS)]);

        /* Compute the sum */
        atomicAdd(&S[IDX2C(j, 0, nb_LigneS)], expZ_d[IDX2C(i, j, d_Z.rows)]);
    }
}

/* Compute the softmax on the device */
static __global__ void softmax_kernel_div(fmatrix d_Z, float *S, int nb_LigneS)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id % d_Z.rows; // line id
    int j = case_id / d_Z.rows; // col id

    /* If j is coherent */
    if (j < d_Z.cols)
    {
        /* Divide by the sum */
        d_Z.data[IDX2C(i, j, d_Z.rows)] = d_Z.data[IDX2C(i, j, d_Z.rows)] / S[IDX2C(j, 0, nb_LigneS)];
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
    fmatrix M = fmatrix_create_minus_inf_on_device(Z_d.cols, 1);
    gpuErrchk(cudaPeekAtLastError());

    /* Compute the maximum over each col */
    max_kernel<<<dimGrid, dimBlock>>>(Z_d.data, Z_d.rows, Z_d.cols, M.data, M.cols);
    gpuErrchk(cudaPeekAtLastError());
    // printf("max\n");
    // fmatrix_device_print(M);

    /* One thread per element */
    thread_nb = Z_d.cols * Z_d.rows;
    dimGrid = 1 + (thread_nb / THREADS_PER_BLOCK);
    dimBlock = THREADS_PER_BLOCK;

    /* Replace by exponentials and compute the sum */
    fmatrix expZ_d = fmatrix_create_null_on_device(Z_d.rows, Z_d.cols);
    fmatrix S = fmatrix_create_null_on_device(Z_d.cols, 1);
    gpuErrchk(cudaPeekAtLastError());

    softmax_kernel<<<dimGrid, dimBlock>>>(Z_d, M.data, M.cols, S.data, S.cols, expZ_d.data);
    gpuErrchk(cudaPeekAtLastError());
    // printf("expZ_d before div\n");
    // fmatrix_device_print(expZ_d);

    // printf("S before div\n");
    // fmatrix_device_print(S);

    /* Divide by the sum */
    softmax_kernel_div<<<dimGrid, dimBlock>>>(expZ_d, S.data, S.cols);
    gpuErrchk(cudaPeekAtLastError());

    // printf("expZ_d\n");
    // fmatrix_device_print(expZ_d);

    fmatrix_free_on_device(&M);
    fmatrix_free_on_device(&S);

    return expZ_d;
}