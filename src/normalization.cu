#include "normalization.cuh"

#define THREADS_PER_BLOCK 1024

/* Compute the mean on the device */
__global__ void mean_kernel(fmatrix d_X, float *d_Mu, int nb_LigneMu)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id; // The column id
    float tmp = 0;   // the float to compute the mean across the cols

    /* If the thread is interesting */
    if (i < d_X.rows)
    {
        /* Iterate over the cols */
        for (int j = 0; j < d_X.cols; j++)
        {
            tmp += getfm(d_X, i, j);
        }

        /* Compute the sum */
        d_Mu[IDX2C(i, 0, nb_LigneMu)] = tmp / d_X.cols;
    }
}

/* Compute the std on the device */
__global__ void std_kernel(fmatrix d_X, float *Mu, int nb_LigneMu, float *Std)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id; // row id
    float tmp = 0;   // the float to compute the std across the cols
    if (i < d_X.rows)
    {
        /* Iterate over the cols */
        for (int j = 0; j < d_X.cols; j++)
        {
            tmp += powf(getfm(d_X, i, j) - Mu[IDX2C(i, 0, nb_LigneMu)], 2);
        }
        /* Compute the standard deviation */
        Std[IDX2C(i, 0, nb_LigneMu)] = sqrtf(tmp / d_X.cols);
    }
}

/* Normalize on the device */
__global__ void normalization_kernel(fmatrix d_X, float *Mu, int nb_LigneMu, float *Std)
{
    int case_id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = case_id; // row id
    if (i < d_X.rows)
    {
        /* Iterate over the cols */
        for (int j = 0; j < d_X.cols; j++)
        {
            /* Update the value X(i,j) */
            if (Std[IDX2C(i, 0, nb_LigneMu)] < 1e-5)
            {
                getfm(d_X, i, j) = getfm(d_X, i, j) - Mu[IDX2C(i, 0, nb_LigneMu)];
            }
            else
            {
                getfm(d_X, i, j) = (getfm(d_X, i, j) - Mu[IDX2C(i, 0, nb_LigneMu)]) / Std[IDX2C(i, 0, nb_LigneMu)];
            }
        }
    }
}

/* Compute and return the row-wise mean of the data */
fmatrix compute_mean(fmatrix d_X)
{
    /* Check the input matrix */
    fmatrix_assert(d_X);

    /* One thread per row */
    int thread_nb = d_X.rows;
    dim3 dimGrid(1 + (thread_nb / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);

    /* Create a matrix on the device */
    fmatrix d_Mu = fmatrix_create_on_device(d_X.rows, 1);
    fmatrix_assert(d_Mu);

    // fmatrix_device_print(d_Mu);

    /* Compute the mean with the kernel */
    mean_kernel<<<dimGrid, dimBlock>>>(d_X, d_Mu.data, d_Mu.cols);
    return d_Mu;
}

/* Compute and return the row-wise std of the data */
fmatrix compute_std(fmatrix d_X, fmatrix d_Mu)
{
    /* Check the input matrices */
    fmatrix_assert(d_X);
    fmatrix_assert(d_Mu);

    /* One thread per row */
    int thread_nb = d_X.rows;
    dim3 dimGrid(1 + (thread_nb / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);

    /* Create a matrix filled with nulls on the device*/
    fmatrix d_Std = fmatrix_create_on_device(d_X.rows, 1);

    /* Compute the std with the kernel */
    std_kernel<<<dimGrid, dimBlock>>>(d_X, d_Mu.data, d_Mu.rows, d_Std.data);
    return d_Std;
}

/* Normalize the matrix given in input with the precomputed mean and std */
void parametered_normalize(fmatrix d_X, fmatrix d_Mu, fmatrix d_Std)
{
    // Check the input matrices
    fmatrix_assert(d_X);
    fmatrix_assert(d_Mu);
    fmatrix_assert(d_Std);

    int thread_nb = d_X.cols;
    dim3 dimGrid(1 + (thread_nb / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);
    normalization_kernel<<<dimGrid, dimBlock>>>(d_X, d_Mu.data, d_Mu.rows, d_Std.data);
}

/* Normalize the input matrix */
void normalize(fmatrix d_X)
{
    // Check the input matrix
    fmatrix_assert(d_X);

    // Compute the mean
    fmatrix d_Mu = compute_mean(d_X);
    gpuErrchk(cudaPeekAtLastError());

    // Compute the standard deviation
    fmatrix d_Std = compute_std(d_X, d_Mu);
    gpuErrchk(cudaPeekAtLastError());

    // Normalize the matrix
    parametered_normalize(d_X, d_Mu, d_Std);
    gpuErrchk(cudaPeekAtLastError());
}