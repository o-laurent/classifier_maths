#include "evaluate_accuracy.cuh"

#define THREADS_PER_BLOCK 1024

const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static __global__ void evaluate_accuracy_kernel(fmatrix d_Y, fmatrix d_Z, int *count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_Z.cols)
    {
        float z_max = getfm(d_Z, 0, idx);
        int i_max = 0;
        for (int i = 1; i < d_Z.rows; ++i)
        {
            if (getfm(d_Z, i, idx) > z_max)
            {
                z_max = getfm(d_Z, i, idx);
                i_max = i;
            }
        }
        if (getfm(d_Y, i_max, idx) >= 0.5f)
        {
            atomicAdd(count, 1);
        }
    }
}

float evaluate_accuracy(cublasHandle_t handle, fmatrix d_W, fmatrix d_X, fmatrix d_Y, fmatrix d_Z, bool verbose /* = true*/)
{
    assert(d_Y.cols == d_Z.cols);
    assert(d_Y.rows == d_Z.rows);
    fmatrix_assert(d_Z);

    float alpha = 1.0f;
    float beta = 0.0f;
    if (verbose)
    {
        printf("dw rows %d, dw cols %d, dX rows %d, dX cols %d, dZ rows %d, dZ cols %d\n", d_W.rows, d_W.cols, d_X.rows, d_X.cols, d_Z.rows, d_Z.cols);
        printf("m %d, n %d, k %d\n", d_W.cols, d_X.cols, d_W.rows);
    }
    /* Z = W^T x X */
    cublasStatus_t multstat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_W.cols, d_X.cols, d_W.rows, &alpha, d_W.data, d_W.rows, d_X.data, d_X.rows, &beta, d_Z.data, d_Z.rows);
    gpuErrchk(cudaPeekAtLastError());

    if (multstat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS matrix multiplication failed 1\n");
        printf("%s\n", _cudaGetErrorEnum(multstat));
    }

    int true_class = 0;

    int *d_count = 0;
    gpuErrchk(cudaMalloc((void **)&d_count, sizeof(int)));
    gpuErrchk(
        cudaMemcpy(d_count, &true_class, sizeof(int), cudaMemcpyHostToDevice));

    /* One thread per column */
    int threadsPerBlock = d_Z.cols;
    int blocksPerGrid = 1;
    if (threadsPerBlock > THREADS_PER_BLOCK)
    {
        blocksPerGrid = (threadsPerBlock - 1) / THREADS_PER_BLOCK + 1;
        threadsPerBlock = THREADS_PER_BLOCK;
    }

    evaluate_accuracy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Y, d_Z, d_count);
    device_synchronize();
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(
        cudaMemcpy(&true_class, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    int nb_tested = d_X.cols;
    if (verbose)
    {
        printf("Correct results: %d out of %d\n", true_class, nb_tested);
        printf("Accuracy: %f\n", (float)true_class / (float)nb_tested);
    }

    cudaFree(d_count);
    return (float)true_class / (float)d_Z.cols;
}

float evaluate_logloss(cublasHandle_t handle, fmatrix d_P, fmatrix d_Y, bool verbose /* =true */)
{
    assert(d_Y.cols == d_P.cols);
    assert(d_Y.rows == d_P.rows);

    /* One thread per element */
    int dimBlock = fmatrix_elements(d_P);
    int dimGrid = 1;
    if (dimBlock > THREADS_PER_BLOCK)
    {
        dimGrid = (dimBlock - 1) / THREADS_PER_BLOCK + 1;
        dimBlock = THREADS_PER_BLOCK;
    }

    /* Create the matrix which will contain the log of P */
    fmatrix d_logP = fmatrix_create_on_device(d_P.rows, d_P.cols);
    fmatrix_assert(d_logP);

    /* Compute the log */
    log_kernel<<<dimGrid, dimBlock>>>(d_P, d_logP);
    gpuErrchk(cudaPeekAtLastError());

    fmatrix d_Z = fmatrix_create_on_device(d_Y.cols, d_P.cols);

    float J = 0;
    float *d_J = NULL;
    cudaMalloc((void **)&d_J, sizeof(float));
    cudaMemcpy(d_J, &J, sizeof(float), cudaMemcpyHostToDevice);

    float alpha = -1.0f;
    float beta = 0.0f;

    if (verbose)
    {
        printf("dY rows %d, dY cols %d, dP rows %d, dP cols %d, dZ rows %d, dZ cols %d\n", d_Y.rows, d_Y.cols, d_P.rows, d_P.cols, d_Z.rows, d_Z.cols);
        printf("m %d, n %d, k %d\n", d_Y.cols, d_P.cols, d_Y.rows);
    }

    // dZ =dY^T*dP
    cublasStatus_t multstat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_Y.cols, d_logP.cols, d_Y.rows, &alpha, d_Y.data, d_Y.rows, d_logP.data, d_logP.rows, &beta, d_Z.data, d_Z.rows);
    if (multstat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS matrix multiplication failed 2\n");
        printf("%s\n", _cudaGetErrorEnum(multstat));
        gpuErrchk(cudaPeekAtLastError());
    }

    /* One thread per column */
    dimBlock = d_Z.cols;
    dimGrid = 1;
    if (dimBlock > THREADS_PER_BLOCK)
    {
        dimGrid = (dimBlock - 1) / THREADS_PER_BLOCK + 1;
        dimBlock = THREADS_PER_BLOCK;
    }

    sum_diag_kernel<<<dimGrid, dimBlock>>>(d_Z, d_J);
    gpuErrchk(cudaPeekAtLastError());

    if (verbose)
    {
        printf("In logloss: d_P\n");
        fmatrix_device_print(d_P);

        printf("In logloss: d_logP\n");
        fmatrix_device_print(d_logP);

        printf("In logloss: d_Y\n");
        fmatrix_device_print(d_Y);

        printf("In logloss: dZ = -dY^T*dP\n");
        fmatrix_device_print(d_Z);
    }

    cudaMemcpy(&J, d_J, sizeof(float), cudaMemcpyDeviceToHost);

    /* Free memory */
    cudaFree(d_J);
    fmatrix_free_on_device(&d_logP);
    fmatrix_free_on_device(&d_Z);

    return J;
}
