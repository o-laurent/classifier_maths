#include "linear_classification.cuh"

using namespace std;

// Number of thread per block
#define THREADS_PER_BLOCK 1024
/* Constants for housing data set */
#define data_columns (9)
#define above_threshold (265000.0)

/////////////////////////////////////////////////////////
// Number of rows in arrays to print for debugging
/////////////////////////////////////////////////////////
#define print_rows (10)

/////////////////////////////////////////////////////////
// Main program
/////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    /* Parameters for the data set */
    size_t N_train = 12000; // 12000; // points for training (Google: 12000)
    size_t N_test = 5000;   // 5000; // points for validation (Google: 5000)
    size_t N = N_train;
    size_t Nall = N_train + N_test;

    /* Hyperarameters for Stochastic Gradient Descent */
    int nb_iter = 100;          // default: 10;
    int periods = nb_iter;      // reporting period
    int batch_size = N;         // defeault: N;
    float learning_rate = 1e-7; // default: 1e-7
    bool verbose = false;       // Show logs

    /* Reading the data set */
    fmatrix alldata = fmatrix_create_on_host(Nall, data_columns);
    read_csv("sample_data/california_housing_train.csv", alldata.data, Nall, data_columns);

    size_t D = data_columns - 1 + 1; // remove output column, add column with const. 1.0
    size_t M = 2;                    // number of labels (one-hot encoding)
    fmatrix Xall = fmatrix_create_on_host(D, Nall);
    fmatrix Yall = fmatrix_create_on_host(M, Nall);
    get_inputs_and_labels(alldata.data, &Xall.data, &Yall.data, Nall, data_columns, D, M);

    /////////////////////////////////////////////////////////
    // Inputs and labels are now available in X and Y.
    // Each input is a column in X; X is of dimension D x N
    // each label is a column in Y; Y is of dimension M x N
    /////////////////////////////////////////////////////////

    // Logfile
    FILE *fp = fopen("log.txt", "w");

    /* Memory Allocation and Initialization */
    fmatrix h_X = fmatrix_subcolumns(Xall, 0, N);
    fmatrix h_Y = fmatrix_subcolumns(Yall, 0, N);
    fmatrix h_Xtest = fmatrix_subcolumns(Xall, N, Nall);
    fmatrix h_Ytest = fmatrix_subcolumns(Yall, N, Nall);
    fmatrix h_W = fmatrix_create_on_host(D, M);
    fmatrix h_J = fmatrix_create_on_host(1, 1);

    xavier_weight_init(1.0, h_W);

    /* Copy data to device */
    fmatrix d_X = fmatrix_copy_to_device(h_X);
    fmatrix d_Y = fmatrix_copy_to_device(h_Y);
    fmatrix d_Xtest = fmatrix_copy_to_device(h_Xtest);
    fmatrix d_Ytest = fmatrix_copy_to_device(h_Ytest);
    fmatrix d_W = fmatrix_copy_to_device(h_W);
    fmatrix d_J = fmatrix_copy_to_device(h_J);

    /* Normalize */
    fmatrix d_Mu = compute_mean(d_X);
    fmatrix d_Std = compute_std(d_X, d_Mu);

    if (verbose)
    {
        printf("d_W default:\n");
        fmatrix_device_print(d_W);

        printf("d_X:\n");
        fmatrix_device_print(d_X);

        printf("d_Mu:\n");
        fmatrix_device_print(d_Mu);

        printf("d_Std:\n");
        fmatrix_device_print(d_Std);
    }

    parametered_normalize(d_X, d_Mu, d_Std);
    parametered_normalize(d_Xtest, d_Mu, d_Std);

    /* Create auxiliary matrices on device */
    fmatrix d_Z = fmatrix_create_on_device(M, batch_size);
    fmatrix d_P = fmatrix_create_on_device(M, batch_size);
    fmatrix d_G = fmatrix_create_on_device(D, M);
    fmatrix d_Ztest = fmatrix_create_on_device(M, d_Xtest.cols);

    /////////////////////////////////////////////////////////
    // Batch Gradient Descent
    /////////////////////////////////////////////////////////
    // fmatrix_device_print(d_X);
    // fmatrix_device_print(d_W);

    /* Create Handle */
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialisation failed\n");
    }

    /* Evaluate the starting accuracy */
    float accuracy = 0;
    accuracy = evaluate_accuracy(handle, d_W, d_Xtest, d_Ytest, d_Ztest, verbose);
    float J = evaluate_logloss(handle, d_P, d_Y, verbose);
    printf("Initial accuracy: %f - initial logloss: %f\n", accuracy, J);

    float alpha = 1.0f;
    float beta = 0.0f;

    clock_t t_start_total, t_end;
    t_start_total = clock();

    int batch_pointer = 0;
    for (int i = 0; i < nb_iter; ++i)
    {

        ////////////////////////////////
        // compute Z = W^T X
        // --> each column z of Z corresponds to one column x of X
        ////////////////////////////////
        cublasStatus_t multstat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_W.cols, d_X.cols, d_W.rows, &alpha, d_W.data, d_W.rows, d_X.data, d_X.rows, &beta, d_Z.data, d_Z.rows);
        if (verbose)
        {
            printf("d_W:");
            fmatrix_device_print(d_W);

            printf("d_X:");
            fmatrix_device_print(d_X);

            printf("dw rows %d, dw cols %d, dX rows %d, dX cols %d, dZ rows %d, dZ cols %d\n", d_W.rows, d_W.cols, d_X.rows, d_X.cols, d_Z.rows, d_Z.cols);
            printf("m %d, n %d, k %d\n", d_W.cols, d_X.cols, d_W.rows);

            printf("d_Z:");
            fmatrix_device_print(d_Z);
        }
        if (multstat != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS matrix multiplication failed 3 %d\n", multstat);
            gpuErrchk(cudaPeekAtLastError());
        }

        // compute softmax per column of Z and store in P
        d_P = softmax_col(d_Z);

        // Q := P-Y
        fmatrix d_Q = fmatrix_add(d_P, -1.0f, d_Y);

        if (verbose)
        {
            printf("d_P:");
            fmatrix_device_print(d_P);

            printf("d_Y:");
            fmatrix_device_print(d_Y);

            printf("d_Q:");
            fmatrix_device_print(d_Q);
        }

        // compute gradient G = XQ^T
        multstat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d_X.rows, d_Q.rows, d_X.cols, &alpha, d_X.data, d_X.rows, d_Q.data, d_Q.rows, &beta, d_G.data, d_G.rows);
        if (verbose)
        {
            printf("d_X:");
            fmatrix_device_print(d_X);

            printf("d_Q:");
            fmatrix_device_print(d_Q);

            printf("d_G:");
            fmatrix_device_print(d_G);

            printf("dw rows %d, dw cols %d, dX rows %d, dX cols %d, dG rows %d, dG cols %d\n", d_X.rows, d_X.cols, d_Q.rows, d_Q.cols, d_G.rows, d_G.cols);
            printf("m %d, n %d, k %d\n", d_X.rows, d_Q.rows, d_X.cols);
        }
        if (multstat != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS matrix multiplication failed 4\n");
            gpuErrchk(cudaPeekAtLastError());
        }

        // update weights W = W - learning_rate*G
        d_W = fmatrix_add(d_W, -learning_rate, d_G);
        if (verbose)
        {
            fmatrix_device_print(d_W);
        }

        // printf("W:\n");fmatrix_device_print(d_W);

        ////////////////////////////////
        // For reporting, compute logloss and accuracy
        ////////////////////////////////
        if (i % (nb_iter / periods) == 0)
        {
            float accuracy = evaluate_accuracy(handle, d_W, d_Xtest, d_Ytest, d_Ztest, verbose);
            J = evaluate_logloss(handle, d_P, d_Y, verbose);
            printf("iter: %d, logloss: %f, accuracy: %f\n", i, J, accuracy);
            fprintf(fp, "%f,%f\n", J, accuracy);
        }
    }
    t_end = clock();
    float duration = ((float)(t_end - t_start_total)) / CLOCKS_PER_SEC;
    printf("Duration (s): %f\n", duration);
    /* Evaluate the accuracy */
    accuracy = evaluate_accuracy(handle, d_W, d_Xtest, d_Ytest, d_Ztest, verbose);
    J = evaluate_logloss(handle, d_P, d_Y, verbose);
    printf("Final accuracy: %f - Final logloss: %f\n", accuracy, J);

    printf("Final weights: \n");
    fmatrix_device_print(d_W);

    /* Memory clean up */
    fmatrix_free_on_host(&h_W);
    fmatrix_free_on_host(&Xall);
    fmatrix_free_on_host(&Yall);

    fmatrix_free_on_device(&d_X);
    fmatrix_free_on_device(&d_Y);
    fmatrix_free_on_device(&d_Xtest);
    fmatrix_free_on_device(&d_Ytest);
    fmatrix_free_on_device(&d_W);
    fmatrix_free_on_device(&d_Z);
    fmatrix_free_on_device(&d_J);
    cublasDestroy(handle);

    // Close log file
    fclose(fp);
}