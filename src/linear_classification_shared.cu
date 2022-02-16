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

/* Main function */
extern "C"
{
    float linear_classification(unsigned nb_iter, unsigned batch_size_in, float learning_rate, float rate_decay)
    {
        /* Parameters for the data set */
        unsigned N_train = 12000; // 12000; // points for training (Google: 12000)
        unsigned N_test = 5000;   // 5000; // points for validation (Google: 5000)
        unsigned N = N_train;
        unsigned Nall = N_train + N_test;

        /* Clamp batch size between 1 and N */
        unsigned batch_size = min(N, max(1u, batch_size_in));

        /* Hyperarameters for Stochastic Gradient Descent */
        unsigned periods = nb_iter; // reporting period
        bool verbose = false;       // Show logs

        /* Reading the data set */
        fmatrix alldata = fmatrix_create_on_host(Nall, data_columns);
        read_csv("sample_data/california_housing_train.csv", alldata.data, Nall, data_columns);

        unsigned D = data_columns - 1 + 1; // remove output column, add column with const. 1.0
        unsigned M = 2;                    // number of labels (one-hot encoding)
        fmatrix Xall = fmatrix_create_on_host((int)D, (int)Nall);
        fmatrix Yall = fmatrix_create_on_host((int)M, (int)Nall);
        get_inputs_and_labels(alldata.data, &Xall.data, &Yall.data, Nall, data_columns, D, M);

        // Logfile
        FILE *fp = fopen("log.txt", "w");

        /* Memory Allocation and Initialization */
        fmatrix h_X = fmatrix_subcolumns(Xall, 0, (int)N);
        fmatrix h_Y = fmatrix_subcolumns(Yall, 0, (int)N);
        fmatrix h_Xtest = fmatrix_subcolumns(Xall, (int)N, (int)Nall);
        fmatrix h_Ytest = fmatrix_subcolumns(Yall, (int)N, (int)Nall);
        fmatrix h_W = fmatrix_create_on_host((int)D, (int)M);
        fmatrix h_J = fmatrix_create_on_host(1, 1);

        xavier_weight_init(1.0, h_W);

        /* Copy data to device */
        fmatrix d_X_tot = fmatrix_copy_to_device(h_X);
        fmatrix d_Y_tot = fmatrix_copy_to_device(h_Y);
        fmatrix d_Xtest = fmatrix_copy_to_device(h_Xtest);
        fmatrix d_Ytest = fmatrix_copy_to_device(h_Ytest);
        fmatrix d_W = fmatrix_copy_to_device(h_W);
        fmatrix d_J = fmatrix_copy_to_device(h_J);

        /* Normalize */
        fmatrix d_Mu = compute_mean(d_X_tot);
        fmatrix d_Std = compute_std(d_X_tot, d_Mu);

        parametered_normalize(d_X_tot, d_Mu, d_Std);
        parametered_normalize(d_Xtest, d_Mu, d_Std);
        gpuErrchk(cudaPeekAtLastError());

        /* Create auxiliary matrices on device */
        fmatrix d_Z = fmatrix_create_on_device((int)M, (int)batch_size);
        fmatrix d_P = fmatrix_create_on_device((int)M, (int)batch_size);
        fmatrix d_G = fmatrix_create_on_device((int)D, (int)M);
        fmatrix d_Ztest = fmatrix_create_on_device((int)M, d_Xtest.cols);

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

        float alpha = 1.0f;
        float beta = 0.0f;

        clock_t t_start_total, t_end;
        t_start_total = clock();

        for (unsigned i = 0; i < nb_iter; ++i)
        {
            /* Reset the values for a new epoch */
            unsigned batch_pointer = 0;
            float J = 0;
            unsigned nb_col = batch_size;
            d_P.cols = (int)batch_size;
            d_Z.cols = (int)batch_size;

            /* Loop on mini-batches */
            while (batch_pointer < N)
            {
                /* If the batch is smaller */
                if (batch_pointer + batch_size > N)
                {
                    nb_col = N - batch_pointer;
                    d_P.cols = (int)nb_col;
                    d_Z.cols = (int)nb_col;
                }
                fmatrix d_X = fmatrix_subcolumns(d_X_tot, batch_pointer, batch_pointer + nb_col);
                fmatrix d_Y = fmatrix_subcolumns(d_Y_tot, batch_pointer, batch_pointer + nb_col);

                /* compute Z = W^T X */
                cublasStatus_t multstat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_W.cols, d_X.cols, d_W.rows, &alpha, d_W.data, d_W.rows, d_X.data, d_X.rows, &beta, d_Z.data, d_Z.rows);
                gpuErrchk(cudaPeekAtLastError());

                if (multstat != CUBLAS_STATUS_SUCCESS)
                {
                    printf("CUBLAS matrix multiplication failed 3 %d\n", multstat);
                }

                /* compute softmax per column of Z and store in P */
                d_P = softmax_col(d_Z);
                gpuErrchk(cudaPeekAtLastError());

                // Q := P-Y
                fmatrix d_Q = fmatrix_add(d_P, -1.0f, d_Y);
                gpuErrchk(cudaPeekAtLastError());

                // compute gradient G = XQ^T
                multstat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d_X.rows, d_Q.rows, d_X.cols, &alpha, d_X.data, d_X.rows, d_Q.data, d_Q.rows, &beta, d_G.data, d_G.rows);
                gpuErrchk(cudaPeekAtLastError());

                if (multstat != CUBLAS_STATUS_SUCCESS)
                {
                    printf("CUBLAS matrix multiplication failed 4\n");
                }

                // update weights W = W - learning_rate*G
                d_W = fmatrix_add(d_W, -learning_rate, d_G);
                gpuErrchk(cudaPeekAtLastError());

                /* Compute J for reporting */
                J = evaluate_logloss(handle, d_P, d_Y, verbose);

                /* Increase the pointer */
                batch_pointer += batch_size;
            }

            /* Log accuracy and loss */
            if (i % (nb_iter / periods) == 0)
            {
                float accuracy = evaluate_accuracy(handle, d_W, d_Xtest, d_Ytest, d_Ztest, verbose);
                fprintf(fp, "%f,%f\n", J, accuracy);
            }

            /* reduce the learning rate */
            learning_rate *= rate_decay;
        }
        t_end = clock();
        float duration = ((float)(t_end - t_start_total)) / CLOCKS_PER_SEC;
        /* Evaluate the accuracy */
        accuracy = evaluate_accuracy(handle, d_W, d_Xtest, d_Ytest, d_Ztest, verbose);

        /* Memory clean up */
        fmatrix_free_on_host(&h_W);
        fmatrix_free_on_host(&Xall);
        fmatrix_free_on_host(&Yall);

        fmatrix_free_on_device(&d_X_tot);
        fmatrix_free_on_device(&d_Y_tot);
        fmatrix_free_on_device(&d_Xtest);
        fmatrix_free_on_device(&d_Ytest);
        fmatrix_free_on_device(&d_W);
        fmatrix_free_on_device(&d_Z);
        fmatrix_free_on_device(&d_J);
        cublasDestroy(handle);

        // Close log file
        fclose(fp);
        return accuracy;
    }
}