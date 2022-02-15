#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_stuff.cuh"
#include "fmatrix.cuh"

// for reading CSV files, we use some C++
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#define THREADS_PER_BLOCK

int fmatrix_elements(fmatrix mat)
{
    return mat.cols * mat.rows;
}

int fmatrix_size(fmatrix mat)
{
    return fmatrix_elements(mat) * sizeof(float);
}

void fmatrix_assert(fmatrix mat)
{
    assert(mat.data);
    assert(mat.cols);
    assert(mat.rows);
}

fmatrix fmatrix_create_on_host(int rows, int cols)
{
    assert(cols > 0);
    assert(rows > 0);
    fmatrix mat;
    mat.cols = cols;
    mat.rows = rows;
    mat.data = (float *)malloc(fmatrix_size(mat));
    assert(mat.data);
    return mat;
}

fmatrix fmatrix_create_on_device(int rows, int cols)
{
    assert(cols > 0);
    assert(rows > 0);
    fmatrix mat;
    mat.cols = cols;
    mat.rows = rows;
    gpuErrchk(
        cudaMalloc((void **)&(mat.data), fmatrix_size(mat)));
    return mat;
}

void fmatrix_data_to_device(fmatrix mat_host, fmatrix mat_device)
{
    fmatrix_assert(mat_host);
    fmatrix_assert(mat_device);
    assert(mat_host.cols == mat_device.cols);
    assert(mat_host.rows == mat_device.rows);
    gpuErrchk(
        cudaMemcpy(mat_device.data, mat_host.data,
                   fmatrix_size(mat_host),
                   cudaMemcpyHostToDevice));
}

void fmatrix_data_to_host(fmatrix mat_host, fmatrix mat_device)
{
    fmatrix_assert(mat_host);
    fmatrix_assert(mat_device);
    assert(mat_host.cols == mat_device.cols);
    assert(mat_host.rows == mat_device.rows);
    gpuErrchk(
        cudaMemcpy(mat_host.data, mat_device.data,
                   fmatrix_size(mat_device),
                   cudaMemcpyDeviceToHost));
}

fmatrix fmatrix_copy_to_host(fmatrix mat_device)
{
    fmatrix_assert(mat_device);
    fmatrix mat_host = fmatrix_create_on_host(mat_device.rows, mat_device.cols);
    fmatrix_data_to_host(mat_host, mat_device);
    return mat_host;
}

fmatrix fmatrix_copy_to_device(fmatrix mat_host)
{
    fmatrix_assert(mat_host);
    fmatrix mat_device = fmatrix_create_on_device(mat_host.rows, mat_host.cols);
    fmatrix_data_to_device(mat_host, mat_device);
    return mat_device;
}

/** We could do it like this, but it would not set our pointer M.data to 0.
... fmatrix_free_on_host(M)
void fmatrix_free_on_host(fmatrix mat) {
    fmatrix_assert(mat);
  free(mat.data);
  mat.data = 0;
  mat.cols = 0;
  mat.rows = 0;
}
*/

void fmatrix_free_on_host(fmatrix *mat)
{
    fmatrix_assert(*mat);
    free(mat->data);
    mat->data = 0;
    mat->cols = 0;
    mat->rows = 0;
}

void fmatrix_free_on_device(fmatrix *mat)
{
    fmatrix_assert(*mat);
    gpuErrchk(cudaFree(mat->data));
    mat->data = 0;
    mat->cols = 0;
    mat->rows = 0;
}

fmatrix fmatrix_subcolumns(fmatrix M, int a, int b)
{
    fmatrix_assert(M);
    fmatrix A = {
        .data = &getfm(M, 0, a),
        .cols = b - a,
        .rows = M.rows};
    fmatrix_assert(A);
    return A;
}

__host__
    __device__ void
    fmatrix_print(fmatrix mat, int nb)
{
    if (nb < 0 || nb > mat.rows)
    {
        nb = mat.rows;
    }
    printf("[\n");
    for (int i = 0; i < nb; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            printf("%f", getfm(mat, i, j));
            if (j + 1 < mat.cols)
            {
                printf(",\t");
            }
        }
        if (i + 1 < nb)
        {
            printf(";\n");
        }
    }
    if (nb < mat.rows)
    {
        printf("\n...\n");
    }
    printf("\n]\n");
}

void fmatrix_device_print(fmatrix mat, int nb)
{
    // allocate copy
    fmatrix tmp = fmatrix_copy_to_host(mat);
    fmatrix_print(tmp, nb);
    fmatrix_free_on_host(&tmp);
}

void fmatrix_device_to_csv(const char *filename, fmatrix mat)
{
    // Open file
    FILE *fp = fopen(filename, "w");
    // allocate copy
    fmatrix tmp = fmatrix_copy_to_host(mat);
    for (int i = 0; i < tmp.rows; i++)
    {
        for (int j = 0; j < tmp.cols; j++)
        {
            // Note: %.15g gives 15 significant digits (full double precision)
            fprintf(fp, "%.15g", getfm(tmp, i, j));
            if (j + 1 < tmp.cols)
            {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }
    fmatrix_free_on_host(&tmp);
    // Close file
    fclose(fp);
}

__global__ void fmatrix_create_random_on_device_kernel(fmatrix M, int seed = 0)
{
    // choose a seed (here: the same each launch)
    int sequence = 0;
    // first, initialize the random numbers
    curandState state;
    curand_init(seed, sequence, 0, &state);
    for (int i = 0; i < fmatrix_elements(M); ++i)
    {
        // curand_uniform creates numbers between 0 and 1
        M.data[i] = (curand_uniform(&state) - 0.5) * 2.0;
    }
}

__global__ void fmatrix_create_null_on_device_kernel(fmatrix M)
{
    for (int i = 0; i < fmatrix_elements(M); ++i)
    {
        M.data[i] = 0;
    }
}

__global__ void fmatrix_create_minus_inf_on_device_kernel(fmatrix M)
{
    for (int i = 0; i < fmatrix_elements(M); ++i)
    {
        M.data[i] = -INFINITY;
    }
}

fmatrix fmatrix_create_random_on_device(int rows, int cols, int seed /*= 0*/)
{
    // Create an uninitialized matrix on the device
    fmatrix M = fmatrix_create_on_device(rows, cols);
    // Call a kernel with a single thread to fill the values
    fmatrix_create_random_on_device_kernel<<<1, 1>>>(M, seed);

    return M;
}

fmatrix fmatrix_create_null_on_device(int rows, int cols)
{
    // Create an uninitialized matrix on the device
    fmatrix M = fmatrix_create_on_device(rows, cols);
    // Call a kernel with a single thread to fill the values
    fmatrix_create_null_on_device_kernel<<<1, 1>>>(M);

    return M;
}

fmatrix fmatrix_create_minus_inf_on_device(int rows, int cols)
{
    // Create an uninitialized matrix on the device
    fmatrix M = fmatrix_create_on_device(rows, cols);
    // Call a kernel with a single thread to fill the values
    fmatrix_create_minus_inf_on_device_kernel<<<1, 1>>>(M);

    return M;
}

/* Count the number of rows and columns in a csv files (without headers) */
void count_elements_in_csv(const char *filename, int *rows, int *cols)
{
    // Note: for the sake of convenience, we use some C++ functions here
    using namespace std;

    *rows = 0;
    *cols = 0;
    string row_as_string;
    string value;
    ifstream infile;
    infile.open(filename, ifstream::in);
    if (infile.is_open())
    {
        while (getline(infile, row_as_string, '\n'))
        {
            istringstream line_stream(row_as_string);
            int tempcols = 0;
            while (getline(line_stream, value, ','))
            {
                ++tempcols;
            }
            if (tempcols > *cols)
            {
                *cols = tempcols;
            }
            ++(*rows);
        }
        infile.close();
    }
    else
        cout << "Cannot open file." << endl;
}

/** Read the data from a csv file into an fmatrix on the host.
 *  Careful: We assume that the matrix has the right dimensions!
 *  Use count_elements_in_csv(...) to get the dimensions if
 *  unknown.
 */
void fmatrix_fill_from_csv(fmatrix h_M, const char *filename)
{
    // Note: for the sake of convenience, we use some C++ functions here
    using namespace std;
    string row_as_string;
    string value;
    ifstream infile;
    infile.open(filename, ifstream::in);
    int row = 0;
    if (infile.is_open())
    {
        while (getline(infile, row_as_string, '\n'))
        {
            istringstream line_stream(row_as_string);
            int col = 0;
            while (getline(line_stream, value, ','))
            {
                getfm(h_M, row, col) = strtod(value.c_str(), NULL);
                ++col;
            }
            ++row;
        }
        infile.close();
    }
    else
        cout << "Cannot open file." << endl;
}

fmatrix fmatrix_device_from_csv(const char *filename)
{
    // first read the file to count the number of elements
    int rows = 0;
    int cols = 0;
    count_elements_in_csv(filename, &rows, &cols);

    // allocate the matrix on the host
    fmatrix h_M = fmatrix_create_on_host(rows, cols);

    // read the data into the host matrix
    fmatrix_fill_from_csv(h_M, filename);

    // copy the matrix to the device
    fmatrix M = fmatrix_copy_to_device(h_M);

    // destroy the host matrix
    fmatrix_free_on_host(&h_M);

    return M;
}