#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>

/*Matrix multiplication functions and other auxiliary functions*/
#include "preprocess_data.cuh"

using namespace std;

/* transform matrix index to vector offset
   Since CUDA uses column major,
   ld = number of rows
   Example of use: a[IDX2C(0, 1, 50)] */
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

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
// Functions for preprocessing the data set
/////////////////////////////////////////////////////////

/* Split data into inputs and labels. Allocated memory for inputs and labels.
   Since cuBLAS is column major, each input is in a column.
   We also add 1.0 as first element to each input vector.
*/
void get_inputs_and_labels(float *data_array, float **input_array, float **label_array, int nbrows, int nbcols, int nb_inputs, int nb_labels, bool verbose /*= false */)
{
    // The inputs are the first nbrows-1 columns.
    // The labels are the last column (index nbrows-1), booleanized
    // by the condition >= above_threshold
    *input_array = (float *)malloc(nbrows * nb_inputs * sizeof(float));
    *label_array = (float *)malloc(nbrows * nb_labels * sizeof(float));
    // cout << &input_array << " and "<< &label_array << " data " << data_array << std::endl;
    if (verbose)
    {
        cout << "Allocated memory for inputs: " << nbrows << " rows, " << nb_inputs << " columns." << std::endl;
        cout << "Allocated memory for labels: " << nbrows << " rows, " << nb_labels << " columns." << std::endl;
    }
    // Copy the data to X
    for (int i = 0; i < nbrows; i++)
    {
        // Set the first element of each x to 1
        (*input_array)[IDX2C(0, i, nb_inputs)] = 1.0;
        // Copy the rest of x
        for (int j = 1; j < nb_inputs; j++)
        {
            (*input_array)[IDX2C(j, i, nb_inputs)] = data_array[IDX2C(i, j - 1, nbrows)];
        }
        float median_house_value = data_array[IDX2C(i, nbcols - 1, nbrows)];
        (*label_array)[IDX2C(0, i, nb_labels)] = 0.0;
        (*label_array)[IDX2C(1, i, nb_labels)] = 0.0;
        if (median_house_value >= above_threshold)
        {
            (*label_array)[IDX2C(0, i, nb_labels)] = 1.0;
        }
        else
        {
            (*label_array)[IDX2C(1, i, nb_labels)] = 1.0;
        }
    }

    // Show some entries for double checking
    if (verbose)
    {
        cout << "Inputs (first " << print_rows << "):" << std::endl;
        for (int j = 0; j < nb_inputs; j++)
        {
            for (int i = 0; i < nbrows && i < print_rows; i++)
            {
                cout << (*input_array)[IDX2C(j, i, nb_inputs)] << "\t";
            }
            cout << "\n";
        }
        cout << "Labels (first " << print_rows << "):" << std::endl;
        for (int j = 0; j < nb_labels; j++)
        {
            for (int i = 0; i < nbrows && i < print_rows; i++)
            {
                cout << (*label_array)[IDX2C(j, i, nb_labels)] << "\t";
            }
            cout << "\n";
        }
    }
}