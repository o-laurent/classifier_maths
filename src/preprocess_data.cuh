#ifndef preprocess_data_H
#define preprocess_data_H
#include <cuda_runtime.h>

void get_inputs_and_labels(float *data_array, float **input_array, float **label_array, int nbrows, int nbcols, int nb_inputs, int nb_labels, bool verbose = false);

#endif