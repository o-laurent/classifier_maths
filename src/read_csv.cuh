#ifndef read_csv_H
#define read_csv_H
#include <cuda_runtime.h>

void read_csv(const char *filename, float *data_array, int nbrow, int nbcol);

#endif