#ifndef cuda_stuff_H
#define cuda_stuff_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//MACRO TO DEBUG CUDA FUNCTIONS
/** Error checking,
 *  taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans)                      \
   {                                        \
      gpuAssert((ans), __FILE__, __LINE__); \
   }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

/* transform matrix index to vector offset
   Since CUDA uses column major, 
   nb_rows = number of rows */
#define IDX2C(i, j, nb_rows) (((j) * (nb_rows)) + (i))

void device_synchronize();

#endif
