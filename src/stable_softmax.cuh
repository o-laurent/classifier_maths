#ifndef stable_softmax_H
#define stable_softmax_H

#include "fmatrix.cuh"
#include <assert.h>

/* Compute and apply the stable softmax */
fmatrix stable_softmax(fmatrix Z_d);

#endif