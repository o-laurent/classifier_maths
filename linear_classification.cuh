#ifndef LINEAR_CLASSIFICATION_H
#define LINEAR_CLASSIFICATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <fstream>

/*Matrix multiplication functions and other auxiliary functions*/
#include "fmatrix.cuh"
#include "read_csv.cuh"
#include "preprocess_data.cuh"
#include "classifier_math.cuh"
#include "evaluate_accuracy.cuh"
#include "cublas_v2.h"
#include "normalization.cuh"

/* Includes, cuda */
#include <cuda_runtime.h>
#endif