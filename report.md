# Linear Classification using CUDA

## Presentation

All three questions have been implemented and optimised to some extent. The CuBLAS handle is for instance shared all along the computations.

The hyperparameters are optimized using Parzen Tree Estimators (via optuna). The objective of the optimization is to maximize the accuracy on the test set, while keeping the same number of epochs - total time would be more interesting than the number of epochs considering the speedup using greater batches, but it was deemed more difficult to control. Interesting hyperparameters include {'batch_size': 256, 'learning_rate': 0.0193, 'rate_decay': 0.57} with a test accuracy of 0.8478. The computation takes less than 1s with the following configuration: i7-11800H + RTX3070M.

The CUDA C++ code is called by python using ctypes. Memory leaks have been reduced to the minimum.

## How to use the software 

### CUDA

Change the parameters in the `src/linear_classification.cu` and `src/linear_classification_batch.cu` to your likings.

Type `make all -j` in the root directory to build the 2 binaries and the shared library.

For the batch linear classification, just type `./linear_classification`.

For the mini-batch linear classification, just type `./batch_linear_classification`.

Use `make check` and `make batch_check` for memory checks using cuda-gdb.

### Python for hyperparameters optimization

Create a virtualenv with Python 3 (3.10.1 was used during development) and type `pip install -r requirements.txt`.

Set `do_optimize = True` in `optimize.py`. Set the number of iterations for the optimization and run `python3 optimize.py`.

The obtained parameters might be overfitted on the test set.