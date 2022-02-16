""" Optimisation script for the linear classification. """

import matplotlib.pyplot as plt
import ctypes
import optuna

# Link with the shared library
classifier_lib = ctypes.CDLL('./lclass.so')

# nb_iter, batch_size, learning rate, rate_decay
classifier_lib.linear_classification.argtypes = [
    ctypes.c_uint, ctypes.c_uint, ctypes.c_float, ctypes.c_float]

# accuracy
classifier_lib.linear_classification.restype = ctypes.c_float


def plot_file(filename="log.txt"):
    """ Plot a log file.

    Args:
        filename (str, optional): The path to the file. Defaults to "log.txt".
    """
    data = pd.read_csv('log.txt', sep=',', header=None)
    fig, ax = plt.subplots()
    ax.plot(data[0], label="logloss")
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot([], [])
    ax2.plot(data[1], label="accuracy")
    ax2.legend()
    plt.show()


def optimize(trial):
    """ Wrapper for optuna. """
    batch_size = int(trial.suggest_loguniform('batch_size', 1, 12000))
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-10, 1)
    rate_decay = trial.suggest_float('rate_decay', 0, 1)
    return -classifier_lib.linear_classification(10, batch_size, learning_rate, rate_decay)


do_optimize = True  # Set to true to optimize
n_trials = 100  # Number of tries

if do_optimize:
    study = optuna.create_study()
    study.optimize(optimize, n_jobs=1, n_trials=n_trials)
    print(study.best_params)
    batch_size = study.best_params['batch_size']
    learning_rate = study.best_params['learning_rate']
    rate_decay = study.best_params['rate_decay']
