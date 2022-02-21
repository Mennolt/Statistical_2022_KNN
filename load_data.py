import numpy as np

def load_data(loc : str) -> np.ndarray:
    """
    Loads a minst datafile and converts it into an numpy ndarray
    :param loc: location of the datafile
    :return: the data as an ndarray
    """
    return np.genfromtxt(loc, delimiter=",")