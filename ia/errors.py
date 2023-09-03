import numpy as np


def eqm(y, ŷ):
    return np.sum(np.square(ŷ - y)) / y.size


def d_eqm(y, ŷ):
    return 2*(ŷ - y)


def eam(y, ŷ):
    return np.sum(np.abs(ŷ - y)) / y.size


def d_eam(y, ŷ):
    return np.greater(ŷ, y) * 2 - 1


def eqlm(y, ŷ):
    return np.sum(np.square(np.log(1 + ŷ) - np.log(1 + y))) / y.size


def d_eqlm(y, ŷ):
    return 2*(np.log(1 + ŷ) - np.log(1 + y)) * 1 / (1 + ŷ)


def entropia_creuada(y, ŷ):
    return -np.sum(y*np.log(ŷ + 1e-7) + (1 - y)*np.log(1 - ŷ + 1e-7)) / y.size


def d_entropia_creuada(y, ŷ):
    return (-y/(ŷ + 1e-70) + (1 - y)/(1 - ŷ + 1e-70))
