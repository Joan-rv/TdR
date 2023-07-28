import numpy as np

# error quadràtic mig
def eqm(y, ŷ):
    return np.sum(np.square(ŷ - y)) / y.size

# derivada
def d_eqm(y, ŷ):
    return 2*(ŷ - y)

def entropia_creuada(y, ŷ):
    return -np.mean(y*np.log(ŷ) + (1 - y)*np.log(1 - ŷ))

def d_entropia_creuada(y, ŷ):
    return - (y/(ŷ + 1e-70) - (1 - y)/(1 - ŷ))