import numpy as np

# error quadràtic mig
def eqm(y, ŷ):
    return np.sum(np.square(ŷ - y)) / y.size

# derivada
def d_eqm(y, ŷ):
    return 2*(ŷ - y)