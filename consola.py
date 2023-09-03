from ia.capes import Perceptró, Convolució, MaxPooling, Aplana
from ia.activacions import ReLU, Softmax
from ia.errors import eqm, d_eqm
import ia.utils as utils
from ia.xarxa_neuronal import XarxaNeuronal

import numpy as np


def main():
    np.seterr(all='raise', under='ignore')

    xarxa = XarxaNeuronal([
        Convolució(3, 16),
        ReLU(),
        MaxPooling(2),
        Aplana(),
        Perceptró(256, optimitzador='adam'),
        ReLU(),
        Perceptró(128, optimitzador='adam'),
        ReLU(),
        Perceptró(10, optimitzador='adam'),
        Softmax(),
    ])

    # Definir hiperparàmetres
    alfa = 0.001
    iteracions = 100
    tamany_lots = 100
    f_error = eqm
    d_error = d_eqm

    # Llegir dades i crear lots
    digits, imatges, _ = utils.llegir_dades()
    imatges = np.reshape(imatges, (-1, 28, 28, 1))

    X, X_prova = np.split(imatges, [40000])
    Y, Y_prova = np.split(utils.one_hot(digits), [40000], axis=1)

    X_lots = np.split(X, X.shape[0]/tamany_lots)
    Y_lots = np.split(Y, Y.shape[1]/tamany_lots, axis=1)

    # Entrenar xarxa neuronal
    for i in range(1, iteracions):
        precisió = 0
        error = 0
        for X_lot, Y_lot in zip(X_lots, Y_lots):
            sortida = xarxa.propaga(X_lot)
            precisió += np.sum(np.argmax(sortida, 0) ==
                               np.argmax(Y_lot, 0))/Y.shape[1]
            error += f_error(Y_lot, sortida)/len(X_lots)

            xarxa.retropropaga(alfa, d_error, Y_lot, i)

        print(
            f"Iteració: {i}; error: {error:.6f}; precisió: {precisió*100:.2f}%")


if __name__ == '__main__':
    main()
