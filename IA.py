import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xarxa_neuronal import XarxaNeuronal
from capes import Perceptró, Aplana, MaxPooling, Convolució
from activacions import Sigmoide, ReLU, Softmax
from errors import eqm, d_eqm, entropia_creuada, d_entropia_creuada
from utils import normalitza_dades, one_hot, imprimeix_imatge


def llegir_dades():
    # Llegir CSV i guardar en variables
    entrenament = pd.read_csv("train.csv").to_numpy()
    prova = pd.read_csv("test.csv").to_numpy()

    # Ordre aleatori
    np.random.shuffle(entrenament)

    # Obtenir primera columna
    entrenament_digits = entrenament[:, 0]
    # Obtenir les imatges i normalitzar-les per facilitar les operacions
    entrenament_imatges = normalitza_dades(entrenament[:, 1:])
    prova_imatges = normalitza_dades(prova)
    return entrenament_digits, entrenament_imatges, prova_imatges


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
    digits, imatges, _ = llegir_dades()
    imatges = np.reshape(imatges, (-1, 28, 28, 1))

    X, X_prova = np.split(imatges, [40000])
    Y, Y_prova = np.split(one_hot(digits), [40000], axis=1)

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
