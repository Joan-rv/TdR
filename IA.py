import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from xarxa_neuronal import XarxaNeuronal
from capes import Perceptró, Aplana, MaxPooling
from activacions import Sigmoide, ReLU, Softmax
from errors import eqm, d_eqm, entropia_creuada, d_entropia_creuada

def processa_imatges(imatges):
    return (imatges-np.min(imatges))/(np.max(imatges)-np.min(imatges))

def one_hot(Y):
    Y_one_hot = np.zeros((Y.shape[0], 10))
    for (y, y_one_hot) in zip(Y, Y_one_hot):
        y_one_hot[y] = 1
    return Y_one_hot.T

def imprimeix_imatge(imatge):
    pixels = imatge.reshape((28, 28))
    plt.imshow(pixels, cmap='Greys_r')
    plt.axis('off')
    plt.show()

def llegir_dades():
    # Llegir CSV i guardar en variables
    entrenament = pd.read_csv("train.csv").to_numpy()
    prova = pd.read_csv("test.csv").to_numpy()
    
    # Ordre aleatori
    np.random.shuffle(entrenament)


    # Obtenir primera columna
    entrenament_digits = entrenament[:, 0]
    # Obtenir les imatges i processar-les per facilitar les operacions
    entrenament_imatges = processa_imatges(entrenament[:, 1:])
    prova_imatges = processa_imatges(prova)
    return entrenament_digits, entrenament_imatges, prova_imatges

def main():
    np.seterr(all='raise', under='ignore')

    digits, imatges, _ = llegir_dades()
    imatges = np.reshape(imatges, (-1, 28, 28))

    X, X_prova = np.split(imatges, [40000])
    Y, Y_prova = np.split(one_hot(digits), [40000], axis=1)

    temp = np.random.permutation(len(X))
    X = X[temp]
    Y = Y.T[temp].T

    tamany_lots = 100
    X_lots = np.split(X, X.shape[0]/tamany_lots)
    Y_lots = np.split(Y, Y.shape[1]/tamany_lots, axis=1)
    
    xarxa = XarxaNeuronal([
        Aplana(),
        Perceptró(28**2, 256, optimitzador='adam'), 
        ReLU(),
        Perceptró(256, 128, optimitzador='adam'), 
        ReLU(),
        Perceptró(128, 10, optimitzador='adam'),
        Softmax(),
    ])

    alfa = 0.001

    iteracions = 10000
    for i in range(1, iteracions):
        for X_lot, Y_lot in zip(X_lots, Y_lots):
            sortida = xarxa.propaga(X_lot)

            xarxa.retropropaga(alfa, d_eqm, Y_lot, i)


        sortida = xarxa.propaga(X)
        precisió_entrenament = np.sum(np.argmax(sortida, 0) == np.argmax(Y, 0))/Y.shape[1]
        sortida = xarxa.propaga(X_prova)
        precisió_prova = np.sum(np.argmax(sortida, 0) == np.argmax(Y_prova, 0))/Y_prova.shape[1]
        print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%, precisió real: {precisió_prova*100:.2f}%")

if __name__ == '__main__':
    main()
