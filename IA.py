import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from capes import Perceptró
from activacions import Sigmoide, Softmax
from errors import eqm, d_eqm

def processa_imatges(imatges):
    imatges = (imatges-np.min(imatges))/(np.max(imatges)-np.min(imatges))
    imatges = imatges.T
    return imatges

def one_hot(Y):
    Y_one_hot = np.zeros((Y.shape[0], 10))
    for (y, y_one_hot) in zip(Y, Y_one_hot):
        y_one_hot[y] = 1
    return Y_one_hot.T

def imprimeix_imatge(image):
    pixels = image.reshape((28, 28))
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
    entrenament_digits, entrenament_imatges, prova_imatges = llegir_dades()

    X = entrenament_imatges
    Y = one_hot(entrenament_digits)

    xarxa = [
        Perceptró(28**2, 128), 
        Sigmoide(),
        Perceptró(128, 10),
        Softmax(),
    ]

    iteracions = 1000
    for i in range(iteracions):
        sortida = X 
        for capa in xarxa:
            sortida = capa.propaga(sortida)
        
        precisió = np.sum(np.argmax(sortida, 0) == entrenament_digits)/entrenament_digits.size
        print(f"Iteració: {i}; precisió: {precisió*100:.2f}%")

        alfa = 0.1
        delta = d_eqm(Y, sortida)
        for capa in reversed(xarxa):
            delta = capa.retropropaga(delta, alfa)


if __name__ == '__main__':
    main()