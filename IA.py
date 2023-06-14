#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Funcions útils

def imprimeix_imatge(image):
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='Greys')
    plt.axis('off')
    plt.show()

def sigmoide(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0)

# derivada de relu
def d_sigmoide(x):
    return sigmoide(x) * (1-sigmoide(x))

def processa_imatges(imatges):
    imatges = (imatges-np.min(imatges))/(np.max(imatges)-np.min(imatges))
    imatges = imatges.T
    return imatges

# LLegir dades i processar-les

def llegir_dades():
    entrenament = pd.read_csv("train.csv").to_numpy()
    prova = pd.read_csv("test.csv").to_numpy()
    np.random.shuffle(entrenament)


    entrenament_digits = entrenament[:, 0]
    entrenament_imatges = processa_imatges(entrenament[:, 1:])
    prova_imatges = processa_imatges(prova)
    return entrenament_digits, entrenament_imatges, prova_imatges

# Inicialtizar valors i definir funcions

def valors_inicials():
    w1 = np.random.randn(16, 28*28)
    w2 = np.random.randn(10, 16)
    b1 = np.random.randn(16, 1)
    b2 = np.random.randn(10, 1)
    return w1, w2, b1, b2

def propaga(w1, b1, w2, b2, imatges):
    z1 = w1.dot(imatges) + b1
    a1 = sigmoide(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(digits):
    one_hot_digits = np.zeros((digits.shape[0], 10))
    for (digit, one_hot_vec) in zip(digits, one_hot_digits):
        one_hot_vec[digit] = 1
    return one_hot_digits.T

def retropropaga(z1, a1, z2, a2, w2, digits, imatges):
    tamany, m = imatges.shape
    dz2 = 2*(a2 - one_hot(digits))
    dw2 = 1/m * (dz2.dot(a1.T))
    db2 = 1/m * (np.sum(dz2, 1))
    dz1 = w2.T.dot(dz2)*d_sigmoide(z1)
    dw1 = 1/m * (dz1.dot(imatges.T))
    db1 = 1/m * np.sum(dz1,1)

    return dw1, db1, dw2, db2

if __name__ == '__main__':
    entrenament_digits, entrenament_imatges, prova_imatges = llegir_dades()
    w1, w2, b1, b2 = valors_inicials()
    imprimeix_imatge(entrenament_imatges.T[0])
    print(entrenament_digits[0])
    precisió = 0
    alfa = 0.15
    iter = 0
    while alfa < 0.9:
        iter += 1

        z1, a1, z2, a2 = propaga(w1, b1, w2, b2, entrenament_imatges)
        print(a2.shape)
        dw1, db1, dw2, db2 = retropropaga(z1, a1, z2, a2, w2, entrenament_digits, entrenament_imatges)

        w1 -= alfa * dw1
        b1 -= alfa * np.reshape(db1, (16,1))
        w2 -= alfa * dw2
        b2 -= alfa * np.reshape(db2, (10, 1))

        precisió = np.sum(np.argmax(a2, 0) == entrenament_digits)/entrenament_digits.size

        print(f"Iteració: {iter}, precisió: {precisió*100:.2f}%", end="\r")