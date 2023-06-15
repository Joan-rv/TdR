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

def valors_inicials(dimesnisons):
    w1 = np.random.randn(128, 28*28)
    w2 = np.random.randn(64, 128)
    w3 = np.random.randn(10, 64)
    b1 = np.random.randn(128, 1)
    b2 = np.random.randn(64, 1)
    b3 = np.random.randn(10, 1)
    return w1, w2, w3, b1, b2, b3

def propaga(w1, b1, w2, b2, w3, b3, imatges):
    z1 = w1.dot(imatges) + b1
    a1 = sigmoide(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoide(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def one_hot(digits):
    one_hot_digits = np.zeros((digits.shape[0], 10))
    for (digit, one_hot_vec) in zip(digits, one_hot_digits):
        one_hot_vec[digit] = 1
    return one_hot_digits.T

def retropropaga(z1, a1, z2, a2, z3, a3, w2, w3, digits, imatges):
    tamany, m = imatges.shape
    dz3 = 2*(a3 - one_hot(digits))
    dw3 = 1/m * (dz3.dot(a2.T))
    db3 = 1/m * (np.sum(dz3, 1))
    dz2 = w3.T.dot(dz3)*d_sigmoide(z2)
    dw2 = 1/m * (dz2.dot(a1.T))
    db2 = 1/m * np.sum(dz2,1)
    dz1 = w2.T.dot(dz2)*d_sigmoide(z1)
    dw1 = 1/m * (dz1.dot(imatges.T))
    db1 = 1/m * np.sum(dz1, 1)

    return dw1, db1, dw2, db2, dw3, db3

if __name__ == '__main__':
    entrenament_digits, entrenament_imatges, prova_imatges = llegir_dades()
    w1, w2, w3, b1, b2, b3 = valors_inicials()
    imprimeix_imatge(entrenament_imatges.T[0])
    print(entrenament_digits[0])
    precisió = 0
    alfa = 0.15
    iter = 0
    while alfa < 0.9:
        iter += 1

        z1, a1, z2, a2, z3, a3 = propaga(w1, b1, w2, b2, w3, b3, entrenament_imatges)
        dw1, db1, dw2, db2, dw3, db3 = retropropaga(z1, a1, z2, a2, z3, a3, w2, w3, entrenament_digits, entrenament_imatges)

        w1 -= alfa * dw1
        b1 -= alfa * np.reshape(db1, (128,1))
        w2 -= alfa * dw2
        b2 -= alfa * np.reshape(db2, (16, 1))
        w3 -= alfa * dw3
        b3 -= alfa * np.reshape(db3, (10, 1))

        precisió = np.sum(np.argmax(a3, 0) == entrenament_digits)/entrenament_digits.size

        print(f"Iteració: {iter}, precisió: {precisió*100:.2f}%", end="\r")