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

# derivada de sigmoide
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
    W_l = []
    b_l = []
    for i in range(len(dimesnisons)-1):
        W_l.append(np.random.randn(dimesnisons[i+1], dimesnisons[i]))
    for dimenió in dimesnisons[1:]:
        b_l.append(np.random.randn(dimenió, 1))

    return W_l, b_l

def propaga(W_l, b_l, imatges):
    Z_l = []
    A_l = []
    # valor inicial depén de imatges i no de A_l
    Z_l.append(W_l[0].dot(imatges) + b_l[0])

    for i in range(1, len(W_l)):
        A_l.append(sigmoide(Z_l[i-1]))
        Z_l.append(W_l[i].dot(A_l[i - 1]) + b_l[i])
    
    A_l.append(softmax(Z_l[-1]))

    return Z_l, A_l

def one_hot(digits):
    one_hot_digits = np.zeros((digits.shape[0], 10))
    for (digit, one_hot_vec) in zip(digits, one_hot_digits):
        one_hot_vec[digit] = 1
    return one_hot_digits.T

def retropropaga(Z_l, A_l, W_l, b_l, digits, imatges):
    n, m = imatges.shape
    delta_l = [None] * len(W_l)
    dW_l = [None] * len(W_l)
    db_l = [None] * len(b_l)

    delta_l[-1] = 2*(A_l[-1] - one_hot(digits))

    for i in range(len(W_l) - 1, 0, -1): # desde (len(W_l) -1) (inclusiu) fins 0 (exclusiu)
        dW_l[i] = 1/m * (delta_l[i].dot(A_l[i - 1].T))
        db_l[i] = 1/m * (np.sum(delta_l[i], 1))

        delta_l[i - 1] = W_l[i].T.dot(delta_l[i]) * d_sigmoide(Z_l[i - 1])
    
    dW_l[0] = 1/m * (delta_l[0].dot(imatges.T))
    db_l[0] = 1/m * (np.sum(delta_l[0], 1))

    return dW_l, db_l

def actualitza_paràmetres(W_l, b_l, dW_l, db_l, alfa):
    W_l = [ W - dW * alfa for (W, dW) in zip(W_l, dW_l) ]
    b_l = [ b - np.reshape(db, b.shape) * alfa for (b, db) in zip(b_l, db_l) ]

    return W_l, b_l

if __name__ == '__main__':
    entrenament_digits, entrenament_imatges, prova_imatges = llegir_dades()
    precisió = 0
    alfa = 0.15
    iter = 0
    W_l, b_l = valors_inicials([784, 16, 10])
    while alfa < 0.9:
        iter += 1

        Z_l, A_l = propaga(W_l, b_l, entrenament_imatges)
        dW_l, db_l = retropropaga(Z_l, A_l, W_l, b_l, entrenament_digits, entrenament_imatges)

        W_l, b_l = actualitza_paràmetres(W_l, b_l, dW_l, db_l, alfa)

        precisió = np.sum(np.argmax(A_l[-1], 0) == entrenament_digits)/entrenament_digits.size
        if (iter % 10 == 0):
            print(f"Iteració: {iter}, precisió: {precisió*100:.2f}%", end="\r")