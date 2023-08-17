import numpy as np
import matplotlib.pyplot as plt

def normalitza_dades(dades):
    return (dades-np.min(dades))/(np.max(dades)-np.min(dades))

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

def finestres(a, forma_finestres, gambada_x=1, gambada_y=1):
    a = np.lib.stride_tricks.sliding_window_view(a, forma_finestres, axis=(-2, -1))
    return a[...,0::gambada_x,0::gambada_y,:,:]