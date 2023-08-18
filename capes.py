import numpy as np
from scipy.signal import convolve2d, correlate2d
import optimitzadors
import utils

class Capa():
    def __init__(self):
        pass
    
    def propaga(self):
        pass

    def retropropaga(self):
        pass

    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return self.__str__()


class Perceptró(Capa):
    def paràmetres_inicials(self, dim_entrada, dim_sortida, optimitzador_txt):
        desviació_estàndard = np.sqrt(2/(dim_entrada + dim_sortida)) 
        W = np.random.normal(0, desviació_estàndard, (dim_sortida, dim_entrada))
        b = np.random.normal(0, desviació_estàndard, (dim_sortida, 1))
        optimitzador = optimitzadors.text_a_optimitzador(optimitzador_txt, dim_sortida, dim_entrada)
        return W, b, optimitzador

    def __init__(self, dim_sortida, dim_entrada = None, optimitzador='cap'):
        self.dim_sortida = dim_sortida
        self.optimitzador_txt = optimitzador
        if dim_entrada is not None:
            self.W, self.b, self.optimitzador = self.paràmetres_inicials(dim_entrada, dim_sortida, optimitzador)
        else:
            self.W, self.b, self.optimitzador = None, None, optimitzadors.text_a_optimitzador(optimitzador, 0, 0)

    def propaga(self, entrada):
        if self.W is None:
            self.W, self.b, self.optimitzador = self.paràmetres_inicials(entrada.shape[0], self.dim_sortida, self.optimitzador_txt)
        self.entrada = entrada
        return self.W.dot(entrada) + self.b

    def retropropaga(self, delta, alfa, iter):
        _, m = self.entrada.shape

        delta_nou = self.W.T.dot(delta)

        dW = 1/m * delta.dot(self.entrada.T)
        db = np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        self.W, self.b = self.optimitzador.actualitza(alfa, self.W, dW, self.b, db, iter)

        return delta_nou

    def __str__(self):
        return self.__class__.__name__ + str((self.dim_sortida, self.optimitzador.__class__.__name__))
    
    def __repr__(self):
        return self.__str__()

class Aplana(Capa):
    def __init__(self):
        pass

    def propaga(self, entrada):
        self.forma = entrada.shape
        return entrada.reshape(entrada.shape[0], -1).T

    def retropropaga(self, delta, *_):
            return delta.reshape(self.forma)

class MaxPooling(Capa):
    def __init__(self, dim_pool = 2):
        self.forma = (dim_pool, dim_pool) 
        self.tamany = dim_pool**2
    
    def propaga(self, entrada):
        self.entrada_forma_vell = entrada.shape
        if entrada.shape[1] % self.forma[0] != 0 or entrada.shape[2] % self.forma[1] != 0:
            entrada = np.pad(entrada, ((0,0), (0, self.forma[0] - entrada.shape[1] % self.forma[0]),
                                    (0, self.forma[1] - entrada.shape[2] % self.forma[1]),(0,0)), constant_values=-np.inf)
        self.entrada_forma = entrada.shape
        n_entrades, altura, amplada, canals = entrada.shape
        sortida_altura = altura // self.forma[0]
        sortida_amplada = amplada // self.forma[1]
        blocs = utils.finestres(entrada, self.forma, axis=(-3, -2), gambades=self.forma).reshape(entrada.shape[0], -1, self.tamany, canals)
        sortida = np.max(np.swapaxes(blocs, 2, 3), axis=3).reshape((n_entrades, sortida_altura, sortida_amplada, canals))
        self.index_maxs = np.equal(blocs, sortida.reshape(n_entrades, -1, 1, canals))
        return sortida

    def retropropaga(self, delta, *_):
        delta = delta.reshape(delta.shape[0], -1, 1, delta.shape[-1])
        delta_nou = self.index_maxs * delta
        delta_nou = delta_nou.reshape((delta.shape[0], -1, self.forma[0], self.entrada_forma[2]//self.forma[1], self.forma[1], delta.shape[-1])).transpose((0, 1, 3, 2, 4, 5)).reshape(self.entrada_forma)
        return delta_nou[..., 0:self.entrada_forma_vell[1], 0:self.entrada_forma_vell[2],:]

    def __str__(self):
        return self.__class__.__name__ + str((self.forma[0]))
    def __repr__(self):
        return self.__str__()

class Convolució(Capa):
    def paràmetres_inicials(self, n_kernels, forma_entrada, dim_kernel):
        self.forma_entrada = forma_entrada
        self.forma_sortida = (forma_entrada[1] - dim_kernel + 1, forma_entrada[2] - dim_kernel + 1)
        W = np.random.randn(n_kernels, dim_kernel, dim_kernel, forma_entrada[1])
        b = np.random.randn(*self.forma_sortida, n_kernels)
        return W, b
    def __init__(self, dim_kernel, n_kernels, forma_entrada=None):
        self.n_kernels = n_kernels
        self.dim_kernel = dim_kernel
        
        self.W, self.b = None, None
        if forma_entrada is not None:
            self.W, self.b = self.paràmetres_inicials(n_kernels, forma_entrada, dim_kernel)


    def propaga(self, entrada):
        if self.W is None:
            self.W, self.b = self.paràmetres_inicials(self.n_kernels, entrada.shape, self.dim_kernel)
        self.entrada = entrada
        sortida = np.zeros((entrada.shape[0], *self.b.shape))
        for i in range(entrada.shape[0]):
            for j in range(entrada.shape[-1]):
                for k in range(self.W.shape[0]):
                    sortida[i,...,k] += correlate2d(entrada[i,...,j], self.W[k,...,j], mode='valid') + self.b[...,k]

        return sortida
    
    def retropropaga(self, delta, alfa, iter):
        dK = np.zeros_like(self.W)
        delta_nou = np.zeros(self.forma_entrada)
        for i in range(self.entrada.shape[0]):
            for j in range(self.entrada.shape[-1]):
                for k in range(self.W.shape[0]):
                    dK[k,...,j] += correlate2d(self.entrada[i,...,j], delta[i,...,k], mode='valid')
                    delta_nou[i,...,j] += convolve2d(delta[i,...,k], self.W[k,...,j], mode='full')
        
        self.W -= alfa * dK
        self.b -= alfa * np.sum(delta)
    
    def __srt__(self):
        return self.__class__.__name__ + str((self.dim_kernel, self.n_kernels))

    def __repr__(self):
        return self.__srt__()
