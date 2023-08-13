import numpy as np
from scipy.signal import convolve2d, correlate2d
import optimitzadors

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
    def paràmetres_inicials(self, dim_entrada, dim_sortida, optimitzador):
        desviació_estàndard = np.sqrt(2/(dim_entrada + dim_sortida)) 
        W = np.random.normal(0, desviació_estàndard, (dim_sortida, dim_entrada))
        b = np.random.normal(0, desviació_estàndard, (dim_sortida, 1))
        optimitzador = optimitzadors.text_a_optimitzador(optimitzador, dim_sortida, dim_entrada)
        return W, b, optimitzador

    def __init__(self, dim_sortida, dim_entrada = None, optimitzador='cap'):
        self.dim_sortida = dim_sortida
        self.optimitzador = optimitzador
        self.W, self.b = None, None
        if dim_entrada is not None:
            self.W, self.b, self.optimitzador = self.paràmetres_inicials(dim_entrada, dim_sortida, optimitzador)
    
    def propaga(self, entrada):
        if self.W is None:
            self.W, self.b, self.optimitzador = self.paràmetres_inicials(entrada.shape[0], self.dim_sortida, self.optimitzador)
        self.entrada = entrada
        return self.W.dot(entrada) + self.b

    def retropropaga(self, delta, alfa, iter):
        _, m = self.entrada.shape

        delta_nou = self.W.T.dot(delta)


        dW = 1/m * delta.dot(self.entrada.T)
        db = np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        self.W, self.b = self.optimitzador.actualitza(alfa, self.W, dW, self.b, db, iter)
        #self.W -= alfa * 1/m * delta.dot(self.entrada.T)
        #self.b -= alfa * 1/m * np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        return delta_nou

    def __str__(self):
        return self.__class__.__name__ + str(self.W.shape) + self.optimitzador.__class__.__name__
    
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
        self.entrada = entrada
        if entrada.shape[2] % self.forma[0] != 0 or entrada.shape[3] % self.forma[1] != 0:
            entrada = np.pad(entrada, ((0,0), (0,0), (0, self.forma[0] - entrada.shape[2] % self.forma[0]),
                                    (0, self.forma[1] - entrada.shape[3] % self.forma[1])), constant_values=-np.inf)
        n_entrades, canals, altura, amplada = entrada.shape
        sortida_altura = altura // self.forma[0]
        sortida_amplada = amplada // self.forma[1]
        sortida = np.zeros((n_entrades, canals, sortida_altura, sortida_amplada))
        self.index_maxs = np.zeros((n_entrades, canals, sortida_amplada*sortida_altura))

        for i in range(entrada.shape[0]):
            for j in range(entrada.shape[1]):
                blocs = entrada[i,j].reshape((-1, self.forma[0], entrada.shape[3]//self.forma[1], self.forma[1])).transpose((0,2,1,3)).reshape(-1, self.tamany)
                self.blocs = blocs
                self.index_maxs[i,j] = np.argmax(blocs, axis=1)
                sortida[i,j] = np.max(blocs, axis=1).reshape(sortida.shape[2:])

        return sortida
    
    def retropropaga(self, delta, *_):
        delta_amplada, delta_altura = delta.shape[2:]
        delta = delta.reshape(delta.shape[0], delta.shape[1], -1)
        delta_nou = np.zeros(self.entrada.shape)

        quocients, residus = np.divmod(self.index_maxs, self.forma[0])
        x = np.arange(delta.shape[2])
        index_x = (residus + self.forma[0]*(x % delta_amplada)).astype(int)
        index_y = (quocients + self.forma[0]*(x // delta_altura)).astype(int)
        delta_nou.reshape((-1))[..., index_y + index_x * delta_nou.shape[3]] = delta

        return delta_nou
    
    def __str__(self):
        return self.__class__.__name__ + str(self.forma[0])
    def __repr__(self):
        return self.__str__()

class Convolució(Capa):
    def paràmetres_inicials(self, n_kernels, forma_entrada, dim_kernel, optimitzador):
        self.forma_sortida = (forma_entrada[2] - dim_kernel + 1, forma_entrada[3] - dim_kernel + 1)
        kernels = np.random.randn(n_kernels, forma_entrada[1], dim_kernel, dim_kernel)
        biaix = np.random.randn(n_kernels, *self.forma_sortida)
        optimitzador = optimitzadors.text_a_optimitzador(optimitzador, self.forma_sortida, forma_entrada)
        return kernels, biaix, optimitzador
    def __init__(self, dim_kernel, n_kernels, forma_entrada=None, optimitzador='cap'):
        self.n_kernels = n_kernels
        self.dim_kernel = dim_kernel
        self.optimitzador = optimitzador
        
        self.kernels, self.biaix = None, None
        if forma_entrada is not None:
            self.kernels, self.biaix, self.optimitzador = self.paràmetres_inicials(n_kernels, forma_entrada, dim_kernel, optimitzador)


    def propaga(self, entrada):
        if self.kernels is None:
            self.kernels, self.biaix, self.optimitzador = self.paràmetres_inicials(self.n_kernels, entrada.shape, self.dim_kernel, self.optimitzador)
        self.entrada = entrada
        sortida = np.zeros((entrada.shape[0], *self.biaix.shape))
        for i in range(entrada.shape[0]):
            for j in range(entrada.shape[1]):
                for k in range(self.kernels.shape[0]):
                    sortida[i,k] += correlate2d(entrada[i,j], self.kernels[k, j], mode='valid') + self.biaix[k]

        return sortida
    
    def retropropaga(self, delta, alfa, iter):
        dK = np.zeros_like(self.kernels)
        delta_nou = np.zeros((100, 1, 28, 28))
        for i in range(self.entrada.shape[0]):
            for j in range(self.entrada.shape[1]):
                for k in range(self.kernels.shape[0]):
                    dK[k,j] += correlate2d(self.entrada[i,j], delta[i,k], mode='valid')
                    delta_nou[i,j] += correlate2d(delta[i,k], self.kernels[k,j], mode='full')
        
        self.kernels, self.biaix = self.optimitzador.actualitza(alfa, self.kernels, dK, self.biaix, np.sum(delta), iter)
        #self.kernels -= alfa * dK
        #self.biaix -= alfa * np.sum(delta)
    
    def __srt__(self):
        return self.__class__.__name__ + str(self.dim_kernel) + str(self.forma_sortida) + str(self.n_kernels)
    def __repr__(self):
        return self.__srt__()
