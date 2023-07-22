import numpy as np
from capes import Capa

class Activació(Capa):
    def __init__(self, funció, derivada) -> None:
        self.funció = funció
        self.derivada = derivada
    
    def propaga(self, entrada):
        self.entrada = entrada
        return self.funció(entrada)
    
    def retropropaga(self, delta, *_):
        return self.derivada(self.entrada) * delta

class Sigmoide(Activació):
    def __init__(self):
        global sigmoide, d_sigmoide
        def sigmoide(x):
            return 1/(1 + np.exp(-x))
        
        def d_sigmoide(x):
            return sigmoide(x) * (1 - sigmoide(x))

        super().__init__(sigmoide, d_sigmoide)

class ReLU(Activació):
    def __init__(self):
        global relu, d_relu
        def relu(x):
            return np.maximum(0, x)
        
        def d_relu(x):
            return (x > 0) * 1
        
        super().__init__(relu, d_relu)

class Softmax(Capa):
    def propaga(self, x):
        exp = np.exp(x - np.max(x, axis=0))
        return exp / exp.sum(axis=0)
        
    def retropropaga(self, delta, *_):
        return delta
        #n = np.size(self.sortida)
        #return ((np.identity(n) - self.sortida.T) * self.sortida).dot(delta)