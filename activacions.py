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
        super().__init__(self.sigmoide, self.d_sigmoide)

    def sigmoide(self, x):
        return 1/(1 + np.exp(-x))
    
    def d_sigmoide(self, x):
        return self.sigmoide(x) * (1 - self.sigmoide(x))

class ReLU(Activació):
    def __init__(self):
        super().__init__(self.relu, self.d_relu)

    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return (x > 0) * 1

class Softmax(Capa):
    def propaga(self, x):
        exp = np.exp(x - np.max(x, axis=0))
        return exp / exp.sum(axis=0)
        
    def retropropaga(self, delta, *_):
        return delta
        #n = np.size(self.sortida)
        #return ((np.identity(n) - self.sortida.T) * self.sortida).dot(delta)