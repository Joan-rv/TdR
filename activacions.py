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


class Tanh(Activació):
    def __init__(self):
        super().__init__(self.tanh, self.d_tanh)

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.square(np.tanh(x))


class Softmax(Activació):
    def __init__(self):
        super().__init__(self.softmax, self.d_softmax)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=0))
        return exp / exp.sum(axis=0)

    def d_softmax(self, delta, *_):
        return 1
