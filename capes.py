import numpy as np

class Capa():
    def __init__(self):
        pass

    def propaga():
        pass

    def retropropaga():
        pass


class Perceptró(Capa):
    def __init__(self, dimensions_entrada, dimensions_sortida):
        self.W = np.random.randn(dimensions_sortida, dimensions_entrada)
        self.b = np.random.randn(dimensions_sortida, 1)
    
    def propaga(self, entrada):
        self.entrada = entrada
        return self.W.dot(entrada) + self.b

    def retropropaga(self, delta, alfa):
        _, m = self.entrada.shape

        delta_nou = self.W.T.dot(delta)

        self.W -= alfa * 1/m * delta.dot(self.entrada.T)
        self.b -= alfa * 1/m * np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        return delta_nou
        