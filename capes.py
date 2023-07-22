import numpy as np
import optimitzadors

class Capa():
    def __init__(self):
        pass

    def propaga():
        pass

    def retropropaga():
        pass

    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return self.__str__()


class Perceptró(Capa):
    def __init__(self, dimensions_entrada, dimensions_sortida, optimitzador='cap'):
        # inicialització Xavier
        maxim = (1.0 / np.sqrt(dimensions_entrada + dimensions_sortida))
        minim = - maxim
        self.W = minim + np.random.random((dimensions_sortida, dimensions_entrada)) * (maxim - minim)
        self.b = minim + np.random.random((dimensions_sortida, 1)) * (maxim - minim)
        if optimitzador == 'cap':
            self.optimitzador = optimitzadors.Cap()
        elif optimitzador == 'adam':
            self.optimitzador = optimitzadors.Adam(dimensions_sortida, dimensions_entrada)
        else:
            raise Exception(f"Optimitzador {optimitzador} desconegut")
    
    def propaga(self, entrada):
        self.entrada = entrada
        return self.W.dot(entrada) + self.b

    def retropropaga(self, delta, alfa, iter):
        _, m = self.entrada.shape

        delta_nou = self.W.T.dot(delta)


        #dW = 1/m * delta.dot(self.entrada.T)
        #db = np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        #self.W, self.b = self.optimitzador.actualitza(alfa, self.W, dW, self.b, db, iter)
        self.W -= alfa * 1/m * delta.dot(self.entrada.T)
        self.b -= alfa * 1/m * np.reshape(1/m * np.sum(delta, 1), self.b.shape)

        return delta_nou

    def __str__(self):
        return self.__class__.__name__ + str(self.W.shape) + self.optimitzador.__class__.__name__
    
    def __repr__(self):
        return self.__str__()
        