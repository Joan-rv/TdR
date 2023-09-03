class XarxaNeuronal():
    def __init__(self, capes):
        self.capes = capes

    def propaga(self, X):
        self.sortida = X
        # La sortida d'una capa és l'entrada de la següent
        for capa in self.capes:
            self.sortida = capa.propaga(self.sortida)
        return self.sortida

    def retropropaga(self, alfa, d_error, Y, iteració):
        delta = d_error(Y, self.sortida)
        # El gradient de sortida d'una capa és el d'entrada de la següent
        for capa in reversed(self.capes):
            delta = capa.retropropaga(delta, alfa, iteració)

    def __srt__(self):
        return self.capes.__str__()

    __repr__ = __srt__
