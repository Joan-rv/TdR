class XarxaNeuronal():
    def __init__(self, capes):
        self.capes = capes
    
    def propaga(self, X):
        self.sortida = X
        for capa in self.capes:
            self.sortida = capa.propaga(self.sortida)
        return self.sortida
    
    def retropropaga(self, alfa, d_error, Y):
        delta = d_error(Y, self.sortida)
        for capa in reversed(self.capes):
            delta = capa.retropropaga(delta, alfa)
    
    def __srt__(self):
        return self.capes.__str__()

    __repr__ = __srt__