from ia.capes import Perceptró, Convolució, MaxPooling, Aplana
from ia.activacions import Sigmoide, Tanh, ReLU, Softmax
from ia.xarxa_neuronal import XarxaNeuronal

xarxa = XarxaNeuronal([
    Convolució(3, 32),
    ReLU(),
    MaxPooling(dim_pool=3),
    Aplana(),
    Perceptró(256, optimitzador='adam'),
    ReLU(),
    Perceptró(128, optimitzador='adam'),
    ReLU(),
    Perceptró(10, optimitzador='adam'),
    Softmax(),
])
