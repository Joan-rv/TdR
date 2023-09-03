import numpy as np
import os
import matplotlib.pyplot as plt
import ia.utils
from ia.errors import eqm, d_eqm, eam, d_eam, eqlm, d_eqlm, entropia_creuada, d_entropia_creuada
from ia.activacions import Sigmoide, Tanh, ReLU, Softmax
from ia.capes import Perceptró, Aplana
from ia.xarxa_neuronal import XarxaNeuronal


def camí(nom):
    return os.path.join("grafs", nom)


tamany_lots = 100
# Llegir dades i crear lots
digits, imatges, _ = ia.utils.llegir_dades()
imatges = np.reshape(imatges, (-1, 28, 28, 1))

X, X_prova = np.split(imatges, [40000])
Y, Y_prova = np.split(ia.utils.one_hot(digits), [40000], axis=1)

X_lots = np.split(X, X.shape[0]/tamany_lots)
Y_lots = np.split(Y, Y.shape[1]/tamany_lots, axis=1)


def graf_precisió_error(model, alfa, f_error, d_error, ax1, ax2, label):
    print(label)
    precisions = []
    errors = []
    for i in range(1, 11):
        precisió = 0
        error = 0
        for X_lot, Y_lot in zip(X_lots, Y_lots):
            sortida = model.propaga(X_lot)
            precisió += 100 * np.sum(np.argmax(sortida, 0) ==
                                     np.argmax(Y_lot, 0))/Y.shape[1]
            error += f_error(Y_lot, sortida)/len(X_lots)

            model.retropropaga(alfa, d_error, Y_lot, i)
        precisions.append(precisió)
        errors.append(error)
    ax1.plot(precisions, label=label)
    ax2.plot(errors, label=label)


def graf_activacions():
    alfa = 0.001
    f_error = eqm
    d_error = d_eqm

    capes = [
        Aplana(),
        Perceptró(64),
        Sigmoide(),
        Perceptró(10),
        Softmax(),
    ]

    activacions = [
        Sigmoide(),
        Tanh(),
        ReLU(),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for activació in activacions:
        capes[2] = activació
        model = XarxaNeuronal(capes)
        graf_precisió_error(model, alfa, f_error, d_error,
                            ax1, ax2, activació.__class__.__name__)

    ax1.set_title("Precisió (%)")
    ax1.legend()
    ax2.set_title("Error")
    ax2.legend()
    plt.savefig(camí("Gràfica_activacions.svg"), bbox_inches='tight')


def graf_errors():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    labels = ["Error quadràtic mig", "Error absolut mig",
              "Error quadràtic logarítmic mig", "Entropia creuada binària"]
    f_errors = [eqm, eam, eqlm, entropia_creuada]
    d_errors = [d_eqm, d_eam, d_eqlm, d_entropia_creuada]
    alfes = [0.001, 1, 0.001, 0.001]
    for f_error, d_error, label, alfa in zip(f_errors, d_errors, labels, alfes):
        model = XarxaNeuronal([
            Aplana(),
            Perceptró(64, optimitzador='adam'),
            ReLU(),
            Perceptró(10, optimitzador='adam'),
            Softmax(),
        ])
        graf_precisió_error(model, alfa, f_error, d_error, ax1, ax2, label)

    ax1.set_title("Precisió (%)")
    ax1.legend()
    ax2.set_title("Error")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(camí("Gràfica_errors.svg"), bbox_inches='tight')


if __name__ == '__main__':
    # graf_activacions()
    graf_errors()
