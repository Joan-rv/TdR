from IA import XarxaNeuronal, Perceptró, Sigmoide, ReLU, d_eqm
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np


def eixos(ax):
    ax.margins(x=0)
    ax.set_xlim()
    ax.grid(True, which='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)


def sigmoide(x):
    return 1/(1 + np.exp(-x))


x = np.linspace(-2, 2, 100)
fig, ax = plt.subplots()
ax.axhline(0, color='black', linewidth=.5)
ax.axvline(0, color='black', linewidth=.5)
ax.plot(x, np.maximum(0, x), label='Sigmoide')
ax.set_aspect('equal')

eixos(ax)

plt.savefig('Gràfica_relu.svg', bbox_inches='tight')

fig, ax = plt.subplots()
ax.axhline(0, color='black', linewidth=.5)
ax.axvline(0, color='black', linewidth=.5)
ax.plot(x, sigmoide(x), label='Sigmoide')
ax.plot(x, np.tanh(x), label='Tanh')
ax.set_aspect('equal')
eixos(ax)

plt.legend()
plt.savefig('Gràfica_sig_tanh.svg', bbox_inches='tight')

xarxa = XarxaNeuronal([
    Perceptró(1, 1),
])

X = np.array([[0], [1]]).T
Y = np.array([[1], [-1]]).T

alfa = 0.1
iteracions = 1000
i = 0
precisió_entrenament = 0
while precisió_entrenament < 0.99:
    i += 1
    sortida = xarxa.propaga(X)
    xarxa.retropropaga(alfa, d_eqm, Y, i)
    precisió_entrenament = np.sum(2*(sortida > 0) - 1 == Y)/Y.shape[1]
    if i % 100 == 0:
        print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")

print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")


x = np.linspace(-1, 1, 100).reshape((1, 100))

fig, ax = plt.subplots()
ax.axhline(0, color='black', linewidth=.5)
ax.axvline(0, color='black', linewidth=.5)
ax.plot(x.T, xarxa.propaga(x).T)

eixos(ax)

ax.axhspan(0, ax.get_ylim()[1], facecolor='g', alpha=0.1)
ax.axhspan(0, ax.get_ylim()[0], facecolor='r', alpha=0.1)

ax.set_xlabel("Entrada")
ax.set_ylabel("Sortida")

plt.savefig('Gràfica_recta_not.svg', bbox_inches='tight')


def graficà_3d_model(model, nom_fitxer_sortida):

    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)

    X, Y = np.meshgrid(x, y)
    xs = X.flatten()
    ys = Y.flatten()

    entrada = np.column_stack((xs, ys)).T

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    Z = model.propaga(entrada)
    Z = Z.reshape((101, 101))
    norm = colors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=0.5)
    ax.plot_surface(X, Y, Z, cmap=colors.LinearSegmentedColormap.from_list(
        'rg', ["r", "g"], N=2), rstride=2, cstride=2, norm=norm)
    Z = np.zeros_like(Z)
    ax.plot_surface(X, Y, Z, alpha=0.2)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")

    plt.savefig(nom_fitxer_sortida, bbox_inches='tight')


xarxa = XarxaNeuronal([
    Perceptró(2, 2),
    Sigmoide(),
    Perceptró(2, 1),
    # Escalonada(),
])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[-1], [1], [1], [-1]]).T

alfa = 0.1
iteracions = 1000
i = 0
precisió_entrenament = 0
while precisió_entrenament < 0.99:
    i += 1
    sortida = xarxa.propaga(X)
    xarxa.retropropaga(alfa, d_eqm, Y, i)
    precisió_entrenament = np.sum(2*(sortida > 0) - 1 == Y)/Y.shape[1]
    if i % 100 == 0:
        print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")

print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")

graficà_3d_model(xarxa, 'Gràfica_xor.svg')

xarxa = XarxaNeuronal([
    Perceptró(2, 1),
])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[-1], [-1], [-1], [1]]).T

alfa = 0.1
iteracions = 1000
i = 0
precisió_entrenament = 0
while precisió_entrenament < 0.99:
    i += 1
    sortida = xarxa.propaga(X)
    xarxa.retropropaga(alfa, d_eqm, Y, i)
    precisió_entrenament = np.sum(2*(sortida > 0) - 1 == Y)/Y.shape[1]
    if i % 10 == 0:
        print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")

graficà_3d_model(xarxa, 'Gràfica_pla_and.svg')
