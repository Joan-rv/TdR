from ia.xarxa_neuronal import XarxaNeuronal
from ia.capes import Perceptró
from ia.activacions import Sigmoide
from ia.errors import d_eqm
from ia.utils import llegir_dades
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.signal import convolve2d
import os


def camí(nom):
    return os.path.join("grafs", nom)


def prepara_graf_2d():
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', linewidth=.5)
    ax.axvline(0, color='black', linewidth=.5)
    ax.set_aspect('equal')
    ax.margins(x=0)
    ax.grid(True, which='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    return fig, ax


def sigmoide(x):
    return 1/(1 + np.exp(-x))


fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
dim_fig = 3
x = np.arange(dim_fig**2).reshape(dim_fig, dim_fig)
ax1.imshow(x, cmap='plasma')
ax2.imshow(x.reshape(1, dim_fig**2), cmap='plasma')
ax1.axis("off")
ax1.set_title("$X$")
ax2.axis("off")
ax2.set_title("$A$")
plt.savefig(camí("Figura_aplanar.svg"), bbox_inches='tight')


fig, (ax1, ax2) = plt.subplots(1, 2)
_, imatges, _ = llegir_dades()
imatges = imatges.reshape(-1, 28, 28)
ax1.imshow(imatges[0], cmap="Greys_r")
nucli = np.array([[1, 0, -1],
                 [2, 0, -2],
                 [1, 0, -1]])
convolucionat = convolve2d(imatges[0], nucli)
ax2.imshow(convolucionat, cmap="Greys_r")
plt.savefig(camí("Figura_exemple_conv.svg"), bbox_inches='tight')


# Generar 100 mostres entre -2 i 2
x = np.linspace(-2, 2, 100)
fig, ax = prepara_graf_2d()
ax.plot(x, np.maximum(0, x), label='Sigmoide')
ax.set_aspect('equal')


plt.savefig(camí('Gràfica_relu.svg'), bbox_inches='tight')

fig, ax = prepara_graf_2d()
ax.plot(x, sigmoide(x), label='Sigmoide')
ax.plot(x, np.tanh(x), label='Tanh')

plt.legend()
plt.savefig(camí('Gràfica_sig_tanh.svg'), bbox_inches='tight')

xarxa = XarxaNeuronal([
    Perceptró(1),
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

fig, ax = prepara_graf_2d()
ax.plot(x.T, xarxa.propaga(x).T)


# Barres de color per indicar activació o no
ax.axhspan(0, ax.get_ylim()[1], facecolor='g', alpha=0.1)
ax.axhspan(0, ax.get_ylim()[0], facecolor='r', alpha=0.1)

ax.set_xlabel("Entrada")
ax.set_ylabel("Sortida")

plt.savefig(camí('Gràfica_recta_not.svg'), bbox_inches='tight')


def graf_3d_model(model, nom_fitxer_sortida):

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

    plt.savefig(camí(nom_fitxer_sortida), bbox_inches='tight')


xarxa = XarxaNeuronal([
    Perceptró(2),
    Sigmoide(),
    Perceptró(1),
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

graf_3d_model(xarxa, 'Gràfica_xor.svg')

xarxa = XarxaNeuronal([
    Perceptró(1),
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
    print(f"Iteració: {i}; precisió: {precisió_entrenament*100:.2f}%")

graf_3d_model(xarxa, 'Gràfica_pla_and.svg')


def graf_descens_grad():
    def f(x1, x2):
        return (1/2)*x1**2 + (5/3)*x2**2 - x1*x2 - 2*(x1 + x2)

    def grad(x1, x2):
        return np.array([x1 - x2 - 2, (10/3)*x2 - x1 - 2])

    def magnitud(x):
        # Fer pitàgores per obtenir magnitud
        return np.sqrt(np.sum(np.square(x)))

    x1, x2, = -2, -3/2
    x1s = [x1]
    x2s = [x2]
    grad_f = grad(x1, x2)

    while 50 > magnitud(grad_f) > 1e-6:
        x1 -= alfa * grad_f[0]
        x2 -= alfa * grad_f[1]
        x1s.append(x1)
        x2s.append(x2)
        grad_f = grad(x1, x2)

    x1 = np.linspace(min(x1s)-1, max(x1s)+1, 200)
    x2 = np.linspace(min(x2s)-1, max(x2s)+1, 200)
    X, Y = np.meshgrid(x1, x2)
    Z = f(X, Y)

    fig = plt.figure()

    plt.imshow(Z, extent=[min(x1s)-1, max(x1s)+1, min(x2s) -
                          1, max(x2s)+1], origin='lower', cmap='jet')
    plt.plot(x1s, x2s)
    plt.plot(x1s, x2s, '*')
    plt.colorbar()


alfa = 0.2
graf_descens_grad()
plt.savefig(camí("Gràfica_descens_grad.svg"), bbox_inches='tight')

alfa = 1
graf_descens_grad()
plt.savefig(camí("Gràfica_salts_grad.svg"), bbox_inches='tight')

alfa = 0.001
graf_descens_grad()
plt.savefig(camí("Gràfica_descens_lent.svg"), bbox_inches='tight')
