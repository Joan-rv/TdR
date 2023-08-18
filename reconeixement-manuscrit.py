import IA as ia
from IA import Perceptró, Aplana, MaxPooling, Convolució, Sigmoide, ReLU, Softmax
import threading
import time
import numpy as np
import pickle
import kivy
from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty, BooleanProperty
from PIL import Image, ImageOps

kivy.require('1.9.0')


class EntrenarPantalla(Screen):
    informacio = StringProperty("Esperant instruccions")
    text_boto = StringProperty("Inicar entrenament")
    entrenant = BooleanProperty(False)
    no_pot_marxar = BooleanProperty(False)

    def processa_entrenar(self):
        self.text_boto = "Aturant entrenament"
        self.informacio = "Aturant entrenament"
        if self.entrenant:
            self.entrenant = False
        else:
            self.informacio = "Iniciant entrenament"
            self.text_boto = "Parar entrenament"
            self.entrenant = True
            self.thread_entrenar = threading.Thread(
                target=self.entrenar, daemon=True)
            self.thread_entrenar.start()

    def entrenar(self):
        global xarxa, iteracions
        self.informacio = "Llegint dades"

        digits, imatges, _ = ia.llegir_dades()
        imatges = imatges.reshape(-1, 28, 28, 1)

        X, X_prova = np.split(imatges, [40000])
        Y, Y_prova = np.split(ia.one_hot(digits), [40000], axis=1)

        temp = np.random.permutation(len(X))
        X = X[temp]
        Y = Y.T[temp].T

        tamany_lots = 100
        X_lots = np.split(X, X.shape[0]/tamany_lots)
        Y_lots = np.split(Y, Y.shape[1]/tamany_lots, axis=1)

        precisió = 0
        alfa = 0.001
        self.informacio = "Iniciant entrenament"
        self.no_pot_marxar = True
        while self.entrenant:
            iteracions += 1

            precisió = 0
            for X_lot, Y_lot in zip(X_lots, Y_lots):
                if not self.entrenant:
                    break
                sortida = xarxa.propaga(X_lot)
                precisió += np.sum(np.argmax(sortida, 0) ==
                                   np.argmax(Y_lot, 0))/Y.shape[1]

                xarxa.retropropaga(alfa, ia.d_eqm, Y_lot, iteracions)

            self.informacio = f"Iteració: {iteracions}, precisió: {precisió*100:.2f}%"
        self.no_pot_marxar = False
        self.informacio = "Esperant instruccions"
        self.text_boto = "Reprendre entrenament"

    def guardar_progres(self):
        thread = threading.Thread(target=self.escriure_progress)
        self.informacio = "Escrivint"
        thread.start()

    def escriure_progress(self):
        global xarxa, iteracions
        with open(f"algorisme{xarxa}.pkl", 'wb') as fitxer:
            pickle.dump((xarxa, iteracions), fitxer)
            self.informacio = "Progrés guardat"

    def recuperar_progres(self):
        thread = threading.Thread(target=self.llegir_progress)
        self.informacio = "Llegint"
        thread.start()

    def llegir_progress(self):
        global xarxa, iteracions
        try:
            print(xarxa)
            with open(f"algorisme{xarxa}.pkl", 'rb') as fitxer:
                xarxa, iteracions = pickle.load(fitxer)
                self.informacio = "Progrés recuperat"
        except FileNotFoundError:
            self.informacio = "Error, no s'ha trobat el fitxer"
            pass

    pass


class ProvarPantalla(Screen):
    prediccio = StringProperty("Realitza una predicció")

    def predieix(self):
        global xarxa
        textura = self.ids.canvas_pintar.export_as_image().texture
        # textura = self.ids.canvas_pintar.texture
        tamany = textura.size
        canvas = textura.pixels
        imatge = Image.frombytes(mode='RGBA', size=tamany, data=canvas)
        imatge = imatge.convert('L')
        imatge = ImageOps.invert(imatge)
        imatge = imatge.crop(imatge.getbbox())
        amplada, altura = imatge.size
        nou_tamany = np.maximum(amplada, altura)
        imatge_nova = Image.new(imatge.mode, (nou_tamany, nou_tamany), (0))
        imatge_nova.paste(imatge, box=(
            (nou_tamany - amplada)//2, (nou_tamany - altura)//2))
        imatge = imatge_nova.resize((20, 20))
        imatge = np.array(imatge)
        imatge = imatge / 255.0
        imatge = np.pad(imatge, 4)
        imatge = imatge.reshape(1, 28, 28, 1)

        sortida = xarxa.propaga(imatge)
        self.prediccio = f"Predicció: {np.argmax(sortida, 0)[0]} | Confiança: {np.max(sortida, 0)[0]*100:.2f}%"
        print(str(np.argmax(sortida, 0)))
        print(str(np.max(sortida, 0)))


class Pintar(Widget):
    def __init__(self, **kwargs):
        super(Pintar, self).__init__(**kwargs)

    ellipses = []
    lines = []

    def on_touch_down(self, touch):
        with self.canvas:
            if self.collide_point(*touch.pos):
                Color(0, 0, 0)
                d = 20
                self.ellipses.append(
                    Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=d / 2)
                self.lines.append(touch.ud['line'])

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud['line'].points += [touch.x, touch.y]
        else:
            with self.canvas:
                Color(0, 0, 0)
                d = 20
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=d / 2)
                self.lines.append(touch.ud['line'])


class MenuPantalla(Screen):
    pass


class ReconeixementDigitsApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuPantalla(name='menu'))
        sm.add_widget(EntrenarPantalla(name='entrenar'))
        sm.add_widget(ProvarPantalla(name='provar'))

        return sm


def main():
    global xarxa
    xarxa = ia.XarxaNeuronal([
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

    global iteracions
    iteracions = 0

    ReconeixementDigitsApp().run()


if __name__ == '__main__':
    main()
