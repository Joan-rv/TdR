import IA as ia
from IA import Perceptró, Sigmoide, Softmax
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
    informacio_candau = threading.Lock()
    with informacio_candau:
        informacio = StringProperty("Esperant instruccions")
    text_boto = StringProperty("Inicar entrenament")
    entrenant = BooleanProperty(False)

    def processa_entrenar(self):
        if self.entrenant:
            self.entrenant = False
            self.thread_entrenar.join()
            with self.informacio_candau:
                self.informacio = "Esperant instruccions"
            self.text_boto = "Reprendre entrenament"
        else:
            with self.informacio_candau:
                self.informacio = "Iniciant entrenament"
            self.text_boto = "Parar entrenament"
            self.entrenant = True
            self.thread_entrenar = threading.Thread(target=self.entrenar, daemon=True)
            self.thread_entrenar.start()
    
    def entrenar(self):
        global xarxa, iteracions
        with self.informacio_candau:
            self.informacio = "Llegint dades"
        entrenament_digits, entrenament_imatges, prova_imatges = ia.llegir_dades()

        X = entrenament_imatges
        Y = ia.one_hot(entrenament_digits)

        precisió = 0
        alfa = 0.1
        while self.entrenant:
            iteracions += 1

            sortida = X 
            for capa in xarxa:
                sortida = capa.propaga(sortida)

            precisió = np.sum(np.argmax(sortida, 0) == entrenament_digits)/entrenament_digits.size

            with self.informacio_candau:
                self.informacio = f"Iteració: {iteracions}, precisió: {precisió*100:.2f}%"

            alfa = 0.1
            delta = ia.d_eqm(Y, sortida)
            for capa in reversed(xarxa):
                delta = capa.retropropaga(delta, alfa)

    def guardar_progres(self):
        global xarxa, iteracions
        with open(f"algorisme{xarxa}.pkl", 'wb') as fitxer:
            pickle.dump((xarxa, iteracions), fitxer)
    
    def recuperar_progres(self):
        global xarxa, iteracions
        try:
            with open(f"algorisme{xarxa}.pkl", 'rb') as fitxer:
                xarxa, iteracions = pickle.load(fitxer)
        except FileNotFoundError:
            pass

    pass

class ProvarPantalla(Screen):
    prediccio = StringProperty("Realitza una predicció")

    def predieix(self):
        global xarxa
        textura = self.ids.canvas_pintar.export_as_image().texture
        #textura = self.ids.canvas_pintar.texture
        tamany=textura.size
        canvas=textura.pixels
        imatge = Image.frombytes(mode='RGBA', size=tamany, data=canvas)
        imatge = imatge.resize((28,28))
        imatge = imatge.convert('L')
        imatge = ImageOps.invert(imatge)
        imatge = np.array(imatge).reshape(784, 1)
        #imatge = np.frombuffer(imatge.tobytes(), dtype=np.uint8).reshape(784, 1)
        imatge = imatge / 255.0

        #entrenament_digits, entrenament_imatges, prova_imatges = ia.llegir_dades()
        #print(entrenament_imatges.shape)

        sortida = imatge
        for capa in xarxa:
            sortida = capa.propaga(sortida)
        #ia.imprimeix_imatge(imatge.T[0])
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
                d = 30
                self.ellipses.append(Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
                touch.ud['line'] = Line(points=(touch.x, touch.y), width= d / 2)
                self.lines.append(touch.ud['line'])

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud['line'].points += [touch.x, touch.y]
        else:
            with self.canvas:
                Color(0, 0, 0)
                d = 30
                touch.ud['line'] = Line(points=(touch.x, touch.y), width= d / 2)
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

if __name__ == '__main__':
    xarxa = [
        Perceptró(28**2, 128), 
        Sigmoide(),
        Perceptró(128, 10),
        Softmax(),
    ]
    iteracions = 0
    ReconeixementDigitsApp().run()