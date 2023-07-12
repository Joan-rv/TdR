import IA as ia
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
from kivy.properties import StringProperty
from PIL import Image, ImageOps

kivy.require('1.9.0')

class EntrenarPantalla(Screen):
    informacio_candau = threading.Lock()
    with informacio_candau:
        informacio = StringProperty("Esperant instruccions")
    text_boto = StringProperty("Inicar entrenament")
    entrenant = threading.Event()
    entrenant.clear()

    def processa_entrenar(self):
        if self.entrenant.is_set():
            self.entrenant.clear()
            self.thread_entrenar.join()
            with self.informacio_candau:
                self.informacio = "Esperant instruccions"
            self.text_boto = "Reprendre entrenament"
        else:
            with self.informacio_candau:
                self.informacio = "Iniciant entrenament"
            self.text_boto = "Parar entrenament"
            self.entrenant.set()
            self.thread_entrenar = threading.Thread(target=self.entrenar, daemon=True)
            self.thread_entrenar.start()
    
    def entrenar(self):
        global W_l, b_l, iter
        with self.informacio_candau:
            self.informacio = "Llegint dades"
        entrenament_digits, entrenament_imatges, prova_imatges = ia.llegir_dades()
        precisió = 0
        alfa = 0.1
        while self.entrenant.is_set():
            iter += 1

            Z_l, A_l = ia.propaga(W_l, b_l, entrenament_imatges)
            dW_l, db_l = ia.retropropaga(Z_l, A_l, W_l, b_l, entrenament_digits, entrenament_imatges)

            W_l, b_l = ia.actualitza_paràmetres(W_l, b_l, dW_l, db_l, alfa)

            precisió = np.sum(np.argmax(A_l[-1], 0) == entrenament_digits)/entrenament_digits.size

            with self.informacio_candau:
                self.informacio = f"Iteració: {iter}, precisió: {precisió*100:.2f}%"

    def guardar_progres(self):
        global W_l, b_l, iter, estructura
        with open(f"algorisme{estructura}.pkl", 'wb') as fitxer:
            pickle.dump((W_l, b_l, iter), fitxer)
    
    def recuperar_progres(self):
        global W_l, b_l, iter, estructura
        try:
            with open(f"algorisme{estructura}.pkl", 'rb') as fitxer:
                W_l, b_l, iter = pickle.load(fitxer)
        except FileNotFoundError:
            pass

    pass

class ProvarPantalla(Screen):
    prediccio = StringProperty("Realitza una predicció")

    def predieix(self):
        global W_l, b_l
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

        _, A_l = ia.propaga(W_l, b_l, imatge)
        #ia.imprimeix_imatge(imatge.T[0])
        self.prediccio = f"Predicció: {np.argmax(A_l[-1], 0)[0]} | Confiança: {np.max(A_l[-1], 0)[0]*100:.2f}%"
        print(str(np.argmax(A_l[-1], 0)))
        print(str(np.max(A_l[-1], 0)))



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
    estructura = [784, 256, 128, 64, 10]
    W_l, b_l = ia.valors_inicials(estructura)
    iter = 0
    ReconeixementDigitsApp().run()