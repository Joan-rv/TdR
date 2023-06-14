import IA as ia
import threading
import time
import numpy as np
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
        global w1, b1, w2, b2
        with self.informacio_candau:
            self.informacio = "Llegint dades"
        entrenament_digits, entrenament_imatges, prova_imatges = ia.llegir_dades()
        precisió = 0
        alfa = 0.15
        iter = 0
        while self.entrenant.is_set():
            iter += 1

            z1, a1, z2, a2 = ia.propaga(w1, b1, w2, b2, entrenament_imatges)
            dw1, db1, dw2, db2 = ia.retropropaga(z1, a1, z2, a2, w2, entrenament_digits, entrenament_imatges)

            w1 -= alfa * dw1
            b1 -= alfa * np.reshape(db1, (16,1))
            w2 -= alfa * dw2
            b2 -= alfa * np.reshape(db2, (10, 1))

            precisió = np.sum(np.argmax(a2, 0) == entrenament_digits)/entrenament_digits.size

            with self.informacio_candau:
                self.informacio = f"Iteració: {iter}, precisió: {precisió*100:.2f}%"



    pass

class ProvarPantalla(Screen):
    def prediccio(self):
        global w1, b1, w2, b2
        textura = self.ids.canvas_pintar.export_as_image().texture
        #textura = self.ids.canvas_pintar.texture
        tamany=textura.size
        canvas=textura.pixels
        dibuix = Image.frombytes(mode='RGBA', size=tamany, data=canvas)
        dibuix = dibuix.resize((28,28))
        dibuix = ImageOps.grayscale(dibuix)
        dibuix = ImageOps.invert(dibuix)
        imatge = np.frombuffer(dibuix.tobytes(), dtype=np.uint8).reshape(784, 1)
        imatge = (imatge-np.min(imatge))/(np.max(imatge)-np.min(imatge))

        #entrenament_digits, entrenament_imatges, prova_imatges = ia.llegir_dades()
        #print(entrenament_imatges.shape)
        print(imatge.shape)

        _, _, _, a2 = ia.propaga(w1, b1, w2, b2, imatge)
        ia.imprimeix_imatge(imatge.T[0])
        print(a2)
        print(str(np.argmax(a2, 0)))



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
    w1, w2, b1, b2 = ia.valors_inicials()
    ReconeixementDigitsApp().run()