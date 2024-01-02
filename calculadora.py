from model import xarxa

import pickle
from PIL import Image, ImageOps
import numpy as np
from math import sqrt, sin, cos, tan, pi

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ColorProperty, StringProperty

from kivy.config import Config
Config.set('graphics', 'width', '1400')


class Pintar(Widget):
    background_color = ColorProperty((1., 1., 1., 1.))

    def __init__(self, **kwargs):
        super(Pintar, self).__init__(**kwargs)
        with self.canvas.before:
            self.background_color
            Rectangle(size=self.size)

    def on_background_color(self, instance, value):
        print(self.background_color)
        with self.canvas.before:
            Color(rgba=self.background_color)
            Rectangle(size=self.size)

    ellipses = []
    lines = []

    def on_touch_down(self, touch):
        with self.canvas:
            if self.collide_point(*touch.pos):
                Color(0, 0, 0)
                d = 1/14 * self.size[1]
                self.ellipses.append(
                    Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=d / 2)
                self.lines.append(touch.ud['line'])

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            try:
                touch.ud['line'].points += [touch.x, touch.y]
            except KeyError:
                self.on_touch_down(touch)
        else:
            with self.canvas:
                Color(0, 0, 0)
                d = 1/14 * self.size[1]
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=d / 2)
                self.lines.append(touch.ud['line'])


class CalculadoraApp(App):
    tokens = []
    text = StringProperty("Resultat")
    error = False

    def build(self):
        pass

    def ac(self):
        self.tokens = []
        self.root.ids.canvas_pintar.canvas.clear()

    def delete(self):
        if len(self.tokens) != 0:
            self.root.ids.canvas_pintar.canvas.clear()
            self.tokens.pop()

    def suma(self):
        self.tokens.append("+")
        self.processa_op()

    def resta(self):
        self.tokens.append("-")
        self.processa_op()

    def multiplica(self):
        self.tokens.append("*")
        self.processa_op()

    def divideix(self):
        self.tokens.append("/")
        self.processa_op()

    def elevar(self):
        self.tokens.append("^")
        self.processa_op()

    def arrel(self):
        self.tokens.append("sqrt(")
        self.processa_op()

    def obre_paren(self):
        self.tokens.append("(")
        self.processa_op()

    def tanca_paren(self):
        self.tokens.append(")")
        self.processa_op()

    def sin(self):
        self.tokens.append("sin(")
        self.processa_op()

    def cos(self):
        self.tokens.append("cos(")
        self.processa_op()

    def tan(self):
        self.tokens.append("tan(")
        self.processa_op()

    def pi(self):
        self.tokens.append("pi")
        self.processa_op()

    def igual(self):
        try:
            text = ''.join(self.tokens)
            text = text.replace("^", "**")
            print(text)
            resultat = eval(text)
            self.tokens = [str(resultat)]
            self.processa_op()
        except SyntaxError:
            self.error = True
            self.text = "Error de sintaxi"
            self.tokens = []
            self.root.ids.canvas_pintar.canvas.clear()

    def processa_op(self):
        self.root.ids.canvas_pintar.canvas.clear()

    def prediu(self):
        global xarxa
        textura = self.root.ids.canvas_pintar.export_as_image().texture
        tamany = textura.size
        canvas = textura.pixels
        imatge = Image.frombytes(mode='RGBA', size=tamany, data=canvas)
        imatges = []
        np_imatges = np.array(imatge)
        for n in range(0, 1200, 200):
            np_imatge = np_imatges[:, n:(n+190), :]
            if np.all(np_imatge == 255):
                continue
            imatges.append(Image.fromarray(np_imatge))
        np_imatges = []
        for imatge in imatges:
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
            np_imatges.append(imatge)

        num = ""
        for imatge in np_imatges:
            sortida = xarxa.propaga(imatge)
            num += str(np.argmax(sortida, 0)[0])
        if num == "":
            return None
        return num

    def imprimeix(self):
        if self.error:
            self.error = False
            return
        if self.prediu() == None:
            pass
        elif len(self.tokens) == 0:
            self.tokens.append(self.prediu())
        elif self.tokens[-1].isdigit():
            self.tokens[-1] = self.prediu()
        else:
            self.tokens.append(self.prediu())
        self.text = "".join(self.tokens)


def main():
    global xarxa
    with open(f"algorisme{xarxa}.pkl", 'rb') as fitxer:
        xarxa, _ = pickle.load(fitxer)
    CalculadoraApp().run()


if __name__ == "__main__":
    main()
