from model import xarxa

import pickle
from PIL import Image, ImageOps
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ColorProperty

from kivy.config import Config
Config.set('graphics', 'width', '1300')


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
            touch.ud['line'].points += [touch.x, touch.y]
        else:
            with self.canvas:
                Color(0, 0, 0)
                d = 1/14 * self.size[1]
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=d / 2)
                self.lines.append(touch.ud['line'])


class CalculadoraApp(App):
    def build(self):
        pass

    def addition(self):
        self.prediu()

    def prediu(self):
        global xarxa
        textura = self.root.ids.canvas_pintar.export_as_image().texture
        # textura = self.ids.canvas_pintar.texture
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
        print(num)


def main():
    global xarxa
    with open(f"algorisme{xarxa}.pkl", 'rb') as fitxer:
        xarxa, _ = pickle.load(fitxer)
    CalculadoraApp().run()


if __name__ == "__main__":
    main()
