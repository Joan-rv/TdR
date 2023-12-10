from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ColorProperty

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