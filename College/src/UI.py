#!/usr/bin/env python
from kivy.config import Config
from kivy.uix.image import AsyncImage
from kivy.uix.image import Image
import sys
import glob
import os.path
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import runTouchApp
from subprocess import Popen
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
Config.set('kivy', 'log_level', 'debug')

kv = """
<Test>:
    ScrollView:
        id: scroll
        GridLayout:
            id: wall
            cols: 5
            size_hint_y:  None
            height: self.minimum_height
"""
Builder.load_string(kv)

class Test(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.callback()

    def callback(self):
        def update_height(img, *args):
            img.height = img.width / img.image_ratio
        dirpath = sys.argv[1]
        files = glob.glob(os.path.join(dirpath, '*.*'))
        btn = Button(text="Test", size_hint_y=None, height=100)
        btn.bind(on_press= lambda x: Popen('python LCDNet.py'))
        self.ids.wall.add_widget(btn)
        for f in files:
            image = AsyncImage(source=f,
                               size_hint=(1, None),
                               keep_ratio=True,
                               allow_stretch=True)
            image.bind(width=update_height, image_ratio=update_height)
            self.ids.wall.add_widget(image)

class TestApp(App):
    def build(self):
        return Test()

if __name__ == '__main__':
    TestApp().run()