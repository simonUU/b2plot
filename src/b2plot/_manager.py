# -*- coding: utf-8 -*-
"""
In this file the plot manager lives..

"""

from _helpers import Singleton
from _style import set_default_style

@Singleton
class TheManager:
    def __init__(self):
        self.xaxis = None
        self.style_set = False

    def get_x_axis(self):
        return self.xaxis

    def set_x_axis(self, axis):
        self.xaxis = axis

    def set_style(self):
        if self.style_set is False:
            set_default_style()
            self.style_set = True

    def new_figure(self):
        self.xaxis = None


manager = TheManager.Instance()


def xaxis():
    return manager.get_x_axis()


def nf():
    manager.new_figure()
