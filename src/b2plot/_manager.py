# -*- coding: utf-8 -*-
"""
In this file the plot manager lives..

"""

from ._helpers import Singleton

import matplotlib.pyplot as plt


@Singleton
class TheManager:
    def __init__(self):
        self.xaxis = None

    def get_x_axis(self):
        return self.xaxis

    def set_x_axis(self, axis):
        self.xaxis = axis

    def figure(self):
        # f = plt.figure(tight_layout={'pad': 0})
        f = plt.figure()
        self.xaxis = None
        return f


manager = TheManager.Instance()


def xaxis():
    return manager.get_x_axis()


def nf():
    return manager.figure()


def figure():
    return manager.figure()
