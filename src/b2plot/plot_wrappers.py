# -*- coding: utf-8 -*-
"""
In this file all the matplolib wrappers are located.

"""

from ._manager import manager
from ._helpers import get_optimal_bin_size
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt


def _hist_init(data, xaxis, range=None):

    manager.set_style()

    if xaxis is None:
        xaxis = manager.get_x_axis()
        if xaxis is None:
            xaxis = get_optimal_bin_size(len(data))
            _, xaxis = np.histogram(data, xaxis, range=range)
    return xaxis


def text(t, x=0.8, y=0.9, fontsize=22, *args, **kwargs):
    """

    :param t:
    :param x:
    :param y:
    :param fontsize:
    :param args:
    :param kwargs:
    :return:
    """
    plt.text(x, y, t, transform=plt.gca().transAxes, fontsize=fontsize, *args, **kwargs)


def hist(data, xaxis=None, fill=False, range=None,  lw=2, *args, **kwargs):
    """

    :param data:
    :param xaxis:
    :param histtype:
    :param range:
    :param lw:
    :param args:
    :param kwargs:
    :return:
    """

    xaxis = _hist_init(data, xaxis, range=range)

    if type(data) is pd.Series:
        data = data.values

    histtype = 'step'
    if fill:
        histtype = 'stepfilled'

    y, xaxis, _ = plt.hist(data, xaxis, range=range, histtype=histtype, lw=lw, *args, **kwargs)


    manager.set_x_axis(xaxis)


def errorhist(data, xaxis=None, color='black', normed=False, fmt=' ', range=None, 
              xerr=False,
              *args, **kwargs):
    """

    :param data:
    :param xaxis:
    :param color:
    :param normed:
    :param range:
    :param args:
    :param kwargs:
    :return:
    """

    xaxis = _hist_init(data, xaxis, range=range)

    if type(data) is pd.Series:
        data = data.values

    y, x = np.histogram(data, xaxis, normed=normed)

    bin_centers = (x[1:] + x[:-1]) / 2.0
    err = np.sqrt(np.array(y))
    if normed:
        yom, x = np.histogram(data, xaxis,)
        err = np.sqrt(np.array(yom)) *(y/yom)
    if xerr is not False:
        xerr = (x[1]-x[0])/2.0
    else:
        xerr = None
    plt.errorbar(bin_centers, y, yerr=err, xerr=xerr, fmt=fmt, color=color, *args, **kwargs)

    manager.set_x_axis(xaxis)