# -*- coding: utf-8 -*-
"""
In this file all the matplolib wrappers are located.

"""

from _manager import manager
from _helpers import get_optimal_bin_size
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


def hist(data, xaxis=None, histtype='step', range=None,  lw=2, *args, **kwargs):
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

    y, xaxis, _ = plt.hist(data, xaxis, range=range, histtype=histtype, lw=lw, *args, **kwargs)

    manager.set_x_axis(xaxis)


def errorhist(data, xaxis=None, color='black', normed=False, fmt=' ', range=None, *args, **kwargs):
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
    xerr = (x[1]-x[0])/2.0

    plt.errorbar(bin_centers, y, yerr=err, xerr=xerr, fmt=fmt, color=color, *args, **kwargs)

    manager.set_x_axis(xaxis)