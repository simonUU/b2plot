 # -*- coding: utf-8 -*-
"""
In this file all the matplolib wrappers are located.

"""

from .helpers import get_optimal_bin_size, TheManager
from .colors import b2cm
import pandas as pd
import numpy as np
from matplotlib.colors import hex2color

import matplotlib.pyplot as plt


def _hist_init(data, bins=None, xrange=None):
    """ Performs and stores or returns the binning

    Args:
        data:
        bins:
        xrange:

    Returns:

    """
    xaxis = TheManager.Instance().get_x_axis()

    if xaxis is None or bins is not None or xrange is not None:
        if bins is None:
            bins = get_optimal_bin_size(len(data))
        _, xaxis = np.histogram(data, bins, xrange)

    return xaxis


def set_xaxis(bins, flat=False):
    TheManager.Instance().set_x_axis(bins)


def get_xaxis():
    return TheManager.Instance().get_x_axis()


def flat_x(x, nbins=25):
    set_xaxis(np.percentile(x, np.linspace(0, 100, nbins)))


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


def xlim(low=None, high=None, ax=None):
    """

    Args:
        low:
        high:
        ax:

    Returns:

    """
    xaxis = TheManager.Instance().get_x_axis()
    if xaxis is not None:
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(np.min(xaxis), np.max(xaxis))
    if low is not None or high is not None:
        ax.set_xlim(low, high)


def save(filename,  *args, **kwargs):
    """ Save a file and do the subplot_adjust to fit the page with larger labels

    Args:
        filename:
        *args:
        **kwargs:

    Returns:

    """
    plt.savefig(filename, bbox_inches='tight', *args, **kwargs)


def save_adjust(filename, bottom=0.15, left=0.13, right=0.96, top=0.95, *args, **kwargs):
    """ Save a file and do the subplot_adjust to fit the page with larger labels

    Args:
        filename:
        bottom:
        left:
        right:
        top:
        *args:
        **kwargs:
bbox_inches='tight',
    Returns:

    """
    plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    plt.savefig(filename,  *args, **kwargs)


