# -*- coding: utf-8 -*-
""" Simple fit to a gaussian distribution

In this example a fit is performed to a simple gaussion distribution.

Observables can be initialised by a list with the column name / variable name as first argument, followed
by the range and/or with the initial value and range:
x = ('x', -3, 3)
x = ('mass', -3, 0.02, 3)

Parameters are initialised with a tuple: sigma=(0,1) or again including a starting parameter: sigma=(0.01, 0, 1)
The order here is not important.

All parameters and observables can also be initialised by a ROOT.RooRealVar.

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_flatness(sig, tag, ax=None, percent_step=5):
    """ Plotting differences of sig distribution in percentiles of tag distribution

    :param sig:
    :param tag:
    :param ax:
    :param percent_step:
    :return:
    """

    if ax is None:
        fix, ax = plt.subplots()

    colormap = plt.get_cmap('magma')
    orig, x = np.histogram(sig, bins=100, range=xrange, normed=True, )
    bin_center = ((x + np.roll(x, 1)) / 2)[1:]
    tmp = orig/orig
    ax.plot(bin_center, tmp, color='black', lw=1)
    for quantil in np.arange(5, 100, percent_step):
        cut = np.percentile(tag, quantil)
        sel = tag >= cut
        y, x = np.histogram(sig[sel], bins=100, range=xrange, normed=True, )
        y /= orig
        ax.fill_between(bin_center, tmp, y, color=colormap(quantil/100.0))
        tmp = y
