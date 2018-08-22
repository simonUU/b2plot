 # -*- coding: utf-8 -*-
"""
In this file all the matplolib wrappers are located.

"""

from .helpers import get_optimal_bin_size, manager
from .colors import b2cm
import pandas as pd
import numpy as np
from matplotlib.colors import hex2color

import matplotlib.pyplot as plt


def _hist_init(data, bins=None, xrange=None):
    xaxis = manager.get_x_axis()
    if xaxis is None or bins is not None or xrange is not None:
        if bins is None:
            bins = get_optimal_bin_size(len(data))
        _, xaxis = np.histogram(data, bins, xrange)

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


STYLES_facecolor = [None, 'none', 'none', 'none', 'none', 'none']
STYLES_hatches = [None, '///', r"\\\ ", ".+", 'xxx', '--', '++', 'xx', '//', '*', 'o', 'O', '.']


def hist(data, bins=None, fill=False, range=None, lw=1., ax=None, style=None, color=None, scale=None, weights=None,
         label=None,  *args, **kwargs):
    """

    Args:
        data:
        bins:
        fill:
        range:
        lw:
        ax:
        style:
        color:
        scale:
        weights:
        *args:
        **kwargs:

    Returns:

    """


    if ax is None:
        ax = plt.gca()

    xaxis = _hist_init(data, bins, xrange=range)

    if type(data) is pd.Series:
        data = data.values

    if isinstance(color, int):
        color = b2cm[color % len(b2cm)]

    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    # convert color
    if not isinstance(color, list) or isinstance(color, tuple):
        color = hex2color(color)

    if style is not None:
        fill = True
    else:
        style = 0

    if weights is None:
        weights = np.ones(len(data))

    if scale is not None:
        if isinstance(scale, int) or isinstance(scale, float):
            if not isinstance(scale, bool):
                weights *= scale
        else:
            print("Please provide int or float with scale")

    if fill:
        fc = (*color, 0.5) if style == 0 else 'none'
        # y, xaxis, _ = ax.hist(data, xaxis, range=range, histtype='step',
        #                       lw=lw, color=color, weights=weights, *args, **kwargs)
        y, xaxis, patches = ax.hist(data, xaxis, range=range, lw=lw, histtype='stepfilled', hatch=STYLES_hatches[style],
                                    edgecolor=color, facecolor=fc, linewidth=lw, weights=weights, label=label,
                                    color=color, *args, **kwargs)
    else:
        y, xaxis, patches = ax.hist(data, xaxis, range=range, histtype='step', lw=lw, color=color, weights=weights,
                                    label=label, *args, **kwargs)

    manager.set_x_axis(xaxis)
    return y, xaxis, patches


def to_stack(df, col, by):

    g = df.groupby(by)

    x_data = []
    for gr in g.groups:
        x_data.append(g.get_group(gr)[col].values)
    return x_data


def stacked(df, col=None, by=None, bins=None, color=None, range=None, lw=.5, ax=None, edgecolor='black',
            *args, **kwargs):
    """ Create stacked histogram

    Args:
        df (DataFrame or list of arrays):
        col:
        by:
        bins:
        color:
        lw:
        *args:
        **kwargs:

    Returns:

    """

    if isinstance(df, pd.DataFrame):
        assert col is not None, "Please provide column"
        assert by is not None, "Please provide by"

        data = to_stack(df, col, by)

    else:
        assert isinstance(df, list), "Please provide DataFrame or List"
        data = df

    if ax is None:
        ax = plt.gca()

    if color is None:
        from b2plot.colors import b2helix
        n_stacks = len(data)
        if n_stacks < 20:
            color = b2helix(n_stacks)

    xaxis = _hist_init(data[0], bins, xrange=range)

    y, xaxis, _ = ax.hist(data, xaxis, histtype='stepfilled',
                          lw=lw, color=color, edgecolor=edgecolor, stacked=True, *args, **kwargs)

    manager.set_x_axis(xaxis)



def errorbar(data, bins=None, color=None, normed=False, fmt='.', range=None, scale=None,
              xerr=False, box=False, ax=None, weights=None, plot_zero=False, *args, **kwargs):
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

    xaxis = _hist_init(data, bins, xrange=range)

    if ax is None:
        ax = plt.gca()

    if type(data) is pd.Series:
        data = data.values

    if weights is None:
        weights = np.ones(len(data))

    if scale is not None:
        if isinstance(scale, int) or isinstance(scale, float):
            if not isinstance(scale, bool):
                weights *= scale
        else:
            print("Please provide int or float with scale")

    y, x = np.histogram(data, xaxis, normed=normed, weights=weights)

    if isinstance(color, int):
        color = b2cm[color % len(b2cm)]

    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    bin_centers = (x[1:] + x[:-1]) / 2.0
    # https://www-cdf.fnal.gov/physics/statistics
    err = (-0.5 + np.sqrt(np.array(y + 0.25)), +0.5 + np.sqrt(np.array(y + 0.25)))  # np.sqrt(np.array(y))
    if normed:
        yom, x = np.histogram(data, xaxis, weights=weights)
        err = (np.sqrt(np.array(yom)) *(y/yom), np.sqrt(np.array(yom)) * (y/yom))
    if xerr is not False:
        xerr = (x[1]-x[0])/2.0
    else:
        xerr = None

    toplot = np.ones(len(y)).astype(bool)

    if plot_zero is False:

        toplot[y == 0] = False
        err = (err[0][[toplot]], err[1][toplot])
        if xerr is not None:
            xerr = xerr[toplot]

    if box:
        xerr = (x[:-1] - x[1:]) / 2.0
        xerr = xerr[toplot]
        hi = err[0] + err[1]
        lo = y[toplot] - err[0]
        plt.errorbar(bin_centers[toplot], y[toplot], color=color, xerr=xerr, fmt=' ')
        plt.bar(bin_centers[toplot], hi, bottom=lo, align='center', color=color, alpha=.7,
                width=2 * xerr,
                edgecolor=color, *args, **kwargs)
    else:
        plt.errorbar(bin_centers[toplot], y[toplot], yerr=err, xerr=xerr, fmt=fmt, color=color, *args, **kwargs)

    manager.set_x_axis(xaxis)




def xlim(low=None, high=None, ax=None):
    """

    Args:
        low:
        high:
        ax:

    Returns:

    """
    xaxis = manager.get_x_axis()
    if xaxis is not None:
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(np.min(xaxis), np.max(xaxis))
    if low is not None or high is not None:
        ax.set_xlim(low, high)


def save(filename, bottom=0.15, left=0.13, right=0.96, top=0.95, *args, **kwargs):
    """ Save a file and do the subplot_adjust to fit the page with larger labels

    Args:
        filename:
        bottom:
        left:
        right:
        top:
        *args:
        **kwargs:

    Returns:

    """
    plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    plt.savefig(filename, *args, **kwargs)


def sig_bkg_plot(df, col, by=None, ax=None, bins=None, range=None, labels=None):

    # foreseen usage
    if isinstance(df, pd.DataFrame):
        # by is not a boolean index
        if isinstance(by, str):
            x = to_stack(df, col, by)
            if len(x) > 2 :
                print("Waring, more than two categories in %s!" % by)
                assert len(x) > 1, "Did not found any categories in %s!" % by
            x_sig = x[1]
            x_bkg = x[0]
        # by is a boolean index
        else:
           x_sig = df[col][by].values
           x_bkg = df[col][~by].values
    # Alternative usage, passing two arrays
    else:
        x_sig = df
        x_bkg = col

    xaxis = _hist_init(np.append(x_sig, x_bkg), bins, xrange=range)

    if labels is None:
        labels = ["Background", "Signal"]

    hist(x_bkg, xaxis, style=0, label=labels[0], ax=ax)
    hist(x_sig, xaxis, lw=2, color=0, label=labels[1], ax=ax)

    plt.legend()
    xlim()
