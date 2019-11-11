# -*- coding: utf-8 -*-
""" Analysis tools

"""
import b2plot
import numpy as np
import matplotlib.pyplot as plt
from .functions import _hist_init


def plot_flatness(sig, tag, bins=None, ax=None, xrange=None, percent_step=5):
    """ Plotting differences of sig distribution in percentiles of tag distribution

    Args:
        sig:
        tag:
        bins:
        ax:
        xrange:
        percent_step:

    """



    if ax is None:
        fix, ax = plt.subplots()

    xaxis = _hist_init(sig, bins=bins, xrange=xrange)

    colormap = plt.get_cmap('magma')
    orig, x = np.histogram(sig, bins=xaxis, range=xrange, normed=True, )
    bin_center = ((x + np.roll(x, 1)) / 2)[1:]
    tmp = orig/orig
    ax.plot(bin_center, tmp, color='black', lw=1)
    for quantil in np.arange(5, 100, percent_step):
        cut = np.percentile(tag, quantil)
        sel = tag >= cut
        y, x = np.histogram(sig[sel], bins=x, range=xrange, normed=True, )
        y /= orig
        ax.fill_between(bin_center, tmp, y, color=colormap(quantil/100.0))
        tmp = y


def profile(x, y, bins=None, range=None, fmt='.', *args, **kwargs):
    import scipy

    xaxis = _hist_init(x, bins, xrange=range)

    means = scipy.stats.binned_statistic(x, y, bins=xaxis, statistic='mean').statistic
    std = scipy.stats.binned_statistic(x, y, bins=xaxis, statistic=scipy.stats.sem).statistic

    bin_centers = (xaxis[:-1] + xaxis[1:]) / 2.
    plt.errorbar(x=bin_centers, y=means, yerr=std, linestyle='none', fmt=fmt, *args, **kwargs)


def ratio(y1, y2, y1_err=None, y2_err= None):
    """ calculate the ratio between two histograms y1/y2

    Args:
        y1: y values of first histogram
        y2: y values of second histogram
        y1_err: (optional) error of first
        y2_err: (optional) error of second

    Returns:
        ratio, ratio_error

    """
    assert len(y1) == len(y2), "y1 and y2 length does not match"
    y1e = np.sqrt(y1) if y1_err is None else y1_err
    y2e = np.sqrt(y2) if y2_err is None else y2_err
    r = y1/y2
    re = np.sqrt((y1/(1.0*y2*y2))*(y1/(1.0*y2*y2))*y2e*y2e+(1/(1.0*y2))*(1/(1.0*y2))*y1e*y1e)
    return r, re


def data_mc_ratio(data, mc, label_data='Data', label_mc="MC",
                  y_label=None, figsize=None, ratio_range=(0, 2),
                  *args, **kwarg):
    f, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, figsize=figsize)
    ax0 = axes[0]

    hm = b2plot.hist(mc, lw=2, ax=ax0, label=label_mc, *args, **kwarg)
    hd = b2plot.errorhist(data, ax=ax0, label=label_data, color='black')
    ax0.legend()

    ax1 = axes[1]
    ry, rye = ratio(hd[0], hm[0])
    b2plot.errorbar(hd[1], ry, rye, ax=ax1, color='grey')
    ax1.axhline(1, color='grey', lw=0.5, ls='--')
    f.subplots_adjust(hspace=0.1)

    ax1.set_ylim(*ratio_range)
    b2plot.xlim()
    if y_label is not None:
        ax0.set_ylabel(y_label)
        ax1.set_ylabel("Ratio")
        ax1.yaxis.set_label_coords(-0.08, 0.5)
        ax0.yaxis.set_label_coords(-0.08, 0.5)


def purity_hist(x, mask, nbins=10, ax=None):
    """ Plots the distribution x in an equal frequency binning with the purity regarding mask

    Args:
        x: Distribution
        mask: boolean list for signal and background flags
        nbins: number of bins for the histogram

    Returns:

    """

    x = x[np.isfinite(x)]
    bins = np.percentile(x, np.linspace(0, 100, nbins))

    y_, _ = np.histogram(x, bins)
    y_1, _ = np.histogram(x[mask], bins)
    y_0, _ = np.histogram(x[~mask], bins)

    pur = y_1 / (y_1 + y_0)
    pur_err = (pur * (1 - pur)) / (y_)

    x_ = np.arange(len(y_) + 1)
    x_centers = (x_[1:] + x_[:-1]) / 2.

    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(x_[:-1], y_, width=np.diff(x_), ec="k", align="edge")
    ax.bar(x_[:-1], y_0, width=np.diff(x_), ec="k", align="edge", color='white')

    ax2 = ax.twinx()
    ax2.errorbar(x_centers, pur, np.sqrt(pur_err), color='black', fmt="o--")
    ax2.set_xticklabels([], [])
    ax2.set_xticks([], [])

    ax2.set_ylabel("Purity")
    ax2.set_ylim(0)
    ax.bar(x_[:-1], y_1, width=np.diff(x_), ec="k", align="edge")

    np.append(bins, x.max())
    _ = ax.set_xticks(x_)
    _ = ax.set_xticklabels(['%3.3e' % f for f in bins], rotation=90, fontfamily='monospace')