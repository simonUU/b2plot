# -*- coding: utf-8 -*-
""" Analysis tools

"""
import b2plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .functions import _hist_init, to_stack, hist, xlim, get_xaxis


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


def mask_append(xs,xb):
    """ Merge xs and xb into one vector and return it with a boolean mask for each category
    """
    return np.append(xs,xb), np.append(np.ones(len(xs)), np.zeros(len(xb)))==1


def purity_hist(x, mask, nbins=10, do_plot=True, figsize=None, xticks_fontsize=None):
    """ Plots the distribution x in an equal frequency binning with the purity regarding mask

    Args:
        x: Distribution or signal distribution
        mask: Boolean mask or background distribution
        nbins:
        do_plot:
        figsize:
        xticks_fontsize:

    Returns:

    """
    if len(pd.unique(mask)) > 2:
        # if signal and background distribution are given as x and mask
        x, mask = mask_append(x, mask)

    x = x[np.isfinite(x)]
    bins = np.percentile(x, np.linspace(0, 100, nbins))

    y_, _ = np.histogram(x, bins)
    y_1, _ = np.histogram(x[mask], bins)
    y_0, _ = np.histogram(x[~mask], bins)

    pur = y_1 / (y_1 + y_0)
    pur_err = (pur * (1 - pur)) / (y_)

    if do_plot:
        x_ = np.arange(len(y_) + 1)
        x_centers = (x_[1:] + x_[:-1]) / 2.

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(x_[:-1], x_, weights=y_1, alpha=0.8)
        ax.hist(x_[:-1], x_, weights=y_0, histtype='step', lw=2)
        ax.hist(x_[:-1], x_, weights=y_, histtype='step', color='grey')

        ax2 = ax.twinx()
        ax2.errorbar(x_centers, pur, np.sqrt(pur_err), color='black', fmt="o--")
        ax2.set_xticklabels([], [])
        ax2.set_xticks([], [])

        ax2.set_ylabel("Purity")
        ax2.set_ylim(0)

        np.append(bins, x.max())
        _ = ax.set_xticks(x_)
        _ = ax.set_xticklabels(['%1.2e' % f for f in bins], rotation=-90, fontfamily='monospace',
                               fontsize=xticks_fontsize)
        plt.xlim(np.min(x_), np.max(x_))
        plt.sca(ax)
    return pur, np.sqrt(pur_err), bins


def flat_bins(x, set=False, nbins=None,  fontsize=None, rotation=-90):
    if nbins is None:
        nbins = len(get_xaxis())-1
    bins = np.percentile(x, np.linspace(0, 100, nbins+1))
    np.append(bins, x.max())
    if set:
        ax = plt.gca()
        x_ = np.linspace(0, 110, len(bins)+1)
        _ = ax.set_xticks(x_)
        _ = ax.set_xticklabels(['%1.2e' % f for f in bins], rotation=rotation, fontfamily='monospace',
                    fontsize=fontsize)
    else:
        return bins, np.linspace(0, 100, len(bins)+1)

def purity_flatness_proba(x, mask, nbins=10, do_plot=False):
    """ Returns the probability that the purity of x[mask] and x[~mask] is flat.

    This can be used as a measure of the information content in this observable regarding the mask.

    Args:
        x: Observable or signal distribution
        mask: Boolean mask, like 'is signal', 'is background' or background distribution
        nbins: Number of bins to calculate the purity
        do_plot (bool): plot the purity distribution

    Returns:
        chi2: probability of CDF(flat) against CDF(purity), where a value of 1 corresponds to the purity being compatible
        with a flat distribution.

    """

    if len(pd.unique(mask)) > 2:
        # if signal and background distribution are given as x and mask
        x, mask = mask_append(x, mask)
    
    if np.std(x)==0:
        return 1

    pur, pure, b = purity_hist(x, mask, nbins=nbins, do_plot=do_plot)

    mask = (np.isfinite(pur)) & (pur != 0)
    p = pur[mask]
    pe = pure[mask] * 100

    cs = np.nancumsum(p) * 100
    cl = np.cumsum(np.full(len(cs), np.mean(p), )) * 100
    chi2 = ((cl - cs) ** 2) / pe ** 2
    try:
        import scipy
        return scipy.stats.distributions.chi2.sf(np.sqrt(np.mean(chi2)), len(chi2))
    except ImportError:
        return chi2


def sig_bkg_plot(df, col, by=None, ax=None, bins=None, range=None, labels=None, normed=False):
    """

    Args:
        df:
        col:
        by:
        ax:
        bins:
        range:
        labels:
        normed:

    Returns:

    """

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

    hist(x_bkg, xaxis, style=0, label=labels[0], ax=ax, density=normed)
    hist(x_sig, xaxis, lw=2, color=0, label=labels[1], ax=ax, density=normed)

    plt.legend()
    xlim()
