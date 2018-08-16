# -*- coding: utf-8 -*-
"""

"""

from .helpers import manager
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def draw_y_label(label='Entries', unit=None, ha='right', *args, **kwargs):
    """ Plotting scientific notation y label
    :param label:
    :param unit:
    :return:
    """
    x_axis = manager.get_x_axis()
    if unit is None:
        plt.ylabel(label, ha=ha, *args, **kwargs)
    else:
        try:
            width = x_axis[1] - x_axis[0]
        except TypeError:
            plt.ylabel(label+' / ' + unit, ha=ha, *args, **kwargs)
        else:
            plt.ylabel(label+' / %.3f' % width + ' ' + unit, ha=ha, *args, **kwargs)


def watermark(t="2018 (preliminary)", px=0.5, py=0.9, fontsize=16, alpha=0.95,  *args, **kwargs):
    # font = FontProperties()
    # font.set_style('italic')
    # font.set_weight('bold')

    plt.text(px, py, "Belle II", ha='right',
             transform=plt.gca().transAxes,
             fontsize=fontsize,
             style='italic',
             alpha=alpha,  *args, **kwargs
             # fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )
    plt.text(px + 0.02, py, t, ha='left',
             transform=plt.gca().transAxes,
             fontsize=fontsize,
             #          style='italic',
             alpha=alpha,  *args, **kwargs
             #          fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )


def lumi(l="5", px=0.75, py=0.85,):
    plt.text(px, py, "$\smallint L $d$t $=$" + l, transform=plt.gca().transAxes, )


def expand(factor =1.2):
    plt.ylim(0, plt.ylim()[1] * factor)


def set_style():
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4), useMathText=True)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.minorticks_on()
    # plt.tight_layout()

    plt.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.15)


def labels(xlabel=None, ylabel=None, unit=None, root_style=False, *args, **kwargs):

    ha = 'center'
    x,y = .5, .5

    if root_style:
        ha = 'right'
        x,y = 1,1

    if xlabel is not None:
        plt.xlabel(xlabel, horizontalalignment=ha, x=x, *args, **kwargs)

    if unit is not None:
        if unit is not '':
            plt.xlabel(xlabel + ' [' + unit + ']', ha=ha, x=x, *args, **kwargs)
        if ylabel is not None:
            draw_y_label(ylabel, unit,  horizontalalignment=ha, y=y,  *args, **kwargs)
    else:
        if xlabel is not None:
            plt.xlabel(xlabel, horizontalalignment=ha, x=x, *args, **kwargs)
        if ylabel is not None:
            draw_y_label(ylabel,  horizontalalignment=ha, y=y,  *args, **kwargs)

def decorate( *args, **kwargs):
    labels( *args, **kwargs)