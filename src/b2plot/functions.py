
from ._manager import manager
import matplotlib.pyplot as plt


def draw_y_label(label='Entries', unit=None, fontsize=None, ha='right'):
    """ Plotting scientific notation y label
    :param label:
    :param unit:
    :return:
    """
    x_axis = manager.get_x_axis()
    width = 0.0
    if unit is None:
        unit = ''
    try:
        width = x_axis[1] - x_axis[0]
    except TypeError:
        plt.ylabel(label+' / ' + unit, fontsize=fontsize, ha=ha)
    else:
        plt.ylabel(label+' / %.3f'%width + ' ' + unit, fontsize=fontsize, ha=ha)

def set_style():
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
    ax = plt.gca()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.minorticks_on()


def decorate(xlabel, ylabel=None, title=None, unit=None, titlesize=None, fontsize=None, ha='right'):
    
    set_style()
    
    plt.xlabel(xlabel, horizontalalignment='right', x=1.0, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, horizontalalignment='right', y=1.0, fontsize=fontsize)
    #ax.tick_params( length=7, width=.7, )
    #ax.tick_params(which='minor', length=4, width=.5, )
    
    if unit is not None:
        plt.xlabel(xlabel + ' [' + unit + ']', fontsize=fontsize, ha=ha)
        if ylabel is not None:
            draw_y_label(ylabel, unit, fontsize=fontsize, ha=ha)
    else:
        plt.xlabel(xlabel, fontsize=fontsize, ha=ha)
        if ylabel is not None:
            draw_y_label(ylabel, fontsize=fontsize, ha=ha)
    if title is not None:
        plt.title(title, fontsize=titlesize)