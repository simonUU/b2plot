
from _manager import manager
import matplotlib.pyplot as plt


def draw_y_label(label='Entries', unit=None, fontsize=None):
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
        plt.ylabel(label+' /  ' + unit, fontsize=fontsize)
    else:
        plt.ylabel(label+' /  %.3f'%width + ' ' + unit, fontsize=fontsize)


def decorate(xlabel, ylabel=None, title=None, unit=None, titlesize=None, fontsize=None):
    if unit is not None:
        if fontsize is None:
            plt.xlabel(xlabel + ' [' + unit + '] ',)
        else:
             plt.xlabel(xlabel + ' [' + unit + '] ', fontsize=fontsize)
        if ylabel is not None:
            draw_y_label(ylabel, unit, fontsize=fontsize)
    else:
        plt.xlabel(xlabel)
        if ylabel is not None:
            draw_y_label(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=titlesize)