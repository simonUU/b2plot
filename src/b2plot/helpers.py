# -*- coding: utf-8 -*-
"""
Helper function and classes are defined here.

"""


import matplotlib.pyplot as plt


def hist2root(h, name=None, title=''):
    """ Convert your histograms to a root histogram.. if you really want to

    Args:
        h: array with the histogram h=(y_values, bin_edges, (patches=optional))
        name: Name for the root histogram, optional
        title: Title for the root histogram, optional

    Returns:
        TH1F or TH2F depending on the input array

    Examples:
        >>> data = [1,2,3,4,5,6,7]
        >>> h = b2plot.hist(data)
        >>> from b2plot.helpers import hist2root
        >>> root_hist = hist2root(h)

    """
    import root_numpy
    import random
    import ROOT

    y_values = h[0]
    name = "".join(random.choice("fck_root_wtf") for _ in range(5)) if name is None else name
    if len(h) > 3:
        rh = ROOT.TH2F(name, title, len(h[1]) - 1, h[1], len(h[2]) - 1, h[2])
    else:
        rh = ROOT.TH1F(name, title, len(h[1]) - 1, h[1])
    return root_numpy.array2hist(y_values, rh)


def get_optimal_bin_size(n):
    """
    This function calculates the optimal amount of bins for the number of events n.
    :param      n:  number of Events
    :return:        optimal bin size

    """
    return int(3 * n**(1/3.0))


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class TheManager:
    def __init__(self):
        self.xaxis = None

    def get_x_axis(self):
        return self.xaxis

    def set_x_axis(self, axis):
        self.xaxis = axis

    def figure(self):
        # f = plt.figure(tight_layout={'pad': 0})
        f = plt.figure()
        self.xaxis = None
        return f


manager = TheManager.Instance()


def xaxis():
    return manager.get_x_axis()


def nf():
    return manager.figure()


def figure():
    return manager.figure()
