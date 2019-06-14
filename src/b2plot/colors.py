# -*- coding: utf-8 -*-
""" Color definitions


"""

import seaborn as sns


b2cm = ['#E24A33', '#348ABD', '#FBC15E', '#8EBA42', '#988ED5', '#FFB5B8', '#777777']

b2cm2 = ['#fd6769', '#6bcdfd', '#356535', '#971497']

tango_light = ["#ef2929", "#729fcf", "#fcaf3e", "#8ae234", "#ad7fa8", "#fce94f", "#888a85", "#e9b96e"]

tango_medium = ["#cc0000", "#3465a4", "#f57900", "#73d216", "#75507b", "#edd400", "#555753", "#c17d11"]

tango_dark = ["#a40000", "#204a87", "#ce5c00", "#4e9a06", "#5c3566", "#c4a000", "#2e3436", "#8f5902"]

solarized = ["#dc322f", "#268bd2", "#b58900", "#859900", "#2aa198", "#c4a000", "#cb4b16", "#fdf6e3"]

groovebox = ["#cc241d", "#458588", "#d79921", "#98971a", "#b16286", "#689d6a", "#d65d0e", "#fbf1c7"]


def cm(n):
    return b2cm[:n % len(b2cm)]


def b2blues(n=5):
    return sns.color_palette("PuBuGn_d", n)


def b2greens(n=3):
    return sns.color_palette("summer", n)


def b2helix(n):
    return sns.cubehelix_palette(n, start=1.5, rot=1.5, dark=0.3, light=.8, reverse=True)