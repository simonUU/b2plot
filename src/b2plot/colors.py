# -*- coding: utf-8 -*-
""" Color definitions


"""


import seaborn as sns

b2cm = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

b2cm2 = ['#fd6769', '#6bcdfd', '#356535', '#971497']


def cm(n):
    return b2cm[:n % len(b2cm)]


def b2blues(n=5):
    return sns.color_palette("PuBuGn_d", n)


def b2greens(n=3):
    return sns.color_palette("summer", n)


def b2helix(n):
    return sns.cubehelix_palette(n, start=1.5, rot=1.5, dark=0.3, light=.8, reverse=True)