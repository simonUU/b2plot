# -*- coding: utf-8 -*-
""" Stacked plot

In data science, often one wants to compare two different sets of data, like signal/background or prediction and
actual data.

In this very brief script we create two sets of data and compare them in one plot.

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import b2plot as bp

try:
    plt.style.use('belle2')
except OSError:
    print("Please install belle2 matplotlib style")

data = np.random.normal(0, 0.5, 1000)

bp.hist(data)
bp.labels("O", "Entries", "Unit")

# bp.xlim()
# bp.labels('$\Delta M$', "Events", "GeV", 0)
bp.save("histogram.png")
