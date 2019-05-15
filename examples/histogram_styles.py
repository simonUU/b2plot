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


bp.hist(np.random.normal(0, 0.4, 1000), range=(-1, 8), label="None")
for i in range(6):
    bp.hist(np.random.normal(1+i, 0.4, 1000), style=i, label="Style %d"%i)



# bp.xlim()
plt.legend()
bp.labels("O", "Entries", "Unit")
bp.save("histogram_styles.png")
