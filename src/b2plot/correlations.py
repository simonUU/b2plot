# -*- coding: utf-8 -*-
""" Tools for studying correlations

Author:
    - Simon Wehle (swehle@desy.de)

"""

import numpy as np
from scipy.stats import binned_statistic_2d
from scipy import stats
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


def flat_correlation(x,y, nbins='auto', zoom=1, nlabels=5, ax=None, ax_fmt='%.2e', x_label_rot=45):
    """ Calculate and plot a 2D correlation in flat binning.
    This function calculates an equal frequency binning for x and y and fills a 2D histogram with this binning.
    Thus each slice in x and y contains the same number of entries for continuus distributions.
    For uncorrelated distributions the expected amount of each bin is N_expected = N_total / N_bins**2 
    This plot shows the statistical significance of the deviation from N_expected.

    Args:
        x: array of values to be binned in x direction
        y: array of values to be binned in y direction
        nbins: int or 'auto', number of bins in x and y
        zoom: factor f of the significance [-f*5,f*5]
        nlabels: number of x,y labels
        ax: axes, if None, takes current
        ax_fmt: formatter for tick labls
        x_label_rot: rotation for x labels

    Returns:
        chi2 probability for flat distribution
    """
    ax = plt.gca() if ax is None else ax
    
    # calculate equal fequrency binning
    nbins = 2*(3*len(x)**(1/3))**(1/2) if nbins=='auto' else nbins
    binsx = np.percentile(x, np.linspace(0,100, nbins))
    binsy = np.percentile(y, np.linspace(0,100, nbins))
    # Bin count
    bs = binned_statistic_2d(x, y, None, statistic='count', bins=[binsx,binsy])
    # Calculate actual count - expected significance
    nexp = len(x)/(nbins-1)**2
    nerr = np.sqrt(nexp)
    a0 = bs.statistic.T
    a = np.array(a0)
    a = (a - nexp)/nerr
    a[a0==0] = None
    # Plotting
    im = ax.imshow(a, cmap=plt.cm.jet, interpolation='nearest', origin='lower',vmin=-5*zoom, vmax=5*zoom)
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04, ax=ax)
    # set labels
    ax.set_xticks(np.linspace(*ax.get_xlim(), nlabels))
    ax.set_xticklabels([ax_fmt%f for f in np.percentile(x, np.linspace(0,100, nlabels))], rotation=x_label_rot, ha='right')
    ax.set_yticks(np.linspace(*ax.get_ylim(), nlabels))
    ax.set_yticklabels([ax_fmt%f for f in np.percentile(y, np.linspace(0,100, nlabels))]) 
    # Calculate chi2 probability
    flat_probability = stats.distributions.chi2.sf(np.nansum(a*a),(nbins)**2-(nbins-1)-(nbins-1)-1)
    return  flat_probability


def heatmap(x, y, tfs=12, bkg_color='#F1F1F1', separate_first=0, **kwargs):
    """ Calculate a heatmap

    Based on: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(359,122, s=90, n=500) #sns.color_palette("BrBG", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 30, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}
    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right', fontsize=tfs)
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num], fontsize=tfs)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor(bkg_color)
    
    if separate_first:
        l = np.sqrt(len(x))
        plt.axvline(separate_first - .5, color='gray')
        plt.axhline(l - .5 - separate_first , color='gray')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot
        #ax.axis('off')
        plt.box(on=None)
        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
        bar_height = bar_y[1] - bar_y[0]
        print(bar_height)
        ax.barh(
            y=bar_y,
            width=[15]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_ylim(-2,2)
        ax.set_xlim(0, 5)# Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 
    plt.sca(plt.subplot(plot_grid[:,:-1]))


def corrplot(data, size_scale=500, marker='s',tfs=12,
             separate_first=0,
             *args,**kwargs):
    """ Correlation plot

    Based on: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    """
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
        tfs=tfs,
        separate_first=separate_first,
       *args,**kwargs
    )


def corrmatrix(corr, separate_first=0, x_label_rot=45, invert_y=True, label_font_size=None, ax=None):
    """ 
    Recommendation:
    
        with plt.style.context(['default','seaborn-bright']):  
            corrmatrix(corrm_s,separate_first=2)
    """
    ax = plt.gca() if ax is None else ax
    
    sns.heatmap(corr, annot=False, cmap='PiYG',square=True, vmax=1,vmin=-1,)
    plt.ylim(*plt.xlim())
    
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=x_label_rot, horizontalalignment='right', fontsize=label_font_size)
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=label_font_size)
    
    if invert_y:
        plt.gca().invert_yaxis()

    if separate_first > 0:    
        plt.axhline(separate_first, color='gray',lw=1)
        plt.axvline(separate_first, color='gray',lw=1)
