import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version

from .functions import hist, errorbar, stacked, to_stack, xlim, sig_bkg_plot
from .helpers import xaxis, nf, figure
from .decorations import draw_y_label, decorate, expand, watermark, lumi, labels
from .colors import cm, b2helix

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
# mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'

print("For optimal usage set `plt.style.use('belle2')`")
