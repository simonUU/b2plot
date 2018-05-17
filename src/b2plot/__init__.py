import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version

from ._style import set_default_style, set_small_style
from .plot_wrappers import hist, errorhist
from ._manager import xaxis, nf
from .functions import draw_y_label, decorate

