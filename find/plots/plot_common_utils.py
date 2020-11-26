import numpy as np
import seaborn as sns
import matplotlib.lines as mlines

from itertools import cycle
from pylab import plt, mpl, rcParams, Rectangle, Circle

# flatui = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
uni_palette = sns.color_palette("bright", n_colors=20, desat=.5)
uni_colours = sns.color_palette(uni_palette)
uni_cycler = cycle(uni_palette)
uni_linewidth = 1.2

params = {
    #     'axes.labelsize': 11,
    #     'font.size': 11,
    #     'legend.fontsize': 11,
    #     # 'legend.fontsize': 'medium',
    #     # 'figure.titlesize': 'medium',
    #     'xtick.labelsize': 11,
    #     'ytick.labelsize': 11,
    #     'text.usetex': False,
    #     # 'figure.figsize': [10, 15]
    #     # 'ytick.major.pad': 4,
    #     # 'xtick.major.pad': 4,
    #     'font.family': 'Arial',
    'figure.dpi': 300,
    'savefig.dpi': 300,
}
rcParams.update(params)

sns.set_style(
    "whitegrid",
    # "darkgrid",
    {
        'axes.axisbelow': True,
        'axes.edgecolor': '.8',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.labelcolor': '.15',
        'axes.linewidth': 1.0,
        'font.family': [u'sans-serif'],
        'font.sans-serif': [u'Arial',
                            u'Liberation Sans',
                            u'Bitstream Vera Sans',
                            u'sans-serif'],
        'grid.color': '.8',
        'grid.linestyle': u'-',
        'image.cmap': u'Greys',
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'lines.solid_capstyle': u'round',
        'text.color': '.15',
        'xtick.color': '.15',
        'xtick.direction': u'out',
        'xtick.major.size': 0.0,
        'xtick.minor.size': 0.0,
        'ytick.color': '.15',
        'ytick.direction': u'out',
        'ytick.major.size': 0.0,
        'ytick.minor.size': 0.0,
    })


uni_lines = ["-"]
uni_linecycler = cycle(uni_lines)
uni_pts = np.linspace(0, np.pi * 2, 24)
uni_circ = np.c_[np.sin(uni_pts) / 2, -np.cos(uni_pts) / 2]
uni_vert = np.r_[uni_circ, uni_circ[::-1] * 1.0]
uni_open_circle = mpl.path.Path(uni_vert)
uni_extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                      edgecolor='none', linewidth=0)

uni_v = np.r_[uni_circ, uni_circ[::-1] * 0.6]
uni_oc = mpl.path.Path(uni_v)

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=uni_oc,
                  markersize=6, label='Mean and SD'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='*',
                  markersize=5, label='Median'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
                  color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.5,
                  markersize=5, label='Single run')
]

handles_b = [
    mlines.Line2D([0], [1], color='black', label='Mean'),
    Circle((0, 0), radius=1, facecolor='black', alpha=0.35, label='SD')
]
