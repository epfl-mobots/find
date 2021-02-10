from find.plots.correlation import position_correlation
from find.plots.correlation import velocity_correlation
from find.plots.correlation import relative_orientation_correlation

plot_dict = {
    'position_correlation': position_correlation.plot,
    'velocity_correlation': velocity_correlation.plot,
    'relative_orientation_correlation': relative_orientation_correlation.plot,
}

source = 'correlation'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
