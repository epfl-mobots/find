from find.plots.dl_si_2021 import individual_quantities
from find.plots.dl_si_2021 import collective_quantities
from find.plots.dl_si_2021 import correlation_quantities
from find.plots.dl_si_2021 import occupancy_grids
from find.plots.dl_si_2021 import nn_plots

plot_dict = {
    'individual_quantities': individual_quantities.plot,
    'collective_quantities': collective_quantities.plot,
    'correlation_quantities': correlation_quantities.plot,
    'occupancy_grids': occupancy_grids.plot,
    'nn_plots': nn_plots.plot,
}

source = 'dl-si-2021'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
