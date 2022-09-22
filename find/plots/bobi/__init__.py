from find.plots.bobi import individual_quantities
from find.plots.bobi import collective_quantities
from find.plots.bobi import correlation_quantities
from find.plots.bobi import occupancy_grids

plot_dict = {
    'individual_quantities_rob': individual_quantities.plot,
    'collective_quantities_rob': collective_quantities.plot,
    'correlation_quantities_rob': correlation_quantities.plot,
    'occupancy_grids_rob': occupancy_grids.plot,
}

source = 'bobi'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
