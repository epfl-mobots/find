from find.plots.robot import individual_quantities
from find.plots.robot import collective_quantities
from find.plots.robot import correlation_quantities
from find.plots.robot import occupancy_grids
from find.plots.robot import velocity_bplots

plot_dict = {
    'individual_quantities_rob': individual_quantities.plot,
    'collective_quantities_rob': collective_quantities.plot,
    'correlation_quantities_rob': correlation_quantities.plot,
    'occupancy_grids_rob': occupancy_grids.plot,
    'velocity_bplots_rob': velocity_bplots.plot,
}

source = 'robot'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
