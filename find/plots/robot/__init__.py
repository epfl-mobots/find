from find.plots.robot import individual_quantities
from find.plots.robot import collective_quantities
from find.plots.robot import correlation_quantities
from find.plots.robot import occupancy_grids
from find.plots.robot import velocity_bplots
from find.plots.robot import acceleration_bplots
from find.plots.robot import interindividual_dist_bplots
from find.plots.robot import activity_bplots
from find.plots.robot import comp_plot
from find.plots.robot import comp_plot_alt
from find.plots.robot import quantity_stats


plot_dict = {
    'individual_quantities_rob': individual_quantities.plot,
    'collective_quantities_rob': collective_quantities.plot,
    'correlation_quantities_rob': correlation_quantities.plot,
    'occupancy_grids_rob': occupancy_grids.plot,
    'velocity_bplots_rob': velocity_bplots.plot,
    'acceleration_bplots_rob': acceleration_bplots.plot,
    'interindividual_dist_bplots_rob': interindividual_dist_bplots.plot,
    'activity_bplots_rob': activity_bplots.plot,
    'comp_plot_rob': comp_plot.plot,
    'comp_plot_alt_rob': comp_plot_alt.plot,
    'quantity_stats_rob': quantity_stats.plot,
}

source = 'robot'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
