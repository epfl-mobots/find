from find.plots.trajectory_visualisation import visualise_trajectories
from find.plots.trajectory_visualisation import trajectory_grid
from find.plots.trajectory_visualisation import trajectory_segment

plot_dict = {
    'visualise_trajectories': visualise_trajectories.plot,
    'trajectory_grid': trajectory_grid.plot,
    'trajectory_segment': trajectory_segment.plot,
}

source = 'trajectory_visualisation'


def available_plots():
    return list(plot_dict.keys())


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
