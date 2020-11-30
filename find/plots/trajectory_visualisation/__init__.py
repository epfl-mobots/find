from find.plots.trajectory_visualisation import visualise_trajectories

plot_dict = {
    'visualise_trajectories': visualise_trajectories.plot,
}

source = 'trajectory_visualisation'


def available_plots():
    return list(plot_dict.keys())


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
