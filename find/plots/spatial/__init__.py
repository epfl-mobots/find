from find.plots.spatial import angular_velocity

plot_dict = {
    'angular_velocity': angular_velocity.plot
}


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
