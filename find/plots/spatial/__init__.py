from find.plots.spatial import angular_velocity
from find.plots.spatial import distance_to_wall

plot_dict = {
    'angular_velocity': angular_velocity.plot,
    'distance_to_wall': distance_to_wall.plot,
}


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
