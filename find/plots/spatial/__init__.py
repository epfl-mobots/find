from find.plots.spatial import angular_velocity
from find.plots.spatial import distance_to_wall
from find.plots.spatial import grid_occupancy
from find.plots.spatial import interindividual_distance
from find.plots.spatial import relative_orientation
from find.plots.spatial import resultant_acceleration
from find.plots.spatial import resultant_velocity
from find.plots.spatial import future_trajectory_variance
from find.plots.spatial import grid_distribution_comparison
from find.plots.spatial import rwt
from find.plots.spatial import simu_comp

plot_dict = {
    'angular_velocity': angular_velocity.plot,
    'distance_to_wall': distance_to_wall.plot,
    'grid_occupancy': grid_occupancy.plot,
    'interindividual_distance': interindividual_distance.plot,
    'relative_orientation': relative_orientation.plot,
    'resultant_acceleration': resultant_acceleration.plot,
    'resultant_velocity': resultant_velocity.plot,
    'future_trajectory_variance': future_trajectory_variance.plot,
    'grid_distribution_comparison': grid_distribution_comparison.plot,
    'rwt': rwt.plot,
    'simu_comp': simu_comp.plot,
}


source = 'spatial'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
