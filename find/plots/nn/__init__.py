from find.plots.nn import training_history
from find.plots.nn import parameters_to_epoch

plot_dict = {
    'training_history': training_history.plot,
    'parameters_to_epoch': parameters_to_epoch.plot,
}

source = 'nn'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
