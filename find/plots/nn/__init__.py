from find.plots.nn import training_history

plot_dict = {
    'training_history': training_history.plot,
}

source = 'nn'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
