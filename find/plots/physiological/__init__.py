from find.plots.physiological import kick

plot_dict = {
    'kick': kick.plot
}


source = 'physiological'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
