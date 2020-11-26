plot_dict = {
}


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
