from find.plots.dynamics import phase_diagram


plot_dict = {
    'phase_diagram': phase_diagram.plot,
}

source = 'dynamics'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
