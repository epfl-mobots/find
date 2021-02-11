from find.plots.pnas import individual_quantities
from find.plots.pnas import collective_quantities
from find.plots.pnas import correlation_quantities

plot_dict = {
    'individual_quantities': individual_quantities.plot,
    'collective_quantities': collective_quantities.plot,
    'correlation_quantities': correlation_quantities.plot,
}

source = 'pnas'


def available_plots():
    return list(plot_dict.keys())


def get_plot(key):
    return plot_dict[key]
