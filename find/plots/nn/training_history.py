#!/usr/bin/env python
import os
import glob
import argparse
from tqdm import tqdm

from find.plots.common import *


def get_directory_name_at_level(abs_path, depth=0, keep_parents=0):
    path = abs_path
    for _ in range(depth):
        path = os.path.dirname(path)

    parents = ''
    if keep_parents > 0:
        parents = [os.path.dirname(path)]
        for _ in range(keep_parents-1):
            parents.append(os.path.dirname(parents[-1]))
        parents = list(map(lambda p: os.path.basename(p) + '/', parents))
        parents = list(reversed(parents))
        parents = ''.join(parents)
    return parents + os.path.basename(path)


def plot_history(history_files, path, args):
    plot_dict = {}
    min_epoch = np.inf
    for hf in tqdm(history_files, desc='Transforming the history data'):
        with open(hf, "r") as file:
            header = file.readline().strip('\n').split(args.nn_delimiter)
        h = np.loadtxt(hf, skiprows=1, delimiter=args.nn_delimiter)

        if 'epoch' in header:
            epoch_idx = header.index('epoch')
            epoch_count = h[:, epoch_idx]
            h = np.delete(h, epoch_idx, axis=1)
            header.remove('epoch')
        else:
            epoch_count = np.array(list(range(h.shape[0])))

        if len(epoch_count) < min_epoch:
            min_epoch = len(epoch_count)

        for col, quantity in enumerate(header):
            if quantity not in plot_dict.keys():
                plot_dict[quantity] = []
            plot_dict[quantity].append(
                [get_directory_name_at_level(
                    hf, 2, args.nn_num_legend_parents), epoch_count, h[:, col]]
            )

    for k, v in plot_dict.items():
        for snum, (label, x, y) in enumerate(v):
            if args.nn_last_epoch == -1:
                nn_last_epoch = x.shape[0]
            elif args.nn_last_epoch == -2:
                nn_last_epoch = min_epoch
            else:
                nn_last_epoch = args.nn_last_epoch
                if args.nn_last_epoch > x.shape[0]:
                    nn_last_epoch = x.shape[0]
            (label, x, y) = (
                label, x[:nn_last_epoch],  y[:nn_last_epoch])

            if args.nn_num_sample_epochs > 0:
                sample_epochs = args.nn_num_sample_epochs
                if sample_epochs > x.shape[0]:
                    sample_epochs = x.shape[0]
                idcs_keep = np.arange(
                    0,  x.shape[0], sample_epochs)
                (label, x, y) = (label, x[idcs_keep], y[idcs_keep])

            plot_dict[k][snum] = (label, x, y)

    with tqdm(list(plot_dict.keys())) as pbar:
        for it, (k, v) in enumerate(plot_dict.items()):
            palette = sns.color_palette(
                'Spectral', n_colors=len(history_files))
            ccycler = cycle(palette)
            lines = ['-', '--', ':']
            linecycler = cycle(lines)

            pbar.set_description('Plotting {}'.format(k))
            pbar.update(it)
            abs_filename = path + '/' + k + '.png'

            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            for label, x, y in v:
                sns.lineplot(x=x, y=y, ax=ax, label=label,
                             linewidth=uni_linewidth, color=next(ccycler), linestyle=next(linecycler))
            ax.set_xlabel('Epochs')
            ax.set_ylabel(k)
            ax.legend(prop={'size': 4})
            plt.savefig(abs_filename)


def plot(exp_files, path, args):
    history_files = []
    for d in args.nn_compare_dirs:
        history_files.append(glob.glob(d + '/logs/history.csv'))
    history_files = [item for sublist in history_files for item in sublist]
    plot_history(history_files, path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot NN metrics from the training history')
    parser.add_argument('--nn_compare_dirs',
                        type=str,
                        nargs='+',
                        help='List of directories to look through and analyse',
                        required=False)
    parser.add_argument('--nn_compare_out_dir',
                        type=str,
                        help='Directory to output NN analysis results',
                        default='nn_comparison',
                        required=False)
    parser.add_argument('--nn_delimiter',
                        type=str,
                        help='Delimiter used in the log files',
                        default=',',
                        required=False)
    parser.add_argument('--nn_last_epoch',
                        type=int,
                        help='Plot up to nn_last_epoch data points. -1 stands for all, -2 stands for up to the min of iterations across the experiments',
                        default=-1,
                        required=False)
    parser.add_argument('--nn_num_legend_parents',
                        type=int,
                        help='Number of parent directories to show in the legend',
                        default=1,
                        required=False)
    parser.add_argument('--nn_num_sample_epochs',
                        type=int,
                        help='Number of samples to plot. -1 will consider all available points',
                        default=-1,
                        required=False)
    args = parser.parse_args()

    plot(None, './', args)
