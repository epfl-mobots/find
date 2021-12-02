#!/usr/bin/env python
import glob
import argparse

from find.plots.common import *
import find.plots.common as shared

import find.plots.nn.training_history as th
import find.plots.nn.parameters_to_epoch as pte


def plot(exp_files, path, args):
    history_files = []
    for d in args.nn_compare_dirs:
        history_files.append(glob.glob(d + '/logs/history.csv'))

    _, ax = plt.subplots(figsize=(10, 10),
                         nrows=6, ncols=len(history_files),
                         gridspec_kw={'width_ratios': [
                             1] * len(history_files), 'wspace': 1.4, 'hspace': 1.2}
                         )

    config = {
        'gaussian_mae': {
            'title': 'Mean Absolute error\n (training)',
            'limy': (0.02, 0.14, 0.03),
            'limx': (0, 15000, 3000),

        },
        'gaussian_mse': {
            'title': 'Mean Squared error\n (training)',
            'limy': (0.0025, 0.035, 0.01),
            'limx': (0, 15000, 3000),

        },
        'loss': {
            'title': 'Loss\n (training)',
            'limy': (-6.5, 4, 2),
            'limx': (0, 15000, 3000),

        },
        'val_gaussian_mae': {
            'title': 'Mean Absolute error\n (validation)',
            'limy': (0.02, 0.14, 0.03),
            'limx': (0, 15000, 3000),

        },
        'val_gaussian_mse': {
            'title': 'Mean Squared error\n (validation)',
            'limy': (0.0025, 0.035, 0.01),
            'limx': (0, 15000, 3000),

        },
        'val_loss': {
            'title': 'Loss\n (validation)',
            'limy':  (-3.5, 6, 2),
            'limx': (0, 15000, 3000),

        },
        'cax_lstm': 1.1,
        'cax_pfw': 1.1,
    }

    for hf_idx, hf in enumerate(history_files):
        plot_dict = th.prepare_plot_history(hf, path, args)

        for it, (k, v) in enumerate(sorted(plot_dict.items())):
            if len(hf) > 1:
                cax = ax[it, hf_idx]
            else:
                cax = ax[it]
            palette = sns.color_palette(
                'Spectral', n_colors=len(v))
            ccycler = cycle(palette)
            lines = ['-', '--', ':']
            linecycler = cycle(lines)

            for label, x, y in v:
                sns.lineplot(x=x, y=y, ax=cax, label=label,
                             linewidth=uni_linewidth, color=next(ccycler), linestyle=next(linecycler))
                cax.legend().remove()
            cax.set_xlabel('Epochs')
            cax.set_ylabel(config[k]['title'])

            cax.set_yticks(np.arange(
                config[k]['limy'][0], config[k]['limy'][1] + 0.0001, config[k]['limy'][2]))
            cax.set_ylim(config[k]['limy'][0], config[k]['limy'][1] + 0.0001)
            cax.set_xticks(np.arange(
                config[k]['limx'][0], config[k]['limx'][1] + 0.0001, config[k]['limx'][2]))
            cax.set_xlim(config[k]['limx'][0], config[k]['limx'][1] + 0.0001)
            cax.tick_params(axis='x', rotation=45)

            if it == 0:
                if 'pfw' in hf:
                    offset = config['cax_pfw']
                    print('ok')
                else:
                    offset = config['cax_lstm']
                cax.legend(bbox_to_anchor=(
                    offset, 1.0), bbox_transform=cax.transAxes, prop={'size': 4})

    plt.savefig(path + '/nn_th_eval.png', bbox_inches="tight")

    # _, ax = plt.subplots(figsize=(11, 8),
    #                      nrows=2, ncols=len(history_files),
    #                      gridspec_kw={'width_ratios': [
    #                          1] * len(history_files), 'wspace': 1.0, 'hspace': 0.25}
    #                      )

    # for hf_idx, hf in enumerate(history_files):
    #     plot_dict = pte.plot_params_to_epoch(hf, path, args)

    #     row_select = 0
    #     for it, (k, v) in enumerate(sorted(plot_dict.items())):
    #         if 'loss' not in k:
    #             continue

    #         if len(hf) > 1:
    #             cax = ax[row_select, hf_idx]
    #         else:
    #             cax = ax[it]
    #         palette = sns.color_palette(
    #             'Spectral', n_colors=len(hf))
    #         ccycler = cycle(palette)

    #         if k == 'num_params':
    #             continue

    #         xs = []
    #         ys = []
    #         zs = []
    #         labels = []
    #         for idx, (label, x, y) in enumerate(v):
    #             xs.append(plot_dict['num_params'][idx])
    #             ys.append(y[-1])
    #             zs.append(x[-1])
    #             labels.append(label)
    #         zs = np.array(zs) / np.max(zs) * 6

    #         for idx in range(len(xs)):
    #             cax.plot(xs[idx], ys[idx], 'o', markersize=zs[idx],
    #                      label=labels[idx], color=next(ccycler))

    #         cax.set_xlabel('Number of parameters')
    #         cax.set_ylabel(k)
    #         cax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    #         cax.set_yticks(np.arange(
    #             config['loss']['limy'][0], config['loss']['limy'][1] + 0.0001, config['loss']['limy'][2]))
    #         cax.set_ylim(config['loss']['limy'][0],
    #                      config['loss']['limy'][1] + 0.0001)

    #         if row_select == 0:
    #             cax.legend(
    #                 prop={'size': 4}, fontsize='x-small',
    #                 bbox_to_anchor=(1.65, 1.0), borderaxespad=0,
    #                 labelspacing=2.5,
    #                 ncol=1)
    #         row_select += 1

    # plt.savefig(path + '/nn_pte_eval.png', bbox_inches="tight")
