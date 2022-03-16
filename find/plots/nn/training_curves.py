#!/usr/bin/env python
import os
import glob
import argparse
from tqdm import tqdm
from copy import deepcopy

from find.plots.common import *
from find.plots.nn.training_history import prepare_plot_history


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
    return parents + os.path .basename(path)


def plot(exp_files, path, args):
    test_args = deepcopy(args)
    arg_dict = vars(test_args)
    arg_dict['nn_last_epoch'] = -1
    arg_dict['nn_num_sample_epochs'] = -1

    log_files = glob.glob(args.path + '/logs/history.csv')
    test_log_files = glob.glob(args.path + '/test/history.csv')
    # if len(log_files) == 0 or len(test_log_files) == 0:
    #     return 

    plot_dict = prepare_plot_history(log_files, args.path, args)
    # plot_dict_test = prepare_plot_history(test_log_files, args.path, test_args)

    # 'epoch,gaussian_mae,gaussian_mse,loss,val_gaussian_mae,val_gaussian_mse,val_loss'

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    lines = ['-', '--', ':']
    linecycler = cycle(lines)

    for k in ['loss', 'val_loss']:
        _, x, y = plot_dict[k][0]
        sns.lineplot(x=x, y=y, ax=ax,
                        linewidth=uni_linewidth, color='blue', label=k, linestyle=next(linecycler))

    # _, _, yt = plot_dict_test['loss'][0]
    # _, _, xt = plot_dict_test['epochs'][0]

    # sns.lineplot(x=xt, y=yt, ax=ax,
    #                 linewidth=uni_linewidth, color='blue', label='test_loss', linestyle=next(linecycler))  
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_xlim([0, x[-1]])
    # ax.set_ylim([-3, 3]) # TODO: remove
    plt.savefig(args.path + '/test.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    lines = ['-', '--']
    linecycler = cycle(lines)

    for k in ['gaussian_mse', 'val_gaussian_mse']:
        _, x, y = plot_dict[k][0]
        sns.lineplot(x=x, y=y, ax=ax,
                        linewidth=uni_linewidth, color='blue', label=k, linestyle=next(linecycler))

    # _, _, yt = plot_dict_test['gaussian_mse'][0]
    # _, _, xt = plot_dict_test['epochs'][0]

    # sns.lineplot(x=xt, y=yt, ax=ax,
    #                 linewidth=uni_linewidth, color='blue', label='test_gaussian_mse', linestyle=next(linecycler))  

    ax.legend()
    ax.set_xlabel('Epochs')
    # ax.set_xlim([0, x[-1]])
    plt.savefig(args.path + '/test2.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot NN metrics from the training history')
    parser.add_argument('--path', '-p',
                        type=str,
                        help='Path to the experiment',
                        required=True)
    args = parser.parse_args()

    plot(None, './', args)
