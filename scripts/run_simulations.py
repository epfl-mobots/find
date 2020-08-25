#!/usr/bin/env python

import os
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool


model_map = {
    'prob': 'prob',
    'prob_lstm': 'prob_lstm',
    'lstm': 'lstm',
    'ffw': 'ffw',
    'rnn': 'rnn',
    'prob_multi': 'prob',
    'ffw_multi': 'ffw',
    'prob_lstm_multi': 'prob_lstm',
    'lstm_multi': 'lstm',
}

script_map = {
    'prob': 'simulation/probabilistic_sim.py',
    'ffw': 'simulation/feedforward_sim.py',
    'rnn': 'simulation/rnn_sim',
    'prob_multi': 'simulation/probabilistic_multi_sim.py',
    'prob_lstm': 'simulation/probabilistic_lstm_sim.py',
    'lstm': 'simulation/lstm_sim.py',
    'ffw_multi': 'simulation/feedforward_multi_sim.py',
    'prob_lstm_multi': 'simulation/probabilistic_lstm_multi_sim.py',
    'lstm_multi': 'simulation/lstm_multi_sim.py',
}

models = list(model_map.keys())



def run_process(process):
    # os.system('python {} >/dev/null 2>&1'.format(process))
    os.system('python {}'.format(process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple script to run all available simulations in parallel')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--arglist', type=str,
                        help='Argument list to forward to the corresponding',
                        default='',
                        required=False)
    parser.add_argument('--model', '-m',
                        type=str,
                        choices=models)
    parser.add_argument('--model-snap', type=str,
                        help='Model snapshot number',
                        default='',
                        required=False)
    parser.add_argument('--num-processes', '-j',
                        type=int,
                        default=12)
    args = parser.parse_args()

    files = glob.glob(
        str(Path(args.path).joinpath('*processed_positions.dat')))

    if len(args.model_snap) > 1:
        args.model_snap = '_' + args.model_snap

    arglist = []
    for f in files:
        m = model_map[args.model]
        exp_arglist = ' -p ' + args.path + ' -r ' + f + \
            ' -m ' + m + args.model_snap + ' ' + args.arglist
        arglist.append(exp_arglist)

    processes = [script_map[args.model]] * len(files)
    for i in range(len(processes)):
        processes[i] = processes[i] + arglist[i]

    pool = Pool(processes=args.num_processes)
    pool.map(run_process, processes)
