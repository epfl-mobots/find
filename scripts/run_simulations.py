#!/usr/bin/env python

import os
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool


models = ['prob', 'ffw', 'rnn', 'prob_multi']
model_map = {
    'prob': 'prob',
    'ffw': 'ffw',
    'rnn': 'rnn',
    'prob_multi': 'prob',
}

script_map = {
    'prob': 'sim/probabilistic_sim.py',
    'ffw': 'sim/feedforward_sim.py',
    'rnn': 'sim/rnn_sim',
    'prob_multi': 'sim/probabilistic_multi_sim.py',
}


def run_process(process):
    os.system('python {} >/dev/null 2>&1'.format(process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple script to run all available simulations in parallel')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--reference', '-r', type=str,
                        help='Path to a reference experiment position file',
                        required=True)
    parser.add_argument('--iterations', '-i', type=int,
                        help='Number of iteration of the simulation',
                        required=False,
                        default=-1)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--model', '-m',
                        type=str,
                        choices=models)
    args = parser.parse_args()

    files = glob.glob(
        str(Path(args.path).joinpath('*processed_positions.dat')))

    arglist = []
    for f in files:
        m = model_map[args.model]
        exp_arglist = ' -p ' + args.path + ' -r ' + f + \
            ' -m ' + m + ' -t ' + str(args.timestep)
        arglist.append(exp_arglist)

    processes = [script_map[args.model]] * len(files)
    for i in range(len(processes)):
        processes[i] = processes[i] + arglist[i]
        print(processes[i])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
