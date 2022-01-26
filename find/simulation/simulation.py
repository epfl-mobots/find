#!/usr/bin/env python

import os
import sys
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from find.models.loader import Loader
from find.models.storage import ModelStorage
from find.models.model_factory import ModelFactory
from find.simulation.simulation_factory import available_functors, SimulationFactory, get_most_influential_individual


def run_process(process):
    # os.system('python {}'.format(process))
    os.system('python {} >/dev/null 2>&1'.format(process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interaction simulator')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--reference', '-r', type=str,
                        help='Path to a reference experiment position file',
                        required=True)
    parser.add_argument('--load', '-l',
                        type=str,
                        help='Load model from existing file and continue the training process',
                        required=False)
    parser.add_argument('--num_proc', '-j', type=int,
                        help='Number of pool processes to use',
                        default=16,
                        required=False)
    parser.add_argument('--backend',
                        help='Backend selection',
                        default='keras',
                        choices=['keras', 'trajnet'])

    # model selection arguments
    nn_functor_selection = parser.add_argument_group('NN functor selection')
    nn_functor_selection.add_argument('--nn_functor',
                                      default=available_functors()[0],
                                      choices=available_functors())

    # simulation arguments
    simulation_group = parser.add_argument_group('Simulation configuration')
    simulation_group.add_argument('--iterations', '-i', type=int,
                                  help='Number of iteration of the simulation',
                                  default=-1,
                                  required=False)
    simulation_group.add_argument('--timestep', '-t', type=float,
                                  help='Simulation timestep',
                                  required=True)
    simulation_group.add_argument('--num_timesteps', type=int,
                                  help='Number of LSTM timesteps',
                                  default=0)
    simulation_group.add_argument('--prediction_steps', type=int,
                                  help='Number of prediction steps for the NN',
                                  default=1)
    simulation_group.add_argument('--distance_inputs', action='store_true',
                                  help='Use distance data as additional NN inputs',
                                  default=False)
    simulation_group.add_argument('--exclude_index', '-e', type=int,
                                  help='Index of the individual that will be replaced by a virtual agent (-1 will replace all original trajectories)',
                                  required=False,
                                  default=-1)
    simulation_group.add_argument('--polar', action='store_true',
                                  help='Use polar inputs instead of cartesian coordinates',
                                  default=False)
    simulation_group.add_argument('--timesteps_skip', type=int,
                                  help='Timesteps skipped between input and prediction',
                                  default=0,
                                  required=False)
    simulation_group.add_argument('--num_extra_virtu', type=int,
                                  help='Number of virtual individuals in the simulation',
                                  default=0)
    simulation_group.add_argument('--most_influential_individual',
                                  help='Criterion for most influential individual',
                                  default=get_most_influential_individual()[0],
                                  choices=get_most_influential_individual())
    simulation_group.add_argument('--var_coef', type=float,
                                  help='Prediction variance coefficient',
                                  default=1.0,
                                  required=False)
    simulation_group.add_argument('--simu_out_dir', type=str,
                                  help='Directory for simulation output files (always relative to the experiment path)',
                                  default='generated',
                                  required=False)

    args = parser.parse_args()
    args.simu_out_dir = args.path + '/' + args.simu_out_dir

    loader = Loader(path=args.path)
    model_storage = ModelStorage(args.path)
    model_factory = ModelFactory()
    simu_factory = SimulationFactory()

    model = model_storage.load_model(
        args.load, args.backend, args)
    if args.backend == 'keras':
        model.summary()
    elif args.backend == 'trajnet':
        args.nn_functor = 'trajnet_dir'

    print('Using {} backend'.format(args.backend))

    # read reference data
    data, files = loader.load(args.reference, is_absolute=True)
    if args.num_proc < 0:
        args.num_proc = len(data)

    if len(data) == 1:
        # generate simulator
        simu = simu_factory(
            data[0], model, args.nn_functor,  args.backend, args)
        if simu is None:
            import warnings
            warnings.warn('Skipping small simulation')
            exit(1)
        simu.spin()
    else:
        # in case we have multiple reference files then we create a number
        # of parallel process that will each simulate different reference
        # files.
        if '-reference' in sys.argv:
            idx_ref_file = sys.argv.index('-reference') + 1
        else:
            idx_ref_file = sys.argv.index('-r') + 1
        cmd = sys.argv

        simu_list = []
        for i, d in enumerate(data):
            f = files[i]
            cmd[idx_ref_file] = f
            simu_list.append(' '.join(cmd))

        with Pool(processes=args.num_proc) as p:
            total_simus = len(simu_list)
            with tqdm(total=total_simus, desc='Running multiple simulations') as pbar:
                for i, _ in enumerate(p.imap_unordered(run_process, simu_list)):
                    pbar.update()
