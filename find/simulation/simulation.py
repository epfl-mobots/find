#!/usr/bin/env python

import argparse
from find.models.loader import Loader
from find.models.storage import ModelStorage
from find.models.model_factory import ModelFactory
from find.simulation.simulation_factory import available_functors, SimulationFactory, get_most_influential_individual


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
        args.load, 'keras', args)  # TODO: future versions should handle other backends

    # read reference data
    data, _ = loader.load(args.reference, is_absolute=True)

    # generate simulator
    for d in data:
        simu = simu_factory(d, model, args.nn_functor, 'keras', args)
        if simu is None:
            import warnings
            warnings.warn('Skipping small simulation')
            exit(1)
        simu.spin()
