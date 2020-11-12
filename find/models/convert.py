#!/usr/bin/env python

import argparse

from find.models.loader import Loader
from find.models.storage import ModelStorage
from find.models.model_factory import ModelFactory, available_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data converter')
    parser.add_argument('--files', '-f',
                        type=str,
                        help='Files to look for',
                        default='processed_positions.dat',
                        required=False)
    parser.add_argument('--path', '-p',
                        type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--timestep', '-t',
                        type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--timesteps_skip',
                        type=int,
                        help='Timesteps skipped between input and prediction',
                        default=0,
                        required=False)

    # model selection arguments
    model_selection = parser.add_argument_group('Model selection')
    model_selection.add_argument('--model',
                                 default=available_models()[0],
                                 choices=available_models())

    # model options
    model_options = parser.add_argument_group('Model options')
    model_options.add_argument('--polar', action='store_true',
                               help='Use polar inputs instead of cartesian coordinates',
                               default=False)
    model_options.add_argument('--prediction_steps', type=int,
                               help='Trajectory steps to predict',
                               default=1)
    model_options.add_argument('--num_timesteps', type=int,
                               help='Number of LSTM timesteps',
                               default=5)
    model_options.add_argument('--distance_inputs', action='store_true',
                               help='Use distance data as additional NN inputs',
                               default=False)

    # data split
    data_split_options = parser.add_argument_group('Data split options')
    data_split_options.add_argument('--train_fraction',
                                    type=float,
                                    help='Validation set fraction',
                                    default=0.85)
    data_split_options.add_argument('--val_fraction',
                                    type=float,
                                    help='Validation set fraction',
                                    default=0.13)
    data_split_options.add_argument('--test_fraction',
                                    type=float,
                                    help='Test set fraction',
                                    default=0.02)
    args = parser.parse_args()

    # data loading is handled here depending on the number of individuals
    # the loader will also handle the data splitting process according
    # to the arguments provided
    loader = Loader(path=args.path)
    pos, files = loader.load(args.files)
    inputs, outputs = loader.prepare(pos, args)
    td, tv, tt = loader.split_to_sets(inputs, outputs, args)

    # model storage instance to tidy up the directory and take care of saving/loading
    model_storage = ModelStorage(args.path)
    model_storage.save_sets(td, tv, tt)
