#!/usr/bin/env python

import tqdm
import argparse
import numpy as np
from glob import glob

from find.models.loader import Loader
from find.models.storage import ModelStorage
from find.models.model_factory import ModelFactory, available_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model to reproduce fish motion')
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
    parser.add_argument('--epochs', '-e',
                        type=int,
                        help='Number of training epochs',
                        default=1000)
    parser.add_argument('--batch_size', '-b',
                        type=int,
                        help='Batch size',
                        default=256)
    parser.add_argument('--learning_rate', '-r',
                        type=float,
                        help='Learning rate',
                        default=0.0001)
    parser.add_argument('--dump', '-d',
                        type=int,
                        help='Batch size',
                        default=100)
    parser.add_argument('--load', '-l',
                        type=str,
                        help='Load model from existing file and continue the training process',
                        required=False)
    parser.add_argument('--timesteps_skip',
                        type=int,
                        help='Timesteps skipped between input and prediction',
                        default=0,
                        required=False)
    parser.add_argument('--reload_state', action='store_true',
                        help='Perform the data convertion step from the beginning',
                        default=False)
    parser.add_argument('--verbose', action='store_true',
                        help='Print messages where possible',
                        default=False)

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

    # model options
    model_builder = parser.add_argument_group('Model builder options')
    model_builder.add_argument('--model_layers', type=str,
                               nargs="+",
                               default=['LSTM', 'Dense', 'Dense',
                                        'Dense', 'Dense_out'],
                               help='NN structure for model builder',
                               required=False)
    model_builder.add_argument('--model_neurons', type=int,
                               nargs="+",
                               default=[32, 25, 16, 10, -1],
                               help='NN layer neurons',
                               required=False)
    model_builder.add_argument('--model_activations', type=str,
                               nargs="+",
                               default=['sigmoid', 'sigmoid',
                                        'sigmoid', 'tanh', 'None'],
                               help='NN layer activations',
                               required=False)

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

    # logging & stopping criteria options
    logstop_group = parser.add_argument_group('Logging & stopping criteria')
    logstop_group.add_argument('--model_checkpoint', action='store_true',
                               help='Save the best model as a checkpoint',
                               default=False)
    logstop_group.add_argument('--early_stopping', action='store_true',
                               help='Enable early stopping if the NN is converging',
                               default=False)
    logstop_group.add_argument('--min_delta',
                               type=float,
                               help='Minimum delta for early stopping',
                               default=0.1)
    logstop_group.add_argument('--patience',
                               type=int,
                               help='Epoch patience for stopping criteria',
                               default=10)

    logstop_group.add_argument('--enable_tensorboard', action='store_true',
                               help='Enable tensorboard logging',
                               default=False)
    logstop_group.add_argument('--custom_logs', action='store_true',
                               help='Enable custom logging',
                               default=False)

    lr_scheduler_group = parser.add_argument_group(
        'Logging & stopping criteria')
    lr_scheduler_group.add_argument('--lr_time_based_decay', action='store_true',
                                    help='Enable time based decay learning rate',
                                    default=False)

    lr_scheduler_group.add_argument('--lr_exp_decay', action='store_true',
                                    help='Enable exponential decay learning rate',
                                    default=False)
    lr_scheduler_group.add_argument('--exp_decay_k',
                                    type=float,
                                    help='Exponential decay exponent rate',
                                    default=0.1)
    args = parser.parse_args()

    # data loading is handled here depending on the number of individuals
    # the loader will also handle the data splitting process according
    # to the arguments provided
    loader = Loader(path=args.path)
    model_storage = ModelStorage(args.path)

    if args.reload_state:
        pos, files = loader.load(args.files)
        inputs, outputs = loader.prepare(pos, args)
        td, tv, tt = loader.split_to_sets(inputs, outputs, args)

        # model storage instance to tidy up the directory and take care of saving/loading
        model_storage.save_sets(td, tv, tt)
    else:
        td, tv, tt = loader.load_from_sets()

    model_factory = ModelFactory()

    # model can be loaded from file to continue training from snapshot
    init_epoch = 0
    if args.load:
        import os

        basename = os.path.basename(args.load)
        if basename == 'latest':
            model_files = glob(os.path.dirname(args.load) + '/model_*.h5')

            def epoch_checkpoint(key): return int(
                key.split('_')[-1].split('.')[0])
            model_files.sort(key=epoch_checkpoint, reverse=True)
            args.load = model_files[0]

        model = model_storage.load_model(
            args.load, model_factory.model_backend(args.model), args)
        init_epoch = int(os.path.basename(
            args.load).split('_')[-1].split('.')[0]) + 1
    else:
        if 'LSTM' in args.model:
            model = model_factory(
                model_choice=args.model,
                input_shape=(args.num_timesteps, td[0].shape[2]),
                output_shape=td[1].shape[1],
                args=args,
            )
        else:
            model = model_factory(
                model_choice=args.model,
                input_shape=(td[0].shape[1],),
                output_shape=td[1].shape[1],
                args=args,
            )

    if model_factory.model_backend(args.model) == 'keras':
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LearningRateScheduler

        model.summary()

        callbacks = []

        if args.custom_logs:
            callbacks.append(
                CSVLogger(
                    model_storage.get_logs_path() + '/history.csv', separator=',', append=True)
            )

        if args.model_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_storage.get_checkpoint_path() + '/best_model.h5',
                    monitor='loss',
                    save_weights_only=False,
                    save_best_only=True))

        if args.early_stopping:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=1))

        if args.enable_tensorboard:
            callbacks.append(TensorBoard(
                log_dir=model_storage.get_logs_path(),
                histogram_freq=1000,
                write_graph=False,
                write_images=False,
                update_freq="epoch",
                profile_batch=5))

        if args.lr_time_based_decay:
            starting_lr = args.learning_rate
            decay = starting_lr / args.epochs

            def lr_time_based_decay(epoch, lr):
                return lr * 1 / (1 + decay * epoch)
            callbacks.append(LearningRateScheduler(
                lr_time_based_decay, verbose=1))
        elif args.lr_exp_decay:
            import math
            starting_lr = args.learning_rate
            k = args.exp_decay_k

            def lr_exp_decay(epoch, lr):
                return starting_lr * math.exp(-k * epoch)
            callbacks.append(LearningRateScheduler(lr_exp_decay, verbose=1))

        # for epoch in range(init_epoch, args.epochs):
        #     _ = model.fit(td[0], td[1],
        #                   validation_data=(tv[0], tv[1]),
        #                   batch_size=args.batch_size,
        #                   epochs=epoch + 1,
        #                   initial_epoch=epoch,
        #                   callbacks=callbacks,
        #                   verbose=args.verbose)

        _ = model.fit(td[0], td[1],
                      validation_data=(tv[0], tv[1]),
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      initial_epoch=init_epoch,
                      callbacks=callbacks,
                      verbose=args.verbose)

        model_storage.save_model(
            model, model_factory.model_backend(args.model), args, epoch)
