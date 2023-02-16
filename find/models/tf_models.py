import tensorflow as tf
from find.models.tf_losses import *
import find.models.tf_activations as tfa


def LSTM(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=False,
                                   input_shape=input_shape, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(80, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape, activation=None))
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    return model


def PLSTM(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=False,
                                   input_shape=input_shape, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(25, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def PLSTM_SHALLOW(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=False,
                                   input_shape=input_shape, activation='tanh'))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def PLSTM_2L(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=False,
                                   input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    model.add(tf.keras.layers.Reshape((1, 50)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=False,
                                   activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def PLSTM_MULT_PREDS(input_shape, output_shape, args):
    assert args.prediction_steps > 1

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=False,
                                   input_shape=input_shape, activation='tanh'))
    model.add(tf.keras.layers.Dense(80, activation='tanh'))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    model.add(tf.keras.layers.Dense(80, activation='tanh'))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(
        output_shape * args.prediction_steps * 2, activation=None))
    model.add(tf.keras.layers.Lambda(
        lambda x: tf.reshape(x, shape=(-1, 1, args.prediction_steps, output_shape * 2))))
    model.compile(
        loss=multi_dim_gaussian_nll,
        optimizer=optimizer,
    )
    return model


def PFW(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(100, activation=gaussian))
    model.add(tf.keras.layers.Dense(80, activation='tanh'))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    model.add(tf.keras.layers.Dense(80, activation='tanh'))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    loss = gaussian_nll
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[gaussian_mse, gaussian_mae]
                  )
    return model


def LCONV(input_shape, output_shape, args):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True,
                                   input_shape=input_shape, activation='tanh'))
    model.add(tf.keras.layers.Conv1D(
        128, kernel_size=3, input_shape=(100, 1), padding='causal', activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        64, kernel_size=2, padding='causal', activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        32, kernel_size=1, padding='causal', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(30, activation='tanh'))
    model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def PLSTM_model_builder(input_shape, output_shape, args):
    assert len(args.model_layers) == len(
        args.model_neurons), 'Number of layers and neuron mapping should have the same length'
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()

    for idx, l in enumerate(args.model_layers):
        if args.model_activations[idx] == 'None':
            activation = None
        elif args.model_activations[idx] in list(tfa.activations.keys()):
            activation = tfa.activations[args.model_activations[idx]]
        else:
            activation = args.model_activations[idx]

        if l == 'LSTM':
            if idx == 0:
                model.add(tf.keras.layers.LSTM(args.model_neurons[idx], return_sequences=False,
                                               input_shape=input_shape, activation=activation))
            else:
                model.add(tf.keras.layers.LSTM(args.model_neurons[idx], return_sequences=False,
                                               activation=activation))
        elif l == 'Dense':
            model.add(tf.keras.layers.Dense(
                args.model_neurons[idx], activation=activation))
        elif l == 'Reshape':
            model.add(tf.keras.layers.Reshape((1, args.model_neurons[idx])))
        elif l == 'Dropout':
            model.add(tf.keras.layers.Dropout(
                float(args.model_activations[idx])))
        elif l == 'Dense_out':
            if args.model_neurons[idx] > 0:
                neurons = args.model_neurons[idx]
            else:
                neurons = output_shape * 2
            model.add(tf.keras.layers.Dense(
                neurons, activation=activation))
        elif l == 'Norm':          
             model.add(tf.keras.layers.BatchNormalization())

    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def PFW_model_builder(input_shape, output_shape, args):
    assert len(args.model_layers) == len(
        args.model_neurons), 'Number of layers and neuron mapping should have the same length'
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    for idx, l in enumerate(args.model_layers):
        if args.model_activations[idx] == 'None':
            activation = None
        elif args.model_activations[idx] in list(tfa.activations.keys()):
            activation = tfa.activations[args.model_activations[idx]]
        else:
            activation = args.model_activations[idx]

        if l == 'Dense':
            model.add(tf.keras.layers.Dense(
                args.model_neurons[idx], activation=activation))
        elif l == 'Dropout':
            model.add(tf.keras.layers.Dropout(
                float(activation)))
        elif l == 'Dense_out':
            if args.model_neurons[idx] > 0:
                neurons = args.model_neurons[idx]
            else:
                neurons = output_shape * 2
            model.add(tf.keras.layers.Dense(
                neurons, activation=activation))

    model.compile(
        loss=gaussian_nll,
        optimizer=optimizer,
        metrics=[gaussian_mse, gaussian_mae]
    )
    return model


def EncDec(input_shape, output_shape, args):
    encoder_inputs = tf.keras.Input(shape(None, ))
    
    print('here')
    input()
    # optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.LSTM(32, return_sequences=False,
    #                                input_shape=input_shape, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(25, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(10, activation='tanh'))
    # model.add(tf.keras.layers.Dense(output_shape * 2, activation=None))
    # model.compile(
    #     loss=gaussian_nll,
    #     optimizer=optimizer,
    #     metrics=[gaussian_mse, gaussian_mae]
    # )
    # return model
