import glob
import numpy as np
from tqdm import tqdm


def mem_pair_cart(pos, args):
    offset = 1
    if args.timesteps_skip > 0:
        offset = args.timesteps_skip

    input_list = []
    output_list = []
    for p in pos:
        inputs = None
        outputs = None

        if args.distance_inputs:
            dist = np.sqrt((p[:, 0] - p[:, 2]) ** 2 +
                           (p[:, 1] - p[:, 3]) ** 2)
            rad = 1 - np.array([
                np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2),
                np.sqrt(p[:, 2] ** 2 + p[:, 3] ** 2)
            ]).T

            zidcs = np.where(rad < 0)
            if len(zidcs[0]) > 0:
                rad[zidcs] = 0

        pos_t_1 = np.roll(p, shift=1, axis=0)[1:-offset, :]
        pos_t = p[offset:-1, :]
        vel_t = (pos_t - pos_t_1) / args.timestep
        vel_t_1 = np.roll(vel_t, shift=1, axis=0)
        pos_t_1 = pos_t_1[1:-1, :]
        vel_t_1 = vel_t_1[1:-1, :]
        pos_t = pos_t[1:-1, :]
        vel_t = vel_t[1:-1, :]

        if args.distance_inputs:
            dist_t_1 = np.roll(dist, shift=1)[2:-(offset+1)]
            rad_t_1 = np.roll(rad, shift=1, axis=0)[2:-(offset+1), :]

        for fidx in range(p.shape[1] // 2):
            X = []
            Y = []

            X.append(pos_t_1[:, fidx * 2])
            X.append(pos_t_1[:, fidx * 2 + 1])
            X.append(vel_t_1[:, fidx * 2])
            X.append(vel_t_1[:, fidx * 2 + 1])
            if args.distance_inputs:
                X.append(rad_t_1[:, fidx])

            Y.append(vel_t[:, fidx * 2] - vel_t_1[:, fidx * 2])
            Y.append(vel_t[:, fidx * 2 + 1] - vel_t_1[:, fidx * 2 + 1])

            for nidx in range(p.shape[1] // 2):
                if fidx == nidx:
                    continue
                X.append(pos_t_1[:, nidx * 2])
                X.append(pos_t_1[:, nidx * 2 + 1])
                X.append(vel_t_1[:, nidx * 2])
                X.append(vel_t_1[:, nidx * 2 + 1])
                if args.distance_inputs:
                    X.append(rad_t_1[:, nidx])

            if args.distance_inputs:
                X.append(dist_t_1)

            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
        input_list.append(inputs.T)
        output_list.append(outputs.T)
    return input_list, output_list


def ready_data(data, args):
    def split(x, y, args):
        X = np.empty([0, args.num_timesteps, x.shape[1]])
        if args.prediction_steps == 1:
            Y = np.empty([0, y.shape[1]])
        else:
            Y = np.empty([0, 1, args.prediction_steps, y.shape[1]])

        iters = 1
        if args.timesteps_skip > 0:
            iters = args.timesteps_skip

        for idxskip in range(iters):
            xh = x[idxskip::(args.timesteps_skip + 1)].copy()
            yh = y[idxskip::(args.timesteps_skip + 1)].copy()

            for i in range(args.num_timesteps, xh.shape[0] - args.prediction_steps):
                inp = xh[(i-args.num_timesteps):i, :].reshape(1,
                                                              args.num_timesteps, xh.shape[1])

                if args.prediction_steps == 1:
                    out = yh[i-1, :]
                else:
                    out = yh[(i-1):(i-1+args.prediction_steps), :].reshape(1,
                                                                           1, args.prediction_steps, yh.shape[1])
                X = np.vstack((X, inp))
                Y = np.vstack((Y, out))
        return X, Y
    return split(*data, args)


def no_mem_pair_cart(pos, args):
    inputs = None
    outputs = None
    for p in tqdm(pos, desc='Loading files'):
        if p.shape[0] < 2 + args.timesteps_skip:
            continue

        offset = 1
        if args.timesteps_skip > 0:
            offset = args.timesteps_skip

        if args.distance_inputs:
            dist = np.sqrt((p[:, 0] - p[:, 2]) ** 2 +
                           (p[:, 1] - p[:, 3]) ** 2)
            rad = 1 - np.array([
                np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2),
                np.sqrt(p[:, 2] ** 2 + p[:, 3] ** 2)
            ]).T

            zidcs = np.where(rad < 0)
            if len(zidcs[0]) > 0:
                rad[zidcs] = 0

        pos_t_1 = np.roll(p, shift=1, axis=0)[1:-offset, :]
        pos_t = p[offset:-1, :]
        vel_t = (pos_t - pos_t_1) / args.timestep
        vel_t_1 = np.roll(vel_t, shift=1, axis=0)
        pos_t_1 = pos_t_1[1:-1, :]
        vel_t_1 = vel_t_1[1:-1, :]
        pos_t = pos_t[1:-1, :]
        vel_t = vel_t[1:-1, :]

        if args.distance_inputs:
            dist_t_1 = np.roll(dist, shift=1)[2:-(offset+1)]
            rad_t_1 = np.roll(rad, shift=1, axis=0)[2:-(offset+1), :]

        for fidx in range(p.shape[1] // 2):
            X = []
            Y = []

            X.append(pos_t_1[:, fidx * 2])
            X.append(pos_t_1[:, fidx * 2 + 1])
            X.append(vel_t_1[:, fidx * 2])
            X.append(vel_t_1[:, fidx * 2 + 1])
            if args.distance_inputs:
                X.append(rad_t_1[:, fidx])

            Y.append(vel_t[:, fidx * 2] - vel_t_1[:, fidx * 2])
            Y.append(vel_t[:, fidx * 2 + 1] - vel_t_1[:, fidx * 2 + 1])

            for nidx in range(p.shape[1] // 2):
                if fidx == nidx:
                    continue
                X.append(pos_t_1[:, nidx * 2])
                X.append(pos_t_1[:, nidx * 2 + 1])
                X.append(vel_t_1[:, nidx * 2])
                X.append(vel_t_1[:, nidx * 2 + 1])
                if args.distance_inputs:
                    X.append(rad_t_1[:, nidx])

            if args.distance_inputs:
                X.append(dist_t_1)

            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
    return inputs, outputs


def split_cart(num_individuals, data, args):
    if num_individuals == 2:
        if 'LSTM' in args.model:
            X_list, Y_list = mem_pair_cart(data, args)
            x_shape = (0, args.num_timesteps, X_list[0].shape[1])
            if args.prediction_steps == 1:
                y_shape = (0, Y_list[0].shape[1])
            else:
                y_shape = (0, 1, args.prediction_steps, Y_list[0].shape[1])
            Xh = np.empty(x_shape)
            Yh = np.empty(y_shape)
            for idx in tqdm(range(len(X_list)), desc='Converting data to LSTM compatible format'):
                Xi = X_list[idx]
                Yi = Y_list[idx]
                (Xi, Yi) = ready_data((Xi, Yi), args)
                if Xi.shape[0] == 0:
                    continue
                Xh = np.vstack((Xh, Xi))
                Yh = np.vstack((Yh, Yi))
            return Xh, Yh
        else:
            X, Y = no_mem_pair_cart(data, args)
            return X.T, Y.T
    return [], []


class Loader:
    def __init__(self, path):
        self._path = path
        self._num_individuals = None

    def prepare(self, data, args):
        assert self._num_individuals is not None
        if not args.polar:
            return split_cart(self._num_individuals, data, args)
        else:
            return [], []  # not implemented

    def split_to_sets(self, inputs, outputs, args):
        assert sum(
            [args.train_fraction, args.val_fraction, args.test_fraction]) == 1, 'Split fractions should add up to 1.0'

        # set lengths
        train_split = int(inputs.shape[0] * args.train_fraction)
        val_split = int(inputs.shape[0] * args.val_fraction)

        # actual data split
        train_inputs = inputs[:train_split]
        train_outputs = outputs[:train_split]

        val_inputs = inputs[train_split:(train_split + val_split)]
        val_outputs = outputs[train_split:(train_split + val_split)]

        test_inputs = inputs[(train_split + val_split):]
        test_outputs = outputs[(train_split + val_split):]

        return (train_inputs, train_outputs), (val_inputs, val_outputs), (test_inputs, test_outputs)

    def load(self, fname):
        files = glob.glob(self._path + '/raw/*' + fname)
        pos = []
        for f in files:
            matrix = np.loadtxt(f)
            pos.append(matrix)
        self._num_individuals = pos[0].shape[1] // 2
        return pos, files
