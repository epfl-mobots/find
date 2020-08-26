#!/usr/bin/env python

import tqdm
import argparse
from pathlib import Path

from utils.features import Velocities
from utils.losses import *

import tensorflow as tf


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def cart_sim(model, setup, args):
    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    timestep = args.timestep
    for n in range(ref_positions.shape[1] // 2):
        pos_t = np.roll(ref_positions, shift=1, axis=0)[2:, :]
        pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[1:-1, :]
        vel_t = (ref_positions - np.roll(ref_positions,
                                         shift=1, axis=0))[2:, :] / timestep
        vel_t_1 = (ref_positions - np.roll(ref_positions,
                                           shift=1, axis=0))[1:-1, :] / timestep

        X = np.array([pos_t_1[:, 0], pos_t_1[:, 1],
                      vel_t_1[:, 0], vel_t_1[:, 1]])
        Y = np.array([vel_t[:, 0], vel_t[:, 1]])
        if inputs is None:
            inputs = X
            outputs = Y
        else:
            inputs = np.append(inputs, X, axis=1)
            outputs = np.append(outputs, Y, axis=1)
    X = X.transpose()
    Y = Y.transpose()

    sigmas = []

    generated_data = np.matrix([X[0, 0], X[0, 1]])

    if args.iterations < 0:
        iters = ref_positions.shape[0]
    else:
        iters = args.iterations

    for t in tqdm.tqdm(range(iters)):
        if t == 0:
            prediction = np.array(model.predict(X[0].reshape(1, X.shape[1])))
        else:
            dvel_t = (generated_data[-1, :] -
                      generated_data[-2, :]) / args.timestep
            nninput = np.array(
                [generated_data[-1, 0], generated_data[-1, 1], dvel_t[0, 0], dvel_t[0, 1]]).transpose()
            prediction = np.array(model.predict(
                nninput.reshape(1, X.shape[1])))


        def logbound(val, max_logvar=0, min_logvar=-10):
            logsigma = max_logvar - \
                np.log(np.exp(max_logvar - val) + 1)
            logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
            return logsigma

        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        failed = 0
        while True:
            sample_velx = np.random.normal(
                prediction[0, 0], prediction[0, 2], 1)[0]
            sample_vely = np.random.normal(
                prediction[0, 1], prediction[0, 3], 1)[0]

            x_hat = generated_data[-1, 0] + sample_velx * args.timestep
            y_hat = generated_data[-1, 1] + sample_vely * args.timestep

            r = np.sqrt(
                (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

            rv = np.sqrt(sample_velx ** 2 +
                         sample_vely ** 2 -
                         2 * np.abs(sample_velx) * np.abs(sample_vely) * np.cos(np.arctan2(sample_vely, sample_velx)))

            if setup.is_valid(r) and rv <= 1.2:
                generated_data = np.vstack([generated_data, [x_hat, y_hat]])
                sigmas.append(prediction[0, 2:])
                break
            else:
                rold = np.sqrt((generated_data[-1, 0] - setup.center()[0]) ** 2 + (
                    generated_data[-1, 1] - setup.center()[1]) ** 2)

                failed += 1
                if failed > 999:
                    prediction[:, 2] += 0.01
                    prediction[:, 3] += 0.01

    gp_fname = args.reference.replace('processed', 'generated')
    sigma_fname = gp_fname.replace('positions', 'sigmas')
    gv_fname = gp_fname.replace('positions', 'velocities')
    gv = Velocities([np.array(generated_data)], args.timestep).get()

    np.savetxt(gp_fname, generated_data)
    np.savetxt(gv_fname, gv[0])
    np.savetxt(sigma_fname, np.array(sigmas))



def polar_sim(model, setup, args):
    inputs = None
    outputs = None
    p = np.loadtxt(args.reference)
    timestep = args.timestep

    pos_t = np.roll(p, shift=1, axis=0)[2:, :]
    rad_t = np.sqrt( (pos_t[:, 0] - setup.center()[0]) ** 2 + (pos_t[:, 1] - setup.center()[1]) ** 2)

    pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
    rad_t_1 = np.sqrt( (pos_t_1[:, 0] - setup.center()[0]) ** 2 + (pos_t_1[:, 1] - setup.center()[1]) ** 2)

    vel_t = (p - np.roll(p, shift=1, axis=0))[2:, :] / timestep
    radial_vel_t = (pos_t[:, 0] * vel_t[:, 1] - pos_t[:, 1] * vel_t[:, 0]) / (pos_t[:, 0] ** 2 + pos_t[:, 1] ** 2)
    hdg_t = np.array(list(map(angle_to_pipi, np.arctan2(vel_t[:, 1], vel_t[:, 0]))))

    vel_t_1 = (p - np.roll(p, shift=1, axis=0))[1:-1, :] / timestep
    radial_vel_t_1 = (pos_t_1[:, 0] * vel_t_1[:, 1] - pos_t_1[:, 1] * vel_t_1[:, 0]) / (pos_t_1[:, 0] ** 2 + pos_t_1[:, 1] ** 2)
    hdg_t_1 = np.array(list(map(angle_to_pipi, np.arctan2(vel_t_1[:, 1], vel_t_1[:, 0]))))

    X = np.array([rad_t_1, np.cos(hdg_t_1), np.sin(hdg_t_1), vel_t_1[:, 0], vel_t_1[:, 1]])
    Y = np.array([(rad_t-rad_t_1) / timestep, radial_vel_t])
    inputs = X
    outputs = Y

    X = X.transpose()
    Y = Y.transpose()


    if args.iterations < 0:
        iters = p.shape[0]
    else:
        iters = args.iterations


    generated_pos = np.matrix([pos_t_1[0, 0], pos_t_1[0, 1]])    

    for t in tqdm.tqdm(range(iters)):
        if t == 0:
            prediction = np.array(model.predict(X[0].reshape(1, X.shape[1])))
        else:
            vel = (generated_pos[-1, :] - generated_pos[-2, :]) / args.timestep
            rad = np.sqrt( (generated_pos[-1, 0] - setup.center()[0]) ** 2 + (generated_pos[-1, 1] - setup.center()[1]) ** 2)
            hdg = angle_to_pipi(np.arctan2(vel[0, 1], vel[0, 0]))
            nninput = np.array([rad, np.cos(hdg), np.sin(hdg), vel[0, 0], vel[0, 1]]).transpose()
            prediction = np.array(model.predict(nninput.reshape(1, X.shape[1])))


        def logbound(val, max_logvar=0, min_logvar=-10):
            logsigma = max_logvar - \
                np.log(np.exp(max_logvar - val) + 1)
            logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
            return logsigma

        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        failed = 0
        while True:
            sample_velx = np.random.normal(
                prediction[0, 0], prediction[0, 2], 1)[0]
            sample_vely = np.random.normal(
                prediction[0, 1], prediction[0, 3], 1)[0]

            x_hat = generated_pos[-1, 0] + sample_velx * args.timestep
            y_hat = generated_pos[-1, 1] + sample_vely * args.timestep

            r = np.sqrt((x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

            rv = np.sqrt(sample_velx ** 2 + sample_vely ** 2 -
                        2 * np.abs(sample_velx) * np.abs(sample_vely) * np.cos(np.arctan2(sample_vely, sample_velx)))

            if setup.is_valid(r) and rv <= 1.2:
                generated_pos = np.vstack([generated_pos, [x_hat, y_hat]])
                break
            else:
                rold = np.sqrt((generated_pos[-1, 0] - setup.center()[0]) ** 2 + (
                    generated_pos[-1, 1] - setup.center()[1]) ** 2)

                failed += 1
                if failed > 999:
                    prediction[:, 2] += 0.01
                    prediction[:, 3] += 0.01


    gp_fname = args.reference.replace('processed', 'generated')
    gv_fname = gp_fname.replace('positions', 'velocities')
    gv = Velocities([np.array(generated_pos)], args.timestep).get()

    np.savetxt(gp_fname, generated_pos)
    np.savetxt(gv_fname, gv[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Probabilistic model to reproduce fish motion')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--reference', '-r', type=str,
                        help='Path to a reference experiment position file',
                        required=True)
    parser.add_argument('--model', '-m', type=str,
                        help='Model file name to use',
                        required=True)
    parser.add_argument('--iterations', '-i', type=int,
                        help='Number of iteration of the simulation',
                        required=False,
                        default=-1)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--polar', action='store_true',
                        help='Use polar inputs instead of cartesian coordinates',
                        default=False)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()


    if not args.polar:
        cart_sim(model, setup, args)
    else:
        polar_sim(model, setup, args)