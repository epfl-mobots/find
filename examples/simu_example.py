#!/usr/bin/env python
import sys
import argparse
import numpy as np
from utils.features import Velocities

sys.path.append('.')

from simulation.fish_simulation import FishSimulation
from simulation.replay_individual import ReplayIndividual
from simulation.position_stat import PositionStat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        help='Path to a trajectory file',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        default=0.12, 
                        required=False)
    args = parser.parse_args()

    # loading pre-recorded trajectories
    trajectories = np.loadtxt(args.path)
    num_timesteps = trajectories.shape[0]
    print('Simulation will run for', num_timesteps, 'timesteps')

    # initializing the simulation
    simu_args = {
        'stats_enabled': True,
    }
    simu = FishSimulation(args.timestep, num_timesteps, args=simu_args)
    
    # adding individuals to the simulation
    for i in range(trajectories.shape[1] // 2):
        p = trajectories[:, (i * 2) : (i * 2 + 2)]
        v = Velocities([p], args.timestep).get()[0]
        simu.add_individual(ReplayIndividual(p, v))

    # adding stat objects
    simu.add_stat(PositionStat(trajectories.shape[1], 'positions.dat', simu.get_dirname()))

    # run simulation
    simu.spin()
