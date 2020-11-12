#!/usr/bin/env python

from simulation.simulation import Simulation


if __name__ == '__main__':
    args = {
        'stats_enabled': True,
    }
    simu = Simulation(1000, args=args)
    simu.spin()
