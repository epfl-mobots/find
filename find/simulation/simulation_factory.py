import os

import find.simulation.trajnet_functors as tnf

import find.simulation.tf_nn_functors as tfnnf
from find.simulation.tf_nn_functors import get_most_influential_individual
from find.simulation.fish_simulation import FishSimulation
from find.simulation.replay_individual import ReplayIndividual
from find.simulation.nn_individual import NNIndividual
from find.simulation.tf_nn_functors import Multi_plstm_predict, Multi_plstm_predict_traj, closest_individual
from find.simulation.position_stat import PositionStat
from find.simulation.velocity_stat import VelocityStat
from find.simulation.nn_prediction_stat import NNPredictionStat

from find.utils.features import Velocities

nn_functor_choices = {
    'PLSTM': tfnnf.Multi_plstm_predict,
    'PLSTM_SHALLOW': tfnnf.Multi_plstm_predict,
    'PLSTM_2L': tfnnf.Multi_plstm_predict,
    'PLSTM_MULT_PREDS': tfnnf.Multi_plstm_predict_traj,
    'PFW': tfnnf.Multi_pfw_predict,
    'LCONV': tfnnf.Multi_plstm_predict,

    'trajnet_dir': tnf.Trajnet_dir,
}


def available_functors():
    return list(nn_functor_choices.keys())


class SimulationFactory:
    def __call__(self, data, model, nn_functor, backend, args):
        if not os.path.exists(args.simu_out_dir):
            os.makedirs(args.simu_out_dir)

        if backend == 'keras':
            return self._construct_sim(data, model, nn_functor_choices[nn_functor], backend, args)
        elif backend == 'trajnet':
            return self._construct_sim(data, model, nn_functor_choices[nn_functor], backend, args)

    def _construct_sim(self, data, model, nn_functor, backend, args):
        pos = data
        vel = Velocities([pos], args.timestep).get()[0]

        # initializing the simulation
        # if the trajectory is replayed then -1 makes sure that the last model prediction is not added to the generated file to ensure equal size of moves
        iters = pos.shape[0] - \
            2 * args.num_timesteps - 1 if args.iterations < 0 else args.iterations

        simu_args = {'stats_enabled': True, 'simu_dir_gen': False}
        simu = FishSimulation(args.timestep, iters, args=simu_args)

        interaction_functor = None
        if 'LSTM' in args.nn_functor or 'trajnet' in args.nn_functor:
            interaction_functor = nn_functor(
                model, args.num_timesteps, args=args, num_neighs=(pos.shape[1] // 2 - 1))
        else:
            interaction_functor = nn_functor(
                model, args=args, num_neighs=(pos.shape[1] // 2 - 1))

        if iters - args.num_timesteps <= 0:
            return None

        # adding individuals to the simulation
        for i in range(pos.shape[1] // 2):
            if args.exclude_index > -1:
                if args.exclude_index == i:
                    simu.add_individual(
                        NNIndividual(
                            interaction_functor,
                            initial_pos=pos[:(args.num_timesteps+1),
                                            (i * 2): (i * 2 + 2)],
                            initial_vel=vel[:(args.num_timesteps+1),
                                            (i * 2): (i * 2 + 2)]
                        ))
                else:
                    simu.add_individual(ReplayIndividual(
                        pos[args.num_timesteps:, (i * 2): (i * 2 + 2)],
                        vel[args.num_timesteps:, (i * 2): (i * 2 + 2)]))
            else:  # purely virtual simulation
                simu.add_individual(
                    NNIndividual(
                        interaction_functor,
                        initial_pos=pos[:(args.num_timesteps+1),
                                        (i * 2): (i * 2 + 2)],
                        initial_vel=vel[:(args.num_timesteps+1),
                                        (i * 2): (i * 2 + 2)]
                    ))

        if args.exclude_index < 0:
            for _ in range(args.num_extra_virtu):
                simu.add_individual(
                    NNIndividual(
                        interaction_functor,
                        initial_pos=pos[:(args.num_timesteps+1), 0:2],
                        initial_vel=vel[:(args.num_timesteps+1), 0:2]
                    ))

        # generated files have different names if the simulation is virtual or hybrid
        basename = os.path.basename(args.reference)
        if args.exclude_index < 0:
            gp_fname = args.simu_out_dir + '/' + \
                basename.replace('processed', 'generated_virtu')
        else:
            gp_fname = args.simu_out_dir + '/' + basename.replace(
                'processed', 'idx_' + str(args.exclude_index) + '_generated')
        gv_fname = gp_fname.replace('positions', 'velocities')

        # adding stat objects
        simu.add_stat(PositionStat(pos.shape[1], gp_fname))
        simu.add_stat(VelocityStat(vel.shape[1], gv_fname))

        return simu
