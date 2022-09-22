#!/usr/bin/env python
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert idtrackerai .npy to txt files')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the .npy file',
                        required=True)
    args = parser.parse_args()
    
    data = np.load(args.path, allow_pickle=True).item()
    traj = data['trajectories']
    traj = traj.reshape(-1, traj.shape[1] * traj.shape[2])
    np.savetxt(args.path.replace('.npy', '.txt'), traj)