#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *
# from PIL import Image

# image_path = '../plots/fish_dark.png'
# image = Image.open(image_path)

def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def compute_leadership(positions, velocities):
    ang0 = np.arctan2(positions[:, 1] - positions[:, 3], positions[:, 0] - positions[:, 2])
    ang1 = np.arctan2(positions[:, 3] - positions[:, 1], positions[:, 2] - positions[:, 0])

    ang0 = list(map(angle_to_pipi, ang0))
    ang1 = list(map(angle_to_pipi, ang1))
    angle_list = [ang0, ang1]

    previous_leader = -1
    leader_changes = -1
    leadership_timeseries = []

    for i in range(vel.shape[0]):
        # fig = plt.figure(figsize=(5, 5))
        # ax = plt.gca()
        # outer = plt.Circle(
        #     (0, 0), 1.0, color='black', fill=False)
        # ax.add_artist(outer)

        angles = []
        phis = []

        for inum, j in enumerate(range(positions.shape[1] // 2)):
            x = positions[i, j * 2]
            y = positions[i, j * 2 + 1]

            phi = angle_to_pipi(np.arctan2(velocities[i, j * 2 + 1],
                                velocities[i, j * 2])) 
            phis.append(phi)

            angles.append((angle_to_pipi(phi - angle_list[j][i]) + np.pi) % (2 * np.pi) - np.pi)

            # rotated_img = image.rotate(phi * 180 / np.pi)
            # ax.imshow(rotated_img, extent=[x - 0.06, x + 0.06, y - 0.06, y + 0.06], aspect='equal')
            # plt.text(x + 0.025, y + 0.025, str(j), color='r', fontsize=5)

        geo_leader = np.argmin(angles)
        if geo_leader != previous_leader:
            leader_changes += 1
            previous_leader = geo_leader
        leadership_timeseries.append([i, geo_leader])


        # TODO: this is legacy code for visualising the angles and validating the script is correct
        # it should be removed in the future

        # ax.axis('off')
        # ax.set_xlim([-1.1, 1.1])
        # ax.set_ylim([-1.1, 1.1])
        # plt.tight_layout()

        # angles = list(map(lambda x: abs(x * 180) / np.pi, angles))
        # plt.text(0.9, 0.9, 'Leader: ' + str(geo_leader), color='r', fontsize=5)
        # plt.text(0.6, 0.8, 'Ang0: ' + str((ang0[i] * 180) / np.pi), color='r', fontsize=5)
        # plt.text(0.6, 0.7, 'Ang1: ' + str((ang1[i] * 180) / np.pi), color='r', fontsize=5)
        # plt.text(0.6, 0.6, 'Angle0: ' + str(angles[0]), color='r', fontsize=5)
        # plt.text(0.6, 0.5, 'Angle1: ' + str(angles[1]), color='r', fontsize=5)
        # plt.text(0.6, 0.4, 'Heading0: ' + str((phis[0] * 180) / np.pi), color='r', fontsize=5)
        # plt.text(0.6, 0.3, 'Heading1: ' + str((phis[1] * 180) / np.pi), color='r', fontsize=5)
        # png_fname = str(i).zfill(6)
        # plt.savefig(
        #     str(png_fname) + '.png',
        #     transparent=True,
        #     dpi=300
        # )
        # plt.close('all')

    return (leader_changes, leadership_timeseries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    args = parser.parse_args()

    files = glob.glob(args.path + '/*processed_velocities.dat')

    leader_change_count = 0
    for f in files: 
        vel = np.loadtxt(f)
        pos = np.loadtxt(f.replace('velocities', 'positions'))
        (leader_change_count, leadership_timeseries) = compute_leadership(pos, vel)
        np.savetxt(f.replace('velocities', 'leadership_info'), np.array(leadership_timeseries)) 
        np.savetxt(f.replace('velocities', 'leadership_change_count'), np.array([leader_change_count])) 
