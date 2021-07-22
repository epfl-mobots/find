#!/usr/bin/env python
from find.models.tf_losses import *
import argparse
import sys

import matplotlib.lines as mlines
import seaborn as sns
import tensorflow as tf
from pylab import *

sys.path.append('.')

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
palette = flatui
# palette = 'Paired'
# palette = "husl"
colors = sns.color_palette(palette)
sns.set(style="darkgrid")

gfontsize = 10
params = {
    'axes.labelsize': gfontsize,
    'font.size': gfontsize,
    'legend.fontsize': gfontsize,
    'xtick.labelsize': gfontsize,
    'ytick.labelsize': gfontsize,
    'text.usetex': False,
    # 'figure.figsize': [10, 15]
    # 'ytick.major.pad': 4,
    # 'xtick.major.pad': 4,
    'font.family': 'Arial',
}
rcParams.update(params)

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 1.0]
open_circle = mpl.path.Path(vert)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                  edgecolor='none', linewidth=0)
shapeList = [
    Circle((0, 0), radius=1, facecolor=colors[0]),
    Circle((0, 0), radius=1, facecolor=colors[1]),
    Circle((0, 0), radius=1, facecolor=colors[2]),
    # Circle((0, 0), radius=1, facecolor=colors[3]),
    # Circle((0, 0), radius=1, facecolor=colors[4]),
    # Circle((0, 0), radius=1, facecolor=colors[5]),
]

v = np.r_[circ, circ[::-1] * 0.6]
oc = mpl.path.Path(v)

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=oc,
                  markersize=6, label='Mean and SD'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='*',
                  markersize=5, label='Median'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
                  color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.5,
                  markersize=5, label='Single run')
]
handles_b = [
    mlines.Line2D([0], [1], color='black', label='Mean'),
    Circle((0, 0), radius=1, facecolor='black', alpha=0.35, label='SD')
]


def pplots(data, ax, sub_colors=[], exp_title='', ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    sns.pointplot(data=np.transpose(data), palette=sub_colors,
                  size=5, estimator=np.mean,
                  ci='sd', capsize=0.2, linewidth=0.8, markers=[open_circle],
                  scale=1.6, ax=ax)

    sns.stripplot(data=np.transpose(data), edgecolor='white',
                  dodge=True, jitter=True,
                  alpha=.50, linewidth=0.8, size=5, palette=sub_colors, ax=ax)

    medians = []
    for d in data:
        medians.append([np.median(list(d))])
    sns.swarmplot(data=medians, palette=['#000000'] * 10,
                  marker='*', size=5, ax=ax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dense model to reproduce fish motion')
    parser.add_argument('--epochs', '-e', type=int,
                        help='Number of training epochs',
                        default=10000)
    parser.add_argument('--batch_size', '-b', type=int,
                        help='Batch size',
                        default=256)
    args = parser.parse_args()

    X = np.random.rand(200, 1) * 5
    Y = np.cos(X)

    split_at = X.shape[0] - X.shape[0] // 10
    (x_train, x_val) = X[:split_at, :], X[split_at:, :]
    (y_train, y_val) = Y[:split_at, :], Y[split_at:, :]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(20, activation='tanh'))
    model.add(tf.keras.layers.Dense(Y.shape[1] * 2, activation=None))

    loss = gaussian_nll
    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[gaussian_mse, gaussian_mae]
                  )
    model.summary()

    for epoch in range(args.epochs):
        model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=epoch + 1,
                  initial_epoch=epoch,
                  validation_data=(x_val, y_val),
                  verbose=1)

    model.save('cos_model.h5')

    sigmas = []
    for i in range(X.shape[0]):
        p = model.predict(X[i])
        sigmas.append(np.exp(p[0, 1]))

    np.savetxt('cos_sigmas.dat', np.array(sigmas))

    plt.figure()
    plt.plot(sigmas)
    plt.legend(labels=['sigma'])
    plt.show()

    plt.figure()
    plt.plot(Y[:, 0])
    plt.plot(np.array(model.predict(X))[:, 0])
    plt.legend(labels=['real', 'predicted'])
    plt.show()
