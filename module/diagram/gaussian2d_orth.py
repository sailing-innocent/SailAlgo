import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import linalg

import itertools
color_iter = itertools.cycle(["navy", "darkorange", "gold","c", "cornflowerblue"])

def plot_gmm_proj(means, covariances):
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-6.0, 6.0)
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        ell.set_clip_box(ax.bbox)
        # scale
        ax.add_artist(ell)

        gs_x_mu = mean[0]
        gs_x_sigma = v[0]
        gs_y_mu = mean[1]
        gs_y_sigma = v[1]

        # plot x
        X = np.linspace(-6, 6, 50)
        Y = np.exp(-(X - gs_x_mu) ** 2 / (2 * gs_x_sigma ** 2)) / np.sqrt(2 * np.pi * gs_x_sigma ** 2)
        ax_x.plot(X, Y, label=f"mu={gs_x_mu}, sigma={gs_x_sigma}", color=color)

        # plot y
        Y = np.linspace(-6, 6, 50)
        X = np.exp(-(Y - gs_y_mu) ** 2 / (2 * gs_y_sigma ** 2)) / np.sqrt(2 * np.pi * gs_y_sigma ** 2)
        ax_y.plot(X, Y, label=f"mu={gs_y_mu}, sigma={gs_y_sigma}", color=color)


def draw(name: str, outdir: str):
    means = np.array([[0.5, 0.5], [-0.5, -0.5]])
    covariances = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0], [-1.0, 1.0]]])
    plot_gmm_proj(means, covariances)
    # plt.show()
    plt.savefig(f"{outdir}/{name}.png")
    return True 