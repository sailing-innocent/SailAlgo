import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import linalg

import itertools
color_iter = itertools.cycle(["navy", "darkorange","c", "gold","cornflowerblue"])

def plot_cam(ax, pos, lookat, color='gray', near=0.5, far=10, fov=60*np.pi/180):
    cam_dir = lookat - pos
    # normalize the camera direction
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    # right is perpendicular to the camera direction
    cam_right = np.array([cam_dir[1], -cam_dir[0]])
    # draw a triangle patch for the camera
    cam = mpl.patches.Polygon(
        np.array(
            [
                pos,
                pos + near * cam_dir + near * np.tan(fov / 2) * cam_right,
                pos + near * cam_dir - near * np.tan(fov / 2) * cam_right,
            ]
        ),
        closed=True,
        edgecolor="black",
        facecolor=color,
        alpha=0.5,
    )
    ax.add_artist(cam)

    # draw dashed lines for camera culling area
    cam_far = 10
    cam_cull = mpl.patches.Polygon(
        np.array(
            [
                pos,
                pos + cam_far * cam_dir + cam_far * np.tan(fov / 2) * cam_right,
                pos + cam_far * cam_dir - cam_far * np.tan(fov / 2) * cam_right,
            ]
        ),
        closed=True,
        edgecolor="none",
        facecolor=color,
        alpha=0.1,
    )
    ax.add_artist(cam_cull)

def plot_gmm_proj(means, covariances):
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-6.0, 6.0)
    
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_x.axvspan(-1, 1, facecolor='gray', alpha=0.3)
    ax_y.axhspan(-1, 1, facecolor='gray', alpha=0.3)

    plot_cam(ax, np.array([0,5]), np.array([0,0]))
    plot_cam(ax, np.array([5,0]), np.array([0,0]), color='red')

    N = 100
    X_x_sum = np.zeros_like(np.linspace(-6, 6, N))
    Y_x_sum = np.zeros_like(np.linspace(-6, 6, N))
    X_y_sum = np.zeros_like(np.linspace(-6, 6, N))
    Y_y_sum = np.zeros_like(np.linspace(-6, 6, N))

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
        # specially mark mean coordinate
        ax.scatter(mean[0], mean[1], color=color, s=100, marker='o')


        gs_x_mu = mean[0] * 1.0 / (5.0 - mean[1])
        gs_y_mu = mean[1] * 1.0 / (5.0 - mean[0])
        gs_x_sigma = v[0] * 1.0 / (5.0 - mean[1])
        gs_y_sigma = v[1] * 1.0 / (5.0 - mean[0])

    
        X = np.linspace(-6, 6, N)
        Y = np.exp(-(X - gs_x_mu) ** 2 / (2 * gs_x_sigma ** 2)) / np.sqrt(2 * np.pi * gs_x_sigma ** 2)
        ax_x.plot(X, Y, label=f"mu={gs_x_mu}, sigma={gs_x_sigma}", color=color)
        # X_x_sum += X
        # Y_x_sum += Y

        # plot y
        Y = np.linspace(-6, 6, 100)
        X = np.exp(-(Y - gs_y_mu) ** 2 / (2 * gs_y_sigma ** 2)) / np.sqrt(2 * np.pi * gs_y_sigma ** 2)
        ax_y.plot(X, Y, label=f"mu={gs_y_mu}, sigma={gs_y_sigma}", color=color)

        # X_y_sum += X
        # Y_y_sum += Y
    
    # ax_x.plot(X_x_sum, Y_x_sum, label=f"sum of gaussians")
    # ax_y.plot(X_y_sum, Y_y_sum, label=f"sum of gaussians")

def draw(name: str, outdir: str):
    means = np.array([[0.5, 0.5], [-0.5, -0.5],[2.0, -0.2]])
    covariances = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 3.0]]])
    plot_gmm_proj(means, covariances)
    plt.savefig(f"{outdir}/{name}.png")

    return True 