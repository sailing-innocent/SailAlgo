import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import linalg

import itertools
color_iter = itertools.cycle(["navy", "darkorange", "gold","c", "cornflowerblue"])

def plot_gaussians2d(means, covariances):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        x, y = np.linspace(-6.0, 6.0, 50), np.linspace(-6.0, 6.0, 50)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        # calculate the probability density function
        Z = np.zeros_like(X)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                Z[j, k] = np.exp(-0.5 * np.dot(np.dot((pos[j, k] - mean), np.linalg.inv(covar)), (pos[j, k] - mean).T))
        
        # ax.contour(X, Y, Z, colors=color, alpha=0.5)
        # map the color to z value
        color = plt.cm.viridis(Z / Z.max())
        ax.plot_surface(X, Y, Z, facecolors=color, alpha=0.5)
        # ax.plot_surface(X, Y, Z, color=color, alpha=1.0)

        # ax.scatter(x, y, s=5, color=color)

    plt.xlim(-6.0, 6.0)
    plt.ylim(-6.0, 6.0)
    plt.xticks(())
    plt.yticks(())

def draw(name: str, outdir: str):
    # means = np.array([[0.5, 0.5], [-0.5, -0.5]])
    # covariances = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0], [-1.0, 1.0]]])
    means = np.array([[0.5, 0.5]])
    covariances = np.array([[[4.0, 0.0], [0.0, 4.0]]])
    plot_gaussians2d(means, covariances)
    plt.savefig(f"{outdir}/{name}.png")
    return True 