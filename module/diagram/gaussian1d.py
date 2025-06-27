import matplotlib.pyplot as plt
import numpy as np 

def draw(name: str, outdir: str):
    plt.figure(name)
    plt.clf()
    X  = np.linspace(-3, 3, 50)

    def plot_gs1d(mu, sigma):
        Y = np.exp(-(X - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
        plt.plot(X, Y, label=f"mu={mu}, sigma={sigma}")

    mus = [0, 0, 0, -1]
    sigmas = [1, 0.5, 2, 1]
    for mu, sigma in zip(mus, sigmas):
        plot_gs1d(mu, sigma)

    plt.ylim(0, 1)
    plt.legend(
        loc="upper left",
        ncol=2,
        shadow=True,
        fancybox=True,
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.savefig(f"{outdir}/{name}.png")

    return True 


