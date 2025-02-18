import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def hw1p6() -> None:
    n = 1000        # x in R^n
    niter = 3000    # num iterations

    # open the results
    with open(Path("./out/6/x.dat"), mode="rb") as f:
        data = np.fromfile(f, dtype=np.float64)
    x = data.reshape((n, niter+1), order='F')

    # objective function |x|_1
    l1_norm = np.sum(np.abs(x), axis=0)
    fk_best = np.minimum.accumulate(l1_norm)

    # plot the results
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    # xk, k=3000
    axes[0].vlines(range(n), 0.0, x[:, niter])
    axes[0].set_title(r"$x^{(3000)}$")
    axes[0].set_xlabel(r"$i$")
    axes[0].set_ylabel(r"$x_i$")

    # fk_best
    axes[1].semilogy(fk_best)
    axes[1].set_title(r"$f^{(k)}_{\mathrm{best}}$")
    axes[1].set_xlabel(r"$k$")
    axes[1].set_ylabel(r"$\| x^{(k)} \|_1$")

    # save image
    plt.savefig(Path("out/6/hw1p6.pdf"), format="pdf")

    return None


def hw1p7() -> None:
    return None


def main() -> None:
    num = sys.argv[1]
    
    if num == "6":
        hw1p6()
    elif num == "7":
        hw1p7()
    else:
        sys.exit()

if __name__ == "__main__":
    main()
