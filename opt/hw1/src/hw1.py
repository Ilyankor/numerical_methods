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
    axes[0].vlines(range(n), 0.0, x[:, -1])
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

    # print result
    print(f"f_best: {fk_best[-1]:.16f}")

    return None


def hw1p7() -> None:
    m, n = 20, 200      # A in mxn
    niter = 2500        # num iterations

    # open the information
    with open(Path("./out/7/c.dat"), mode="rb") as f:
        c = np.fromfile(f, dtype=np.float64)

    with open(Path("./out/7/A.dat"), mode="rb") as f:
        data = np.fromfile(f, dtype=np.float64)
    A = data.reshape((m, n), order='F')

    with open(Path("./out/7/b.dat"), mode="rb") as f:
        b = np.fromfile(f, dtype=np.float64)

    with open(Path("./out/7/x.dat"), mode="rb") as f:
        data = np.fromfile(f, dtype=np.float64)
    x = data.reshape((n, niter+1), order='F')

    # create variables
    axb = np.array([np.max(A @ x[:, i] - b) for i in range(niter+1)])
    cx = c @ x

    # plot the results
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    # max(ai*xk - bi)
    axes[0].plot(axb)
    axes[0].set_title(r"constraint")
    axes[0].set_xlabel(r"$k$")
    axes[0].set_ylabel(r"$\max_i{\left(\langle a_i, x^{(k)} \rangle - b_i \right)}$")

    # c*x
    axes[1].plot(cx)
    axes[1].set_title(r"objective")
    axes[1].set_xlabel(r"$k$")
    axes[1].set_ylabel(r"$\langle c, x^{(k)} \rangle$")

    # save image
    plt.savefig(Path("out/7/hw1p7.pdf"), format="pdf")

    # print result
    print(f"c*x: {cx[-1]:.16f}")

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
