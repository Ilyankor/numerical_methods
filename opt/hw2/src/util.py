import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence
from numpy.typing import NDArray


def create_arrays(
        question: int,
        m: int,
        n: int,
        nnz: int,
        noise: float,
        r: int = 4,
        values: Sequence[float] = [-10, 10],
        rng: np.random.Generator = np.random.default_rng()
    ) -> dict[str, NDArray[np.float64]]:
    
    """
    Constructs the arrays for the homework problems.

    Args:
        question: Should be 4 or 5.
        m: Number of rows of A.
        n: Number of columns of A.
        nnz: Number of nonzero entries of a sparse array.
        noise: Variance of the normal distribution for perturbing or creating noise.
        r: Rank of the low rank matrix for question 5.
        values: Values of the sparse matrix for question 5.
        rng: A NumPy random generator.

    Raises:
        ValueError: If question is nither 4 nor 5.

    Returns:
        A dictionary with the arrays.
            If question 4: A, x_true, y.
            If question 5: L, S, V.
    """

    match question:

        # question 4 arrays
        case 4:
            # construct A
            A = rng.standard_normal(size=(m, n))
            A /= np.linalg.norm(A, axis=0)

            # construct x_true
            x_true = np.zeros(n)
            indices = rng.choice(n, size=nnz, replace=False)
            x_true[indices] = rng.standard_normal(size=nnz)

            # construct y
            v = np.sqrt(noise) * rng.standard_normal(size=m)
            y = A @ x_true + v

            return {"A": A, "x": x_true, "y": y}
        
        # question 5 arrays
        case 5: 
            # create L
            L1 = rng.standard_normal(size=(m, r))
            L2 = rng.standard_normal(size=(r, n))
            L = L1 @ L2

            # create S
            S = np.zeros(m * n)
            indices = rng.choice(m * n, size=nnz, replace=False)
            S[indices] = rng.choice(values, size=nnz)
            S = S.reshape((m, n))
            
            # create V
            V = np.sqrt(noise) * rng.standard_normal(size=(m, n))

            return {"L": L, "S": S, "V": V}
        
        # raise exception if not 4 nor 5
        case _:
            raise ValueError("question must be 4 or 5")
        
    return None


def save_arrays(
        arr: dict[str, NDArray[np.float64]],
        dirpath: Path | str
    ) -> None:

    """
    Saves a dictionary of arrays to the directory /{dirpath} as separate files.

    Args:
        arr: A dictionary of arrays with filenames as the keys.
        dirpath: The directory to save the dictionary in
    """

    # ensure the directory exists
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    # save the arrays
    for k, v in arr.items():
        np.save(dirpath / f"{k}.npy", v, allow_pickle=False)
    
    return None


def visualize(question: int, arr: dict[str, NDArray[np.float64]], show: bool = True, save: bool = True) -> None:
    """
    Visualizes the results from problems 4 and 5.

    Args:
        question: Problem number, should be 4 or 5.
        arr: A dictionary with keys as the names for the legend.
        show: Displays the figure if True.
        save: Saves the figure in /out/{question} if True.
    """

    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 12,
        "lines.linewidth": 2
    })

    match question:

        # visualize objective functions for problem 4
        case 4:

            plt.figure(figsize=(8, 4))
            for name, obj in arr.items():
                plt.plot(range(1, len(obj)), obj[1:], label=name)
            
            plt.title("Objective functions")
            plt.xlabel(r"iteration $k$")
            plt.ylabel(r"$\frac{1}{2} \| A x^{(k)} - y \|^2_2 + \alpha \| x^{(k)} \|_1$")

            plt.legend()
            plt.grid(True)

            # save the plot
            if save:
                filepath = Path("out/4/graph.pdf")
                plt.savefig(filepath, format="pdf")
            
            # show the plot
            if show:
                plt.show()
            
        # visualize matrices for problem 5
        case 5:
            
            titles = ["Original Matrices", "Reconstructed Matrices"]
            names = list(arr.keys())
            matrices = list(arr.values())
            
            fig = plt.figure(figsize=(8,4))
            subfigs = fig.subfigures(nrows=2, ncols=1)

            for row, subfig in enumerate(subfigs):
                subfig.suptitle(titles[row])
                axs = subfig.subplots(nrows=1, ncols=3)

                for col, ax in enumerate(axs):
                    idx = row * 3 + col

                    ax.matshow(matrices[idx])
                    ax.set_xlabel(names[idx])

                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()

            # save the plot
            if save:
                filepath = Path("out/5/plot.pdf")
                plt.savefig(filepath, format="pdf")
            
            # show the plot
            if show:
                plt.show()
            
        # raise exception if not 4 nor 5
        case _:
            raise ValueError("question must be 4 or 5")

    return None


if __name__ == "__main__":
    exit(0)
