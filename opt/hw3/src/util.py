import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray


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


def visualize(
        question: int,
        arr: dict[str, NDArray[np.float64]],
        show: bool = True,
        save: bool = True
    ) -> None:

    """
    Visualizes the results from problems 2 and 7.

    Args:
        question: Problem number, should be 2 or 7.
        arr: A dictionary of arrays.
        show: Displays the figure if True.
        save: Saves the figure in /out/{question} if True.
    """

    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 12,
        "lines.linewidth": 2
    })

    match question:

        # visualize solutions for problem 2
        case 2:

            # solution with Euler's method
            plt.plot(arr["t_euler"], arr["sol_euler"], label=[r"$p_1(t)$", r"$p_2(t)$"])
    
            plt.title("Predator prey model with Euler's method")
            plt.xlabel(r"$t$")
            plt.ylabel(r"$p(t)$")

            plt.legend()
            plt.grid(True)

            # save the plot
            if save:
                filepath = Path("out/2/euler.pdf")
                plt.savefig(filepath, format="pdf")
            
            # show the plot
            if show:
                plt.show()


            # solution with RK4 and comparison
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle("Predator prey model with Runge-Kutta methods")

            axs[0].plot(arr["t_rk4"], arr["sol_rk4"], label=[r"$p_1(t)$", r"$p_2(t)$"])
            axs[0].set_title("4th order Runge-Kutta")
            axs[0].set_xlabel(r"$t$")
            axs[0].set_ylabel(r"$p(t)$")

            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(arr["t_rk45"], arr["sol_rk45"], label=[r"$p_1(t)$", r"$p_2(t)$"])
            axs[1].set_title("SciPy 4th order embedded Runge-Kutta")
            axs[1].set_xlabel(r"$t$")
            axs[1].set_ylabel(r"$p(t)$")

            axs[1].legend()
            axs[1].grid(True)

            # save the plot
            if save:
                filepath = Path("out/2/rk4.pdf")
                plt.savefig(filepath, format="pdf")
            
            # show the plot
            if show:
                plt.show()


        # visualize solution for problem 7
        case 7:
            
            plt.plot(arr["t"], arr["sol"], label=[r"$s(t)$", r"$i(t)$", r"$r(t)$"])

            plt.title("SIR model with Heun's method")
            plt.xlabel(r"$t$")
            plt.ylabel(r"Population")

            plt.legend()
            plt.grid(True)

            # save the plot
            if save:
                filepath = Path("out/7/plot.pdf")
                plt.savefig(filepath, format="pdf")
            
            # show the plot
            if show:
                plt.show()
            
        # raise exception if not 2 nor 7
        case _:
            raise ValueError("Question number must be 2 or 7.")

    return None


if __name__ == "__main__":
    exit(0)
