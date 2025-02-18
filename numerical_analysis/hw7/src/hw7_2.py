import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable


# fixed point iteration
def fix_point(func:Callable, x0:float, tol:float, n_max:int, actual:float) -> tuple[float, float, float, list]:
    x = x0 # initial guess
    i = 0  # iteration counter
    abs_errors = [] # initialize list of absolute errors

    for i in range(1, n_max+1):
        x = func(x) # fixed point

        err = np.abs(x - func(x)) # error
        abs_errors.append(np.abs(x - actual)) # absolute error
        if err < tol:
            break
    
    # did not converge within n_max
    if err > tol:
        sys.tracebacklimit = 0
        raise Exception(f"The fixed point iteration did not converge within {n_max} iterations.")

    return x, err, i, abs_errors


# main function
def hw7_2_main(x0:float, tol:float, n_max:int, alpha:float) -> None:

    # f(x) = x - (x^4 - 2)/(4x^3)
    f = lambda x: x - (x**4 - 2.0) / (4.0 * x**3)
    
    # main loop
    while True:
        # prompt part selection
        part = input("\nEnter an B or C to select the part.\nType an E to exit to question selection.\n")

        # fixed point iteration
        result = fix_point(f, x0, tol, n_max, alpha)

        # exit
        if part.lower() == "e":
            break
        
        # part B
        elif part.lower() == "b":

            print("\nPart B\n")

            # print result
            print(
f"""
The fixed point iteration converged in {result[2]} iterations.
The fixed point is approximately {result[0]}.
The error is {result[1]}.
"""
            )

        # part C
        elif part.lower() == "c":
            
            print("\nPart C\n")

            # graph the error
            y = np.array(result[3])
            x = np.arange(y.shape[0])

            plt.rcParams["text.usetex"] = True
            plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
            plt.semilogy(x, y, linewidth=2, marker="o")
            plt.title(r"$\log$ of the absolute error at each iteration", fontsize=15)
            plt.xlabel(r"$i$", fontsize=13)
            plt.ylabel(r"$\log{\left\lvert e^{(k)} \right\rvert}$", fontsize=13)
            plt.grid()
            plt.savefig(Path("assets/graph_7_2.svg"), format="svg")
            plt.show()

        else:
            print("Invalid part entry. Try again.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
