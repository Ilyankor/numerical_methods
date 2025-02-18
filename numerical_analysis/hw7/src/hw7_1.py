import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable


# graph a single variable function over an interval [a, b]
def graph(func:Callable[[float], float], a:float, b:float, n_val:int) -> None:
    # ensure that a < b
    if a > b:
        a, b = b, a
    
    # create x and y values
    x = np.linspace(a, b, num=n_val)
    y = [func(x_i) for x_i in x]

    # plot
    plt.rcParams["text.usetex"] = True
    plt.plot(x, y, linewidth=2)
    plt.title(r"$y = f(x)$", fontsize=15)
    plt.xlabel(r"$x$", fontsize=13)
    plt.ylabel(r"$y$", fontsize=13)
    plt.grid()
    plt.savefig(Path("assets/graph_7_1.svg"), format="svg")
    plt.show()

    # save image

    return None


# bisection method
def bisection(func:Callable[[float], float], a:float, b:float, tol:float, n_max:int) -> tuple[float, float, int]:
    # ensure that the interval is valid
    if func(a)*func(b) > 0:
        sys.tracebacklimit = 0
        raise Exception("The function may not change sign within the interval.")
    
    # ensure that a < b
    if a > b:
        a, b = b, a
    
    i = 0 # iteration counter
    x = 0.5 * (a + b) # midpoint
    f_val = func(x) # f(x)

    for i in range(1, n_max+1):
        # check for tolerance
        abs_f_val = np.abs(f_val)
        if abs_f_val < tol:
            break

        if f_val * func(a) > 0: # root is in [x, b]
            a = x
        else: # root is in [a, x]
            b = x

        x = 0.5 * (a + b) # midpoint
        f_val = func(x) # f(x)

    # did not converge within n_max
    if abs_f_val > tol:
        sys.tracebacklimit = 0
        raise Exception(f"The bisection method did not converge within {n_max} iterations.")

    return x, abs_f_val, i-1


# main function
def hw7_1_main(n_val:int, a:float, b:float, tol:float, n_max:int) -> None:

    # f(x) = x/2 - sin(x) + pi/6 - sqrt(3)/2
    f = lambda x: 0.5*x - np.sin(x) + np.pi / 6 - 0.5*np.sqrt(3)
    
    # main loop
    while True:
        # prompt part selection
        part = input("\nEnter an A or C to select the part.\nType an E to exit to question selection.\n")

        if part.lower() == "e":
            break
        
        # part A
        elif part.lower() == "a":

            print("\nPart A\n")

            # graph the function
            graph(f, a, b, n_val)

        # part C
        elif part.lower() == "c":
            
            # bisection method
            result = bisection(f, a, b, tol, n_max)
            print(
f"""
Performed the bisection method for f on the interval [{a}, {b}].
The root is approximately {result[0]}.
The error is {result[1]}.
The method took {result[2]} iterations to be within tolerance {tol}.
"""
            )

        else:
            print("Invalid part entry. Try again.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
