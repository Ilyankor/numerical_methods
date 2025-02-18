import numpy as np
import argparse
from pathlib import Path


# Thomas algorithm
def thomas(tridiagonal:tuple[np.ndarray], rhs:np.ndarray) -> tuple[np.ndarray]:
    # size of system
    n = rhs.shape[0]

    # diagonal, lower diagonal, upper diagonal
    a, b, c = tridiagonal

    # initialize vectors
    alpha = np.zeros(n)
    beta = np.zeros(n-1)
    y = np.zeros(n)
    x = np.zeros(n)

    # factorization algorithm
    alpha[0] = a[0]
    for i in range(n-1):
        beta[i] = b[i] / alpha[i]
        alpha[i+1] = a[i+1] - beta[i] * c[i]

    # substitution algorithms
    # Ly = rhs
    y[0] = rhs[0]
    for i in range(1, n):
        y[i] = rhs[i] - beta[i-1] * y[i-1]

    # Ux = y
    x[-1] = y[-1] / alpha[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - c[i] * x[i+1]) / alpha[i]

    # solution, diagonal of U, lower diagonal of L
    return x, alpha, beta


# homework 2 question 3
def hw2_3_main(tridiagonal:tuple[np.ndarray], rhs:np.ndarray) -> None:

    # perform the thomas algorithm
    sol, U_diag, L_lower = thomas(tridiagonal, rhs)

    # display results
    print(f"\nThe diagonal of U is: \n{U_diag}")
    print(f"\nThe lower diagonal of L is: \n{L_lower}")
    print(f"\nThe solution is: \n{sol}")

    return None

# if the script is run from command line
if __name__ == "__main__":
    
    # command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="The path to the input file")
    args = parser.parse_args()
    input = args.input_file

    # extract the information from an input file
    with open(Path(input), mode="r") as file:
        lines = [x.rstrip() for x in file]
    
    # convert input to proper data
    data = [np.fromstring(x, sep=",") for x in lines]

    # separate the data for the thomas algorithm
    matrix = tuple(data[0:-1])
    rhs = data[-1]

    # run the Thomas algorithm
    sol, U_diag, L_lower = thomas(matrix, rhs)

    # output the resulting information
    with open(Path("output2_3.txt"), mode="w") as out_file:
        out_file.write("\n".join(", ".join(map(str, x)) for x in (sol, U_diag, L_lower)))

    ### input file structure ###
    # diagonal
    # lower diagonal
    # upper diagonal
    # rhs

    ### output file structure ###
    # solution
    # diagonal of U
    # lower diagonal of L
