import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# construction of matrix A
def construct_A(eps:float, n:int) -> np.ndarray:
    d_eps = eps*np.ones(n-1)
    d_eps2 = (eps**2)*np.ones(n-2)
    A = np.eye(n) + np.diag(d_eps, k=1) + np.diag(d_eps2, k=2) + np.diag(d_eps, k=-1) + np.diag(d_eps2, k=-2)

    return A


# construction of iteration matrices
def iteration_matrix(A:np.ndarray, type:str) -> np.ndarray:

    # Jacobi iteration matrix
    if type == "jacobi":
            n = np.shape(A)[0] # size of the array
            D_1 = np.diag(1 / np.diag(A)) # D^-1
            M = np.eye(n) - np.linalg.matmul(D_1, A) # I - D^-1 * A
    
    # Gauss-Seidel iteration matrix
    elif type == "gauss-seidel":
            D = np.diag(np.diag(A)) # diagonal matrix
            E = -1.0 * np.tril(A, k=-1) # lower triangular w/o diag
            M = np.linalg.solve(D-E, D-E-A) # (D-E)^-1 * (D-E-A)

    return M


# spectral radius
def spectral_radius(A:np.ndarray) -> float:
    return np.max(np.absolute(np.linalg.eigvals(A)))


# stationary Richardson method
def richardson(A:np.ndarray, b:np.ndarray, x0:np.ndarray, P:np.ndarray, inverse:bool, tol:float, max_iter:int, track_iter:bool, alpha:float) -> np.ndarray:
    # initialize method
    x = x0 # solution
    r = b - np.linalg.matmul(A, x) # residual
    crit_denom = np.linalg.norm(r) # stopping criteria denominator

    # Richardson method
    for i in range(1, max_iter+1):

        if inverse == True:
            z = np.linalg.matmul(P, r) # direction
        else:
            z = np.linalg.solve(P, r)

        x = x + alpha * z # new solution
        r = r - alpha * np.linalg.matmul(A, z) # new residual

        crit = np.linalg.norm(r) / crit_denom # stopping criteria

        # stopping criteria
        if crit < tol:
            break
    
    # output the results
    if crit > tol:
        return "\nThe Richardson method did not converge within the maximum number of iterations."
    elif track_iter == True:
        return {"solution": x, "iterations": i}
    else:
        return {"solution": x}


# main function
def hw4_1_main(n:int, b:np.ndarray, x0:np.ndarray, tol:float, max_iter:int, start:float, end:float, num_val:int, eps:float) -> None:

    # epsilon values
    eps_val = np.linspace(start, end, num_val)

    while True:
        # prompt part selection
        part = input("\nEnter an A or B to select the part.\nType an E to exit to question selection.\n")

        if part.lower() == "e":
            break
        
        # part A
        elif part.lower() == "a":

            print("\nPart A\n")
            # initialize variables
            spec_jacobi = np.zeros(num_val)
            spec_gauss_seidel = np.zeros(num_val)

            for i in range(num_val):
                # construct A
                A = construct_A(eps_val[i], n)

                # compute spectral values for Jacobi and Gauss-Seidel 
                spec_jacobi[i] = spectral_radius(iteration_matrix(A, "jacobi"))
                spec_gauss_seidel[i] = spectral_radius(iteration_matrix(A, "gauss-seidel"))
            
            # visualize the spectral radii
            plt.rcParams['text.usetex'] = True
            plt.plot(eps_val, spec_jacobi, marker='o', linewidth=1.5)
            plt.plot(eps_val, spec_gauss_seidel, marker='o', linewidth=1.5)

            # plotting design
            plt.title(r"$\varepsilon$ and spectral radius", fontsize=14)
            plt.xlabel(r"$\varepsilon$", fontsize=12)
            plt.ylabel(r"$\rho(B)$", fontsize=12)
            plt.legend(["Jacobi", "Gauss-Seidel"])
            plt.savefig(Path("assets/graph4_1a.svg"))
            plt.show()

        # part B
        elif part.lower() == "b":

            print("\nPart B\n")
            
            # construct A
            A = construct_A(eps, n)

            # Gauss-Seidel P matrix
            D = np.diag(np.diag(A)) # diagonal matrix
            E = -1.0 * np.tril(A, k=-1) # lower triangular w/o diag

            # Gauss-Seidel method
            result = richardson(A, b, x0, D-E, False, tol, max_iter, True, 1.0)
            print(f"The solution is {result["solution"]}.")
            print(f"It took {result["iterations"]} iterations to converge.")
        else:
            print("Invalid part entry. Try again.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
