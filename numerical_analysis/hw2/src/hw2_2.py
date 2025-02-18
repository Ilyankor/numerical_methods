import src.hw2_1
import numpy as np

# LU factorization by Gaussian elimination with pivoting
def lu_pivot_factor(A:np.ndarray) -> tuple[np.ndarray]:
    # size of system
    n = A.shape[0]

    # initialize variables
    P = np.eye(n, dtype=int)
    L = np.eye(n, dtype=float)
    U = A.astype(float)

    # Gaussian elimination method with pivoting
    for k in range(n-1):
        # identify the location of largest entry in the column
        max_pivot = k + np.argmax(np.abs(U[k:, k]))
        
        # switch rows in P, L, U and modify L so 1 stays on diagonal
        P[[k, max_pivot]] = P[[max_pivot, k]]
        L[[k, max_pivot]] = L[[max_pivot, k]]
        L[:, [k, max_pivot]] = L[:, [max_pivot, k]]
        U[[k, max_pivot]] = U[[max_pivot, k]]
        
        # usual Gaussian elimination
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, :] = U[i, :] - L[i, k] * U[k, :]

    return P, L, U


# homework 2 question 2
def hw2_2_main(A:np.ndarray, b:np.ndarray) -> None:

    # LU factorization
    P, L, U = lu_pivot_factor(A)

    # solve (LU)x = Pb
    x = src.hw2_1.lu_solve(L, U, np.linalg.matmul(P, b))
    
    # display results
    np.set_printoptions(suppress=True)
    print(f"\nThe permutation matrix is:\n{P}")
    print(f"\nThe lower triangular matrix is:\n{L}")
    print(f"\nThe upper triangular matrix is:\n{U}")
    print(f"\nThe solution is:\n{x}")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)

    ### input file structure ###
    # n: the size of the matrix
    # the matrix
    # the right hand side vector
