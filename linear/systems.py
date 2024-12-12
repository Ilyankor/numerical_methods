import numpy as np
from utils.utils import is_square

# LU factorization
def lu(A:np.ndarray, pivot) -> tuple[np.ndarray]:
    if is_square(A):
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
    raise ValueError("Matrix must be square")

# LU factorization by Gaussian elimination
def lu_factor(A:np.ndarray) -> tuple[np.ndarray]:
    # size of system
    n = A.shape[0]

    # initialize variables
    L = np.eye(n, dtype=float)
    U = A.astype(float)

    # Gaussian elimination method
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, :] = U[i, :] - L[i, k] * U[k, :]

    return L, U


# backward and forward substitution given LU factorization
def lu_solve(L:np.ndarray, U:np.ndarray, b:np.ndarray) -> np.ndarray:
    # size of system
    n = b.shape[0]

    # initialize variables
    y = np.zeros(n)
    x = np.zeros(n)
    b = b.astype(dtype=float)

    # foward substitution Ly = b
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - np.sum(L[i, 0:i] * y[0:i])
    
    # backward substitution Ux = y
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (1.0 / U[i, i]) * (y[i] - np.sum(U[i, (i+1):] * x[(i+1):]))

    return x

