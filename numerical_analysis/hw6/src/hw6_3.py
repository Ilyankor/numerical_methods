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


# power method with shift
def power_shift(A:np.ndarray, mu:float, q0:np.ndarray, tol:float, max_iter:int, track_iter:bool) -> dict:
    # initialization
    n = A.shape[0] # size of A
    M = A - mu*np.eye(n) # shifted matrix
    P, L, U = lu_pivot_factor(M) # PLU factorization of M
    q = q0 / np.linalg.norm(q0) # normalize initial guess

    # power method with shift
    for i in range(1, max_iter+1):
        z = lu_solve(L, U, np.linalg.matmul(P, q)) # eigenvector
        q = z / np.linalg.norm(z) # normalize eigenvector
        Aq = np.linalg.matmul(A, q) # saved calculation A*q
        s = np.vdot(q, Aq) # eigenvalue

        r = np.linalg.norm(Aq - s*q) # residual
        if r < tol:
            break
    
    # output the results
    if r > tol:
        return {"iterations": "\nThe power method with shift did not converge within the maximum number of iterations."}
    elif track_iter == True:
        return {"eigenvector": q, "eigenvalue": s, "iterations": i}
    else:
        return {"eigenvector": q, "eigenvalue": s}


# construct a Wilkinson matrix of size 2n+1
def wilkinson(n:int) -> np.ndarray:
    off_diag = np.ones(2*n)
    diag = np.abs(np.linspace(-n, n, num=(2*n + 1), endpoint=True))
    W = np.diag(diag) + np.diag(off_diag, k=-1) + np.diag(off_diag, k=1)
    return W


# main function
def hw6_3_main(n:int, mu:float, q0:np.ndarray, tol:float, max_iter:int) -> None:

    # construct the matrix
    A = wilkinson(n)

    # power method with shift
    result = power_shift(A, mu, q0, tol, max_iter, True)

    # output the result
    print(f"The negative eigenvalue with the largest modulus is {result["eigenvalue"]}.")
    print(f"The number of iterations it took was {result["iterations"]}.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
