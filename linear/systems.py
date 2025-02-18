import numpy as np
from utils.utils import is_square

# backward and forward substitution given LU factorization
def lu_solve(L:np.ndarray, U:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    ;....``             .. 
    """
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

# gradient method with preconditioner
def gradient_method(A:np.ndarray, b:np.ndarray, x0:np.ndarray, P_inv:np.ndarray, tol:float, max_iter:int, track_iter:bool) -> np.ndarray:
    # initialize method
    x = x0 # solution
    r = b - np.linalg.matmul(A, x) # residual
    crit_denom = np.linalg.norm(r) # stopping criteria denominator

    # gradient method
    for i in range(1, max_iter+1):
        z = np.linalg.matmul(P_inv, r) # direction
        Az = np.linalg.matmul(A, z) # save an intermediary computation A*z
        alpha = np.sum(z * r) / np.sum(z * Az) # length

        x = x + alpha * z # new solution
        r = r - alpha * Az # new residual

        crit = np.linalg.norm(r) / crit_denom # stopping criteria

        # stopping criteria
        if crit < tol:
            break
    
    # output the results
    if crit > tol:
        return "\nThe gradient method did not converge within the maximum number of iterations."
    elif track_iter == True:
        return {"solution": x, "iterations": i}
    else:
        return {"solution": x}

# conjugate gradient method with no preconditioner
def conjugate_gradient_method(A:np.ndarray, b:np.ndarray, x0:np.ndarray, tol:float, max_iter:int, track_iter:bool) -> dict:
    # initialize method
    x = x0 # solution
    r = b - np.linalg.matmul(A, x) # residual
    p = r # search direction
    crit_denom = np.linalg.norm(r) # stopping criteria denominator

    # conjugate gradient method
    for i in range(1, max_iter+1):
        Ap = np.linalg.matmul(A, p) # save an intermediary computation A*p
        alpha = np.dot(p, r) / np.dot(p, Ap) # length

        x = x + alpha * p # new solution
        r = r - alpha * Ap # new residual

        crit = np.linalg.norm(r) / crit_denom # stopping criteria

        # check stopping criteria
        if crit < tol:
            break

        beta = np.dot(Ap, r) / np.dot(Ap, p) # length
        p = r - beta * p # new search direction
    
    # output the results
    if crit > tol:
        return "\nThe conjugate gradient method did not converge within the maximum number of iterations."
    elif track_iter == True:
        return {"solution": x, "iterations": i}
    else:
        return {"solution": x}
