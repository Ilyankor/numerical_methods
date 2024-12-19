import numpy as np
from ..linear.systems import lu_pivot_factor, lu_solve

# power method
def power_method(A:np.ndarray, q0:np.ndarray=None, tol:float=1e-12, max_iter:int=10000, track_iter:bool=False) -> dict:
    # input validation
    A = np.array(A) # covert to NumPy array

    # normalize initial eigenvalue guess
    q = q0 / np.linalg.norm(q0)

    # power method
    for i in range(1, max_iter+1):
        z = np.linalg.matmul(A, q) 
        q = z / np.linalg.norm(z) # eigenvector
        Aq = np.linalg.matmul(A, q) # save calculation A*q
        v = np.vdot(q, Aq) # eigenvalue

        r = np.linalg.norm(Aq - v*q) # residual
        if r < tol:
            break
    
    # output the results
    if r > tol:
        return {"iterations": f"\nThe power method did not converge within {max_iter} iterations."}
    elif track_iter == True:
        return {"eigenvector": q, "eigenvalue": v, "iterations": i}
    else:
        return {"eigenvector": q, "eigenvalue": v}

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

# QR iteration method
def qr_iteration(A:np.ndarray, tol:float, max_iter:int) -> tuple:
    T_old = A
    for i in range(1, max_iter+1):
        Q, R = np.linalg.qr(T_old)
        T_new = np.linalg.matmul(R, Q)
        if np.linalg.norm(T_new - T_old) < tol:
            break
        T_old = T_new
    
    return np.diag(T_new), i

# # Jacobi method
# def jacobi_eig(A:np.ndarray):
#     return None

# # Lanczos method
# def lanczos(A:np.ndarray):
#     return None
