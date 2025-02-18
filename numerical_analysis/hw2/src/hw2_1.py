import numpy as np

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


# homework 2 question 1 
def hw2_1_main(A:np.ndarray, eps_loc:tuple[int], prompt:str) -> None:

    # x_ex = [1, 1, 1]
    if prompt.lower() == "a":

        # initialize variables
        x_ex = np.array([1, 1, 1], dtype=float)
        eps = 10.0**(-1.0 * np.arange(10))
        original_eps_loc_val = A[eps_loc]

        # solve Ax = b
        for i in range(10):
            # copy the matrix
            A_copy = A.copy()

            # edit the matrix for the epsilon
            A_copy[eps_loc] = original_eps_loc_val + eps[i]

            # determine b
            b = np.linalg.matmul(A_copy, x_ex)

            # LU factorize
            L, U = lu_factor(A_copy)

            # solve for x
            x = lu_solve(L, U, b)

            print(x)
        
        print("\nAs verified, the computed solution is not affected by rounding errors.")

    # x_ex = [log(5/2), 1, 1]
    elif prompt.lower() == "b":

        # initialize variables
        x_ex = np.array([np.log(5/2), 1, 1])
        eps = (1/3) * 10.0**(-1.0 * np.arange(10))
        original_eps_loc_val = A[eps_loc]

        # solve Ax = b
        for i in range(10):
            # copy the matrix
            A_copy = A.copy()

            # edit the matrix for the epsilon
            A_copy[eps_loc] = original_eps_loc_val + eps[i]

            # determine b
            b = np.linalg.matmul(A_copy, x_ex)

            # LU factorize
            L, U = lu_factor(A_copy)

            # solve for x
            x = lu_solve(L, U, b)

            # relative error
            print(np.linalg.norm(x_ex - x) / np.linalg.norm(x_ex))
    
        print("\nAs epsilon gets closer to 0, the relative error increases.")
    
    else:
        print("Invalid prompt input")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)

    ### input file structure ###
    # n: the size of the matrix
    # i, j: the location of the epsilon
    # the matrix
    # the right hand side vector
