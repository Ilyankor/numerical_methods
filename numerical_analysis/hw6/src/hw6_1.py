import numpy as np


# construct a Hilbert matrix
def hilbert(n:int) -> np.ndarray:
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = 1 / (i + j + 1)
                A[j, i] = 1 / (i + j + 1)
    return A


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
    

# main function
def hw6_1_main(n:int, b:np.ndarray, x0:np.ndarray, tol:float, max_iter:int) -> None:

    # construct the matrix
    A = hilbert(n)

    # conjugate gradient method
    result = conjugate_gradient_method(A, b, x0, tol, max_iter, True)

    # output the result
    print(f"The computed solution is {result["solution"]}.")
    print(f"The number of iterations it took was {result["iterations"]}.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
