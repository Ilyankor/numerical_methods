import numpy as np
from tqdm import tqdm


# construct a Hilbert matrix
def hilbert(n:int) -> np.ndarray:
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = 1 / (i + j + 1)
                A[j, i] = 1 / (i + j + 1)
    return A


# gradient method with preconditioner
def gradient_method(A:np.ndarray, b:np.ndarray, x0:np.ndarray, P_inv:np.ndarray, tol:float, max_iter:int, track_iter:bool) -> np.ndarray:
    # initialize method
    x = x0 # solution
    r = b - np.linalg.matmul(A, x) # residual
    crit_denom = np.linalg.norm(r) # stopping criteria denominator

    # gradient method
    for i in tqdm(range(1, max_iter+1)):
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


# main function
def hw4_2_main(n:int, b:np.ndarray, x0:np.ndarray, tol:float, max_iter:int) -> None:
    
    # eigenvalues
    print("Verify all the eigenvalues are positive:\n")
    A = hilbert(n)
    eigen = np.linalg.eigvals(A)
    print(eigen)

    while True:
        # prompt part selection
        q2_prompt = input("\nEnter an A or B to select the part.\nType an E to exit to question selection.\n")

        # exit
        if q2_prompt.lower() == "e":
            break

        # part a
        elif q2_prompt.lower() == "a":

            print("\nPart A\n")

            # preconditioner P^-1 = D^-1
            D_inv = np.diag(1 / np.diag(A))
            
            # gradient method
            result = gradient_method(A, b, x0, D_inv, tol, max_iter, True)
            print(f"\nThe computed solution is {result["solution"]}.")
            print(f"The number of iterations it took was {result["iterations"]}.")

        # part b
        elif q2_prompt.lower() == "b":

            print("\nPart B\n")

            # no preconditioner P^-1 = I
            I = np.eye(n)

            # gradient method
            result = gradient_method(A, b, x0, I, tol, max_iter, True)
            print("\n", result)

            # condition number
            K = np.max(eigen) / np.min(eigen)
            print(f"The condition number of A is {K}.")

        else:
            print("Entry not valid.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)

    # input file structure:
    # dimension of Hilbert matrix
    # rhs
    # initial guess x0
    # tolerance
    # maximum iterations
