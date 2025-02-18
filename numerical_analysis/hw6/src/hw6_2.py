import numpy as np


def power(A:np.ndarray, q0:np.ndarray, tol:float, max_iter:int, track_iter:bool) -> dict:
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


# main function
def hw6_2_main(info:dict) -> None:

    while True:
        print("\nPart B")

        # prompt matrix selection
        q2_prompt = input("\nEnter an A1, A2, or A3 to select the matrix.\nType an E to exit to question selection.\n")

        # exit
        if q2_prompt.lower() == "e":
            break

        # matrix A_1
        elif q2_prompt.lower() == "a1":
            
            print("\nMatrix A_1\n")
            
            # power method
            result = power(info["A1"]["A"], info["A1"]["q0"], info["A1"]["tol"], info["A1"]["max_iter"], True)
            print(f"The eigenvalue with the largest modulus is {result["eigenvalue"]}.")
            print(f"The number of iterations it took was {result["iterations"]}.")

            # eigenvalues
            print("\nFor reference, the eigenvalues computed by NumPy are:")
            print(np.linalg.eigvals(info["A1"]["A"]))

        # matrix A_2
        elif q2_prompt.lower() == "a2":

            print("\nMatrix A_2\n")

            # power method
            result = power(info["A2"]["A"], info["A2"]["q0"], info["A2"]["tol"], info["A2"]["max_iter"], True)
            print(f"The eigenvalue with the largest modulus is {result["eigenvalue"]}.")
            print(f"The number of iterations it took was {result["iterations"]}.")

            # eigenvalues
            print("\nFor reference, the eigenvalues computed by NumPy are:")
            print(np.linalg.eigvals(info["A2"]["A"]))
        
        # matrix A_3
        elif q2_prompt.lower() == "a3":

            print("\nMatrix A_3\n")

            # power method
            result = power(info["A3"]["A"], info["A3"]["q0"], info["A3"]["tol"], info["A3"]["max_iter"], True)
            print(result["iterations"])

            # eigenvalues
            print("\nFor reference, the eigenvalues computed by NumPy are:")
            print(np.linalg.eigvals(info["A3"]["A"]))

        else:
            print("Entry not valid.")

    return None


# automatically exit if script is called directly
if __name__ == "__main__":
    exit(0)
