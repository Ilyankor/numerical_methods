import numpy as np

def is_2d(A:np.ndarray) -> bool:
    """
    Checks if a matrix is 2D.
    """

    A = np.array(A) # convert to NumPy array
    if A.ndim == 2:
        return True
    raise ValueError("Matrix must be 2 dimensional.")

def is_square(A:np.ndarray) -> bool:
    """
    Checks if a matrix is square.
    """

    A = np.array(A) # convert to NumPy array
    if is_2d(A): # check if the array is 2D
        if A.shape[0] == A.shape[1]:
            return True
        raise ValueError("Matrix must be square.")

if __name__ == "__main__":
    exit(0)
