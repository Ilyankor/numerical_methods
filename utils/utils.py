import numpy as np

# check if the matrix is 2D
def is_2d(A:np.ndarray) -> bool:
    A = np.array(A) # covert to NumPy array
    if A.ndim == 2:
        return True
    raise ValueError("Matrix must be 2 dimensional.")

# check if the matrix is square
def is_square(A:np.ndarray) -> bool:
    A = np.array(A) # covert to NumPy array
    if is_2d(A): # check if the array is 2D
        if A.shape[0] == A.shape[1]:
            return True
        raise ValueError("Matrix must be square.")
