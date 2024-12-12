import numpy as np
from utils.utils import is_2d

# transpose of a matrix
def transpose(A:np.ndarray) -> np.ndarray:
    A = np.array(A) # convert to NumPy array
    if is_2d(A): # check if the array is 2D
        T = A.dtype # get the data type of A
        m, n = A.shape # A is m x n
        B = np.zeros((n, m), dtype=T) # initialize transpose
        if T == complex: # conjugate transpose
            for i in range(n):
                for j in range(m):
                    B[i,j] = np.conjugate(A[j,i])
            return B
        for i in range(n): # real transpose
            for j in range(m):
                B[i,j] = A[j,i]
        return B
    raise ValueError("Matrix must be 2 dimensional.")

# determinant
# norms
# inverse
