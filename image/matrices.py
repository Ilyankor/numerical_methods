import numpy as np
from utils.utils import is_2d, is_square

def norm(A:np.ndarray, kind:str="F") -> float:
    """
    Computes the norm of a matrix A.
    
    Args:
        "1": 1-norm (column sum)
        "2": 2-norm (spectral)
        "inf": inf-norm (row sum)
        "F": Frobenius norm
    """

    if is_2d(A):
        A = np.array(A) # convert to NumPy array
        match kind:
            case "1":
                return np.max([np.sum(np.abs(A[:, j])) for j in range(A.shape[1])])
            case "2":
                return np.max(np.linalg.svd(A).S)
            case "inf":
                return np.max([np.sum(np.abs(A[i, :])) for i in range(A.shape[0])])
            case "F":
                return np.sqrt(np.sum(np.abs(A)**2))

# def condition_number(A:np.ndarray) -> float:
#     """
#     Computed the condition number induced by the 2-norm of a matrix A.
#     """
#     if is_square(A):
#         singular_values = np.linalg.svd(A).S
#         return np.max(singular_values) / np.min(singular_values)




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

# spectral radius
def spectral_radius(A:np.ndarray) -> float:
    return np.max(np.absolute(np.linalg.eigvals(A)))

# determinant
# norms
# inverse

# sum without pth root
def norm_sum(image: np.ndarray, p: float) -> float:
    m = image.shape[0]
    n = image.shape[1]
    sum_power = (1.0/(m * n)) * np.sum(np.float_power(image, p))
    return sum_power

# L^p norm
def lp_norm(image: np.ndarray, p: float) -> float:
    norm = norm_sum(image, p) ** (1.0/p)
    return norm

# forward difference
def forward_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[1:m, :] - array[0:m-1, :]
    diff = (m - 2) * diff[1:m-1, 1:n-1]
    return diff

# backward difference
def backward_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[1:m, :] - array[0:m-1, :]
    diff = (m - 2) * diff[0:m-2, 1:n-1]
    return diff

# centered difference
def centered_difference(array: np.ndarray) -> np.ndarray:
    m = array.shape[0]
    n = array.shape[1]

    diff = array[2:m, :] - array[0:m-2, :]
    diff = 0.5 * (m - 2) * diff[:, 1:n-1]
    return diff

# norm of gradient vector
def gradient(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    grad = np.sqrt(dx ** 2 + dy ** 2)
    return grad

# W(1,p) norm
def sobolev_norm(image: np.ndarray, p: float, difference) -> float:
    m = image.shape[0]
    n = image.shape[1]

    # replication convention
    padded = np.pad(image, 1, 'edge')

    norm = (norm_sum(image, p) + (1.0/(m * n)) * np.sum(
        gradient(difference(padded), np.transpose(difference(np.transpose(
        padded)))) ** p)) ** (1.0/p)
    return norm


if __name__ == "__main__":
    exit(0)
