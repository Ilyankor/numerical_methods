import numpy as np
from linear.matrices import norm

# test matrices
A = [
    [-3, complex(5, 2), 7],
    [2, 6, 4],
    [0, 2, 8],
]

print(norm(A, "2"))

print(np.linalg.norm(A, 2))