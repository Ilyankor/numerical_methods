import numpy as np
from utils.utils import is_2d, is_square
from linear.matrices import transpose

# test matrices
A = [
    [0, complex(1,-1), 2],
    [1, 2, 3],
    [8, 8, 4],
    [9, 0, 5]
]

# print(is_2d(A))
# print(is_square(A))

print(np.array(A))
print(transpose(A))