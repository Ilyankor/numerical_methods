import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable

def step_size(k:int) -> np.float64:
    return 1.0/(k + 1.0)


def subgradient(x:np.ndarray) -> np.ndarray:
    return np.sign(x)


def projection(A:np.ndarray, x:np.ndarray, b:np.ndarray, at:np.ndarray, aat:np.ndarray) -> np.ndarray:

    y = np.linalg.matmul(A, x) - b
    z = np.linalg.solve(aat, y)
    x = x - np.linalg.matmul(at, z)

    return x


def proj_subgradient(A:np.ndarray, x0:np.ndarray, b:np.ndarray, n_iter:int, projection:Callable, step_size:Callable, subgradient:Callable) -> np.ndarray:
    _, n = A.shape

    at = np.transpose(A)
    aat = np.linalg.matmul(A, at)

    # project x if not in convex set
    x0 = projection(A, x0, b, at, aat)

    x = np.zeros((n, n_iter+1))
    x[:, 0] = x0
    xk = x0

    for i in range(n_iter):
        xk = xk - step_size(i) * subgradient(xk)
        xk = projection(A, xk, b, at, aat)
        x[:, i+1] = xk
    
    return x


# A_path = Path("./out/6/A.dat")
# x0_path = Path("./out/6/x0.dat")
# b_path = Path("./out/6/b.dat")
# res_path = Path("./out/6/x.dat")

# m, n = 10, 1000
# niter = 3000
# with open(A_path, mode="rb") as f:
#     A = np.reshape(np.fromfile(f), (m,n), order="F")
# with open(x0_path, mode="rb") as f:
#     x0 = np.fromfile(f, dtype=np.float64)
# with open(b_path, mode="rb") as f:
#     b = np.fromfile(f, dtype=np.float64)
# with open(res_path, mode="rb") as f:
#     res = np.reshape(np.fromfile(f, dtype=np.float64), (n,niter+1), order="F")

# start = time.perf_counter()

# x = proj_subgradient(A, x0, b, niter, projection=projection, step_size=step_size, subgradient=subgradient)

# end = time.perf_counter()
# print(end-start)

# l1_norm = np.sum(np.abs(x), axis=0)
# fk_best = np.minimum.accumulate(l1_norm)



# print(np.linalg.norm(np.linalg.matmul(A, x[:, -4])-b))
# res -17
# print(fk_best[-18:])


