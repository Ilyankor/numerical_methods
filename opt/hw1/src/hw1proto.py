import time
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable

# FORTRAN TO DOs:
# Check precision? 12 decimals vs 16
# Check speed? blas on matrix vector operations

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


# check answer to 6

# objective function
def obj(x:np.ndarray) -> np.float64:
    return np.linalg.norm(x, 1.0)

# get information
m, n = 10, 1000

with open(Path("./out/6/A.dat"), mode="rb") as f:
    data = np.fromfile(f, dtype=np.float64)
A = data.reshape((m, n), order='F')

with open(Path("./out/6/b.dat"), mode="rb") as f:
    b = np.fromfile(f, dtype=np.float64)

with open(Path("./out/6/x0.dat"), mode="rb") as f:
    x0 = np.fromfile(f, dtype=np.float64)

# # constraint
# cons = sci.optimize.LinearConstraint(A, b, b)

# #scipy answer
# result = sci.optimize.minimize(obj, x0, constraints=cons, options={"maxiter": 50})
# print(result)

# check answer to 7
# problem parameters
m, n = 20, 200
niter = 2500
p = 1.0e4

# open the information
with open(Path("./out/7/c.dat"), mode="rb") as f:
    c = np.fromfile(f, dtype=np.float64)

with open(Path("./out/7/A.dat"), mode="rb") as f:
    data = np.fromfile(f, dtype=np.float64)
A = data.reshape((m, n), order='F')

with open(Path("./out/7/b.dat"), mode="rb") as f:
    b = np.fromfile(f, dtype=np.float64)

with open(Path("./out/7/x0.dat"), mode="rb") as f:
    x0 = np.fromfile(f, dtype=np.float64)

with open(Path("./out/7/x.dat"), mode="rb") as f:
    data = np.fromfile(f, dtype=np.float64)
res = data.reshape((n, niter+1), order='F')

with open(Path("./out/7/lam.dat"), mode="rb") as f:
    data = np.fromfile(f, dtype=np.float64)
lam = data.reshape((m, niter+1), order='F')

# set up primal dual
x = np.zeros((n, niter+1))
l = np.zeros((m, niter+1))
x[:, 0] = x0
l[:, 0] = 0
xk = x0
lk = np.zeros(m)

at = np.transpose(A)

for i in range(niter):
    y = A @ x[:, i] - b
    M = (y > 0).astype(float)
    y = np.maximum(0, y)

    T_1 = c + at @ (M * (lk + p*y))
    T = np.linalg.norm(np.array([np.linalg.norm(T_1), np.linalg.norm(y)]))
    alpha = 1.0 / ((i+1) * T)

    xk = xk - alpha * T_1
    lk = lk + alpha * y

    x[:, i+1] = xk
    l[:, i+1] = lk

# precision check on Fortran

# # check objective function
# obj_py = np.array([np.dot(c, x[:, i]) for i in range(niter+1)])
# obj_fo = np.array([np.dot(c, res[:, i]) for i in range(niter+1)])

# # primal feasibility
# primal_py = np.array([np.max((A @ x[:, i]) - b) for i in range(niter+1)])
# primal_fo = np.array([np.max((A @ res[:, i]) - b) for i in range(niter+1)])

# # dual feasibility
# dual_py = np.array([np.min(l[:, i]) for i in range(niter+1)])
# dual_fo = np.array([np.min(lam[:, i]) for i in range(niter+1)])

# # complementary slackness
# slack_py = np.array([np.dot(l[:, i], np.maximum(0, (A @ x[:, i]) - b)) for i in range(niter+1)])
# slack_fo = np.array([np.dot(lam[:, i], np.maximum(0, (A @ res[:, i]) - b)) for i in range(niter+1)])


result = sci.optimize.linprog(c, A_ub = A, b_ub = b, bounds=None, method="highs-ipm")
