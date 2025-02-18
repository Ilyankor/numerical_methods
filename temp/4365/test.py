import numpy as np

def f(U):
    return np.array([
        5.0*U[0] + 4.0*U[1] - U[0]*U[2],
        U[0] + 4.0*U[1] - U[1]*U[2],
        U[0]**2 + U[1]**2 - 89
    ])

def df(U):
    return np.array([
        [5.0 - U[2], 4.0, -1.0*U[0]],
        [1.0, 4.0 - U[2], -1.0*U[1]],
        [2.0*U[0], 2.0*U[1], 0]
    ])

def newton(g, dg, U0, tol, nmax):
    U = np.array(U0)
    for i in range(nmax):
        Unew = U - np.linalg.solve(dg(U), g(U))

        if np.linalg.norm(Unew-U,2) < tol:
            return Unew
        U = Unew
    return U

points = [
    [8, -5, 2],
    [-8, 5, 2],
    [9, 3, 7],
    [-9, -3, 7]]

for i in range(len(points)):
    x = newton(f, df, points[i], 10**(-15), 20)
    print(x)