from numpy import array, ndarray
import timeit

def rk4_1(f, h: float, n: int, t0: float, y0: list[float], var) -> list[float]:
    t = t0
    y = y0
    k = len(y)
    for i in range(n):
        s1 = [h * f(t, y, var)[j] for j in range(k)]

        Z = [y[j] + 0.5*s1[j] for j in range(k)]
        s2 = [h * f(t+0.5*h, Z, var)[j] for j in range(k)]

        Z = [y[j] + 0.5*s2[j] for j in range(k)]
        s3 = [h * f(t+0.5*h, Z, var)[j] for j in range(k)]

        Z = [y[j] + s3[j] for j in range(k)]
        s4 = [h * f(t+h, Z, var)[j] for j in range(k)]

        y = [y[j] + (s1[j] + 2*s2[j] + 2*s3[j] + s4[j])/6 for j in range(k)]
        t = t + h
    return y

def rk4_2(f, h: float, n: int, t0: float, y0: ndarray, var) -> ndarray:
    t = t0
    y = y0
    for i in range(n):
        s1 = h * f(t, y, var)
        s2 = h * f((t+0.5*h), (y+0.5*s1), var)
        s3 = h * f((t+0.5*h), (y+0.5*s2), var)
        s4 = h * f(t+h, (y+s3), var)

        y = y + (s1 + 2*s2 + 2*s3 + s4)/6
        t = t + h
    return y

def test_func_1(t: float, y: list[float], var) -> list[float]:
    U = [y[1], y[2], 3*y[0] - y[3] + t**2, 2*y[0] + 4*y[3] + t**3 + 1]
    return U

def test_func_2(t: float, y: ndarray, var) -> ndarray:
    U = array([y[1], y[2], 3*y[0] - y[3] + t**2, 2*y[0] + 4*y[3] + t**3 + 1])
    return U


# benchmark
# setup1 = '''
# from __main__ import rk4_1
# from __main__ import test_func_1'''

# setup2 = '''
# from __main__ import rk4_2
# from __main__ import test_func_2
# from numpy import array'''

# print(min(timeit.Timer("rk4_1(test_func_1, 0.001, 1000, 0.0, [0.2, 3.1, -1.2, 0.0], 0)", setup = setup1).repeat(repeat=7, number=1000))/1000 * 10**(6))
# print(min(timeit.Timer("rk4_2(test_func_2, 0.001, 1000, 0.0, array([0.2, 3.1, -1.2, 0.0]), 0)", setup = setup2).repeat(repeat=7, number=1000))/1000 * 10**(6))