import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


def bisection(
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        zerotol: float = 1e-14,
        method: str = "normal"
    ) -> float:

    # check for sign change
    if f(a) * f(b) >= 0:
        raise ValueError("Function may not change sign within the interval.")
    
    # ensure a < b
    if a > b:
        a, b = b, a

    # initialize
    f_a = f(a)
    max_iter = int(np.ceil(np.log2((b - a) / tol)))

    for _ in range(max_iter):

        mid = 0.5 * (a + b)
        f_mid = f(mid)
        
        match method:
            case "normal":
                if np.abs(f_mid) < zerotol:
                    return mid
                
                loc_flag = f_mid * f_a < 0

            case "lazy":
                loc_flag = f_mid * f_a <= 0

            case _:
                raise ValueError(f"Method {method} is not valid.")

        if loc_flag:    # root is in [a, mid]
            b = mid
        else:           # root is in [mid, b]
            a = mid
            f_a = f_mid

    return mid


def find_zeros(
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int = 100,
        **kwargs
    ) -> NDArray[np.float64]:

    # ensure a < b
    if a > b:
        a, b = b, a

    # subintervals
    points = np.linspace(a, b, num=n+1)

    a_vals = points[:n]
    f_a = f(a_vals)

    b_vals = points[1:n+1]
    f_b = f(b_vals)

    # initialize
    zeros = []

    for i in range(n):
        if f_a[i] * f_b[i] < 0:
            zeros.append(bisection(f, a_vals[i], b_vals[i], **kwargs))
    
    return np.array(zeros)


def newton(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        max_iter: int = 1000,
        tol: float = 1e-12,
        zerotol: float = 1e-14,
        stop: str = "absolute",
    ) -> float:

    # initialize
    x = x0

    for _ in range(max_iter):
        
        dfx = df(x)
        if np.abs(dfx) <= zerotol:
            raise ValueError(f"Zero derivative encountered at {x}.")
        
        x_new = x - f(x) / dfx
        
        match stop:
            case "absolute":
                if np.abs(x_new - x) < tol:
                    return x_new
                
            case "relative":
                if np.abs((x_new - x) / x) < tol:
                    return x_new
            
            case "zero":
                if np.abs(f(x_new)) < tol:
                    return x_new
            
            case _:
                raise ValueError(f"Stopping criterion {stop} is not a valid.")
        
        x = x_new

    raise RuntimeError(f"Newton's method did not converge. Last value is {x}.")


def modified_newton(
        f: Callable[[float], float],
        df: Callable[[float], float],
        d2f: Callable[[float], float],
        x0: float,
        max_iter: int = 1000,
        tol: float = 1e-12,
        zerotol: float = 1e-14,
        stop: str = "absolute",
    ) -> float:

    # initialize
    x = x0

    for _ in range(max_iter):
        
        f_x = f(x)
        dfx = df(x)

        diff = dfx**2 - f_x * d2f(x)
        if np.abs(diff) <= zerotol:
            return x
        
        x_new = x - f_x * dfx / diff
        
        match stop:
            case "absolute":
                if np.abs(x_new - x) < tol:
                    return x_new
                
            case "relative":
                if np.abs((x_new - x) / x) < tol:
                    return x_new
            
            case "zero":
                if np.abs(f(x_new)) < tol:
                    return x_new
            
            case _:
                raise ValueError(f"Stopping criterion {stop} is not a valid.")
        
        x = x_new

    raise RuntimeError(f"Newton's method did not converge. Last value is {x}.")


def secant(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        max_iter: int = 1000,
        tol: float = 1e-12,
        zerotol: float = 1e-14,
        stop: str = "absolute",
    ) -> float:

    # check initial guesses
    if np.abs(x1 - x0) < zerotol:
        raise ValueError(f"Initial guesses {x0} and {x1} must be distinct.")
    
    # initialize
    x_old = x0
    x_cur = x1

    for _ in range(max_iter):

        diff = f(x_cur) - f(x_old)
        if np.abs(diff) <= zerotol:
            raise ValueError(f"Division by zero encountered at {x_cur}.")
        
        x_new = x_cur - f(x_cur) * (x_cur - x_old) / diff
        
        match stop:
            case "absolute":
                if np.abs(x_new - x_cur) < tol:
                    return x_new
                
            case "relative":
                if np.abs((x_new - x_cur) / x_cur) < tol:
                    return x_new
            
            case "zero":
                if np.abs(f(x_new)) < tol:
                    return x_new
            
            case _:
                raise ValueError(f"Stopping criterion {stop} is not a valid.")
        
        x_old = x_cur
        x_cur = x_new

    raise RuntimeError(f"Secant method did not converge. Last value is {x_cur}.")


def regula_falsi(
        f: Callable[[float], float],
        a: float,
        b: float,
        max_iter: int = 1000,
        tol: float = 1e-12,
        zerotol: float = 1e-14,
    ) -> float:

    # check for sign change
    if f(a) * f(b) >= 0:
        raise ValueError("Function may not change sign within the interval.")
    
    # ensure a < b
    if a > b:
        a, b = b, a

    # initialize
    f_a = f(a)
    f_b = f(b)

    for _ in range(max_iter):

        diff = f_b - f_a
        if np.abs(diff) < zerotol:
            raise ValueError("Division by zero encountered.")
        
        p = a - f_a * (b - a) / diff
        f_p = f(p)
        
        if np.abs(f_p) < zerotol or b - a < tol:
            return p
        
        if f_p * f_a < 0:   # root is in [a, p]
            b = p
            f_b = f_p
        else:               # root is in [p, b]
            a = p
            f_a = f_p

    raise RuntimeError(f"Regula falsi did not converge. Last value is {p}.")
