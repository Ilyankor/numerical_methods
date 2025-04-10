import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


def euler(
        f: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
        y0: NDArray[np.float64], 
        t0: float,
        tn: float,
        n: int,
        **kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    """
    Explicit Euler method.
    Solves y'(t) = f(t,y) with y(t0) = y0 on [t0, tn].
    
    Args:
        f: The derivative.
        y0: Initial value.
        t0: Initial time.
        tn: Final time.
        n: Number of steps.
        **kwargs: Additional parameters for f.

    Returns:
        t: The solution times.
        y: The solution.
    """

    # initialize the variables
    t = np.linspace(t0, tn, num=(n+1))
    h = t[1] - t[0]

    k = y0.shape[0]
    y = np.zeros((n+1, k))
    y[0, :] = y0

    # explicit Euler method
    for i in range(n):
        y[i+1, :] = y[i, :] + h * f(t[i], y[i, :], **kwargs)
    
    return t, y


def rk4(
        f: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
        y0: NDArray[np.float64], 
        t0: float,
        tn: float,
        n: int,
        **kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    """
    Runge-Kutta 4th order method.
    Solves y'(t) = f(t,y) with y(t0) = y0 on [t0, tn].
    
    Args:
        f: The derivative.
        y0: Initial value.
        t0: Initial time.
        tn: Final time.
        n: Number of steps.
        **kwargs: Additional parameters for f.

    Returns:
        t: The solution times.
        y: The solution.
    """

    # initialize the variables
    t = np.linspace(t0, tn, num=(n+1))
    h = t[1] - t[0]

    k = y0.shape[0]
    y = np.zeros((n+1, k))
    y[0, :] = y0

    # Rungge-Kutta 4th order
    for i in range(n):
        s1 = f(t[i], y[i, :], **kwargs)
        s2 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * s1, **kwargs)
        s3 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * s2, **kwargs)
        s4 = f(t[i] + h, y[i, :] + h * s3, **kwargs)

        y[i+1, :] = y[i, :] + (h / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
    
    return t, y


def heun(
        f: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
        y0: NDArray[np.float64], 
        t0: float,
        tn: float,
        n: int,
        **kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    """
    Heun's method.
    Solves y'(t) = f(t,y) with y(t0) = y0 on [t0, tn].
    
    Args:
        f: The derivative.
        y0: Initial value.
        t0: Initial time.
        tn: Final time.
        n: Number of steps.
        **kwargs: Additional parameters for f.

    Returns:
        t: The solution times.
        y: The solution.
    """

    # initialize the variables
    t = np.linspace(t0, tn, num=(n+1))
    h = t[1] - t[0]

    k = y0.shape[0]
    y = np.zeros((n+1, k))
    y[0, :] = y0

    # Heun's method
    for i in range(n):
        s1 = f(t[i], y[i, :], **kwargs)
        s2 = f(t[i] + h, y[i, :] + h * s1, **kwargs)
        y[i+1, :] = y[i, :] + 0.5 * h * (s1 + s2)
    
    return t, y


if __name__ == "__main__":
    exit(0)
